# Adapted from https://github.com/NVIDIA/CleanUNet/blob/main/train.py

# Which was adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

# from torch.utils.tensorboard import SummaryWriter
import wandb

import random

from torch.cuda.amp import GradScaler

from src.pruning.util import get_model_macs
from src.util.denoise_eval import validate

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from src.training.train_distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from src.util.dataset import load_CleanNoisyPairDataset
from src.util.stft_loss import MultiResolutionSTFTLoss
from src.util.util import rescale, find_max_epoch, print_size, LinearWarmupCosineDecay, loss_fn
from torch.profiler import profile, record_function, ProfilerActivity

from src.network.network import Net

from torchinfo import summary

# Enable cuDNN auto-tuner https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
torch.backends.cudnn.benchmark = True


def train(num_gpus, rank, group_name,
          exp_path, log, optimization, loss_config):
    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)

    # Create tensorboard logger.
    log_directory = os.path.join(log["directory"], exp_path)

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    trainloader = load_CleanNoisyPairDataset(**trainset_config,
                                             subset='training',
                                             batch_size=optimization["batch_size_per_gpu"],
                                             num_gpus=num_gpus,
                                             num_workers=4)
    print('Data loaded')

    # predefine model
    net = Net(network, network_config).cuda()

    if rank == 0:
        wandb_config = dict(exp_config, **config)

        print_size(net)
        model_summary = summary(net, input_size=(1, 1, int(16000)), depth=1,
                                col_names=["input_size",
                                           "output_size",
                                           "num_params",
                                           "params_percent",
                                           "mult_adds"])
        wandb_config["model_summary"] = {
            'input_size': 16000,
            'total_params': model_summary.total_params,
            'total_param_bytes': model_summary.total_param_bytes,
            'total_mult_adds': model_summary.total_mult_adds if "mamba_v2" in network_config else get_model_macs(net),
            'total_output_bytes': model_summary.total_output_bytes,
            'summary': {f"{n:03d} & {layer.depth}-{layer.depth_index}: {layer.class_name}": layer.num_params for n, layer in
                        enumerate(model_summary.summary_list) if layer.depth == 1 or layer.depth == 2},
        }

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    # Only weight decay 2D parameters following karpathy's nanoGPT https://github.com/Hannibal046/nanoRWKV/blob/main/modeling_gpt.py
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': optimization["weight_decay"]},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if rank == 0:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    if optimization["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"],
                                     betas=optimization["betas"], eps=optimization["eps"],
                                     fused=optimization["fused_adam"], weight_decay=optimization["weight_decay"])
    elif optimization["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(optim_groups, lr=optimization["learning_rate"],
                                      betas=optimization["betas"], eps=optimization["eps"],
                                     fused=optimization["fused_adam"])
    else:
        raise ValueError("Optimizer not supported")

    # define scaler for mixed precision training
    # Creates a GradScaler once at the beginning of training.
    autocast = optimization["autocast"] is not None and optimization["autocast"]
    if autocast:
        scaler = GradScaler()
        if rank == 0:
            print("Autocast and scaler enabled")

    run_id = None

    # load checkpoint
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
        print(f"using checkpoint: {ckpt_iter}")
    else:
        ckpt_iter = log["ckpt_iter"]

    if ckpt_iter >= 0:
        try:
            print("Trying to load old checkpoint")
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            print(f"checkpoint path: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if autocast and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (
                ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')

            if rank == 0:
                if 'run_id' in checkpoint:
                    run_id = checkpoint['run_id']
                    # Continue the existing run
                    wandb.init(
                        project="Slim-Speech-Enhancement",
                        id=run_id,
                        resume="must"
                    )
                else:
                    run_id = wandb.util.generate_id()
                    # start a new wandb run to track this script
                    wandb.init(
                        project="Slim-Speech-Enhancement",
                        config=wandb_config,
                        id=run_id
                    )
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

        if rank == 0:
            run_id = wandb.util.generate_id()

            # start a new wandb run to track this script
            wandb.init(
                project="Slim-Speech-Enhancement",
                config=wandb_config,
                id=run_id
            )

    # training
    n_iter = ckpt_iter + 1

    assert optimization["batch_size_total"] % (optimization["batch_size_per_gpu"] * num_gpus) == 0
    repeats = optimization["batch_size_total"] // (optimization["batch_size_per_gpu"] * num_gpus)

    # define learning rate scheduler and stft-loss
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=optimization["learning_rate"],
        n_iter=optimization["n_iters"],
        iteration=n_iter,
        divider=25,
        warmup_proportion=0.05,
        phase=('linear', 'cosine'),
    )

    if loss_config["stft_lambda"] > 0:
        mrstftloss = MultiResolutionSTFTLoss(**loss_config["stft_config"]).cuda()
    else:
        mrstftloss = None

    micro_step = 0
    reduced_loss = 0
    total_loss = 0

    while n_iter < optimization["n_iters"] + 1:
        # for each epoch
        for clean_audio, noisy_audio, min_max, _ in trainloader:

            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()

            # If you have a data augmentation function augment()
            # noise = noisy_audio - clean_audio
            # noise, clean_audio = augment((noise, clean_audio))
            # noisy_audio = noise + clean_audio

            # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            #     with record_function("model_inference"):

            # back-propagation
            if micro_step == 0:
                optimizer.zero_grad()
                reduced_loss = 0
                total_loss = 0

            X = (clean_audio, noisy_audio)

            loss, loss_dic = loss_fn(net, X, **loss_config, min_max=min_max, mrstftloss=mrstftloss) if not autocast else \
                torch.autocast(device_type="cuda")(loss_fn)(net, X, **loss_config, min_max=min_max,
                                                            mrstftloss=mrstftloss)

            loss /= repeats

            if autocast:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if num_gpus > 1:
                reduced_loss += reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss += loss.item()
            total_loss += loss.item()

            micro_step += 1
            if micro_step == repeats:
                micro_step = 0
            else:
                continue



            if autocast:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), optimization["clip_grad_norm_max"])
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), optimization["clip_grad_norm_max"])
                optimizer.step()

            scheduler.step()

            # output to log
            if n_iter % log["iters_per_valid"] == 0:
                print("iteration: {} \treduced loss: {:.7f} \tloss: {:.7f}".format(
                    n_iter, reduced_loss, total_loss), flush=True)

                if rank == 0:
                    # # save to tensorboard
                    # tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                    # tb.add_scalar("Train/Train-Reduced-Loss", reduced_loss, n_iter)
                    # tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                    # tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)
                    log_file = {
                        "n_iter": n_iter,
                        "n_epoch": (n_iter * optimization["batch_size_total"]) / trainloader.dataset.__len__(),
                        "train_set_seconds": n_iter * optimization["batch_size_total"] * trainset_config[
                            "crop_length_sec"],
                        "Train/Train-Loss": total_loss,
                        "Train/Train-Reduced-Loss": reduced_loss,
                        "Train/Gradient-Norm": grad_norm,
                        "Train/learning-rate": optimizer.param_groups[0]["lr"]
                    }
                    for key, value in loss_dic.items():
                        log_file["Train/Train-Loss/" + key] = value

                    if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0:
                        # Run validation
                        VCTK_DEMAND = "dataset" in trainset_config and trainset_config["dataset"] == "VCTK-DEMAND"
                        test_folder = trainset_config["root"] + ("test_set" if VCTK_DEMAND else "datasets/test_set/synthetic/no_reverb")
                        validation_result = validate(net, test_folder, VCTK_DEMAND=VCTK_DEMAND)

                        pesq_wb = validation_result['Test/pesq_wb'] / validation_result['Test/count']
                        pesq_nb = validation_result['Test/pesq_nb'] / validation_result['Test/count']
                        stoi = validation_result['Test/stoi'] / validation_result['Test/count']

                        print("iteration: {} \tpesq_wb: {:.4f} \tpesq_nb: {:.4f} \tstoi: {:.4f}".format(
                            n_iter, pesq_wb, pesq_nb, stoi), flush=True)

                        for key in validation_result.keys():
                            if key != 'Test/count':
                                log_entry = key
                                if VCTK_DEMAND:
                                    log_entry = log_entry.replace("Test/", "Test/VCTK-DEMAND/")
                                log_file[log_entry] = validation_result[key] / validation_result['Test/count']

                    # log metrics to wandb
                    wandb.log(log_file)

            # save checkpoint
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'iter': n_iter,
                            'run_id': run_id,
                            'network_config': network_config,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict() if autocast else None,
                            'training_time_seconds': int(time.time() - time0)},
                           os.path.join(ckpt_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1
            if n_iter >= optimization["n_iters"] + 1:
                break

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.json',
                        help='JSON file for training configuration')
    parser.add_argument('-e', '--experiment', type=str, default='configs/exp/DNS-large-high.json',
                        help='JSON file for model configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    global config
    config = json.loads(data)
    with open(args.experiment) as f:
        data = f.read()
    exp_config = json.loads(data)

    train_config = config["train_config"]  # training parameters
    global dist_config
    dist_config = config["dist_config"]  # to initialize distributed training
    global network
    network = exp_config["network"]
    global network_config
    network_config = exp_config["network_config"]  # to define network
    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    exp_path = exp_config["exp_path"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    if num_gpus != train_config["optimization"]["n_gpus"]:
        raise Exception(f"Number of GPUs in config file does not match detected GPUs, {num_gpus} != {train_config['optimization']['n_gpus']}")

    torch.autograd.set_detect_anomaly(True)  # will crash the program if there is an error in the gradient with mixed precision training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, exp_path, **train_config)
