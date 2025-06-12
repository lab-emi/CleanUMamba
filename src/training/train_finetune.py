# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

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
from src.util.util import LinearWarmupCosineDecay, loss_fn

from src.network.network import Net

from torchinfo import summary

# Enable cuDNN auto-tuner https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
torch.backends.cudnn.benchmark = True


def train(num_gpus, rank, group_name,
          exp_path, log, optimization, loss_config):

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

    time0 = time.time()
    # try loading the checkpoint
    try:
        files = os.listdir(ckpt_directory)
        if len(files) == 0:
            print("No checkpoints present")
            raise Exception("No checkpoints present")
        if f'{exp_config["checkpoint_nr"]}.pkl' not in files:
            print(f"File '{exp_config['checkpoint_nr']}.pkl' not in files")
            raise Exception("No checkpoints present")


        # load checkpoint file
        model_path = os.path.join(ckpt_directory, f'{exp_config["checkpoint_nr"]}.pkl')
        print(f"Loading checkpoint {model_path}")
        checkpoint = torch.load(model_path)

        print("extracting net")
        net = checkpoint["model"]  # As the physical model is pruned we save the model not the state dict

        print("creating optimizer")
        optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

        print("loading optimizer state dict")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'training_time_seconds' in checkpoint:
            time0 -= checkpoint['training_time_seconds']

        print("Checkpoint loaded")
    except:
        print("Cant load the checkpoint")
        return

    if rank == 0:
        ckpt_directory = log_directory + f"/checkpoints_{exp_config['checkpoint_nr']}_finetune"
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    wandb_config = dict(exp_config, **config)
    model_summary = summary(net, input_size=(1, 1, int(16000)), depth=1,
                            col_names=["input_size", "output_size",
                                       "num_params",
                                       "params_percent",
                                       "mult_adds"])
    wandb_config["model_summary"] = {
        'input_size': 16000,
        'total_params': model_summary.total_params,
        'total_param_bytes': model_summary.total_param_bytes,
        # 'total_mult_adds': model_summary.total_mult_adds if "mamba_v2" in network_config else get_model_macs(net),
        'total_mult_adds': get_model_macs(net),
        'total_output_bytes': model_summary.total_output_bytes,
        'summary': {f"{n:03d} & {layer.depth}-{layer.depth_index}: {layer.class_name}": layer.num_params for n, layer in
                    enumerate(model_summary.summary_list) if layer.depth == 1 or layer.depth == 2},
    }
    run_id = wandb.util.generate_id()
    wandb.init(
        project="Slim-Speech-Enhancement",
        config=wandb_config,
        id=run_id
    )

    # training
    n_iter = 0

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

    mrstftloss = MultiResolutionSTFTLoss(
        sc_lambda=0.5,
        mag_lambda=0.5,
        band="high",
        hop_sizes=[50, 120, 240],
        win_lengths=[240, 600, 1200],
        fft_sizes=[512, 1024, 2048]
    ).cuda()

    micro_step = 0
    reduced_loss = 0
    total_loss = 0


    while n_iter < optimization["n_iters"] + 1:
        # for each epoch
        for clean_audio, noisy_audio, min_max, _ in trainloader:

            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()

            # back-propagation
            if micro_step == 0:
                optimizer.zero_grad()
                reduced_loss = 0
                total_loss = 0

            X = (clean_audio, noisy_audio)

            loss, loss_dic = loss_fn(net, X, **loss_config, min_max=min_max, mrstftloss=mrstftloss)

            loss /= repeats

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
                            'model': net,
                            'optimizer_state_dict': optimizer.state_dict(),
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
    parser.add_argument('-e', '--experiment', type=str, default='configs/exp/finetune/DNS-CleanUMamba-Pruning_Finetuning_12_E8_200K.json',
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
