import json
import os
import argparse
import torch
import torch.nn as nn
import wandb

from src.pruning.importance import get_prune_channels
from src.pruning.layerwise_calibration import calibrator
from src.pruning.pruninggroup import CleanUMambaPrunableChannels
from src.pruning.util import print_prune_stats, add_prune_to_log, add_train_to_log, add_validation_to_log, \
    load_model_dict, get_state, prune_parameter_and_grad
from src.util.dataset import load_CleanNoisyPairDataset
from src.util.stft_loss import MultiResolutionSTFTLoss
from src.util.util import loss_fn, LinearWarmupCosineDecay


def pruning_pipeline(teacher_exp_path, log, optimization, **kwargs):
    training_samples = exp_config["pruning_config"]["training_samples"]
    pruning_grad_samples = exp_config["pruning_config"]["pruning_grad_samples"]
    pruning_repeats = exp_config["pruning_config"].get("pruning_repeats", 1)

    total_loss_samples = training_samples + pruning_grad_samples
    batch_size = optimization["batch_size_per_gpu"]
    prune_steps = exp_config["pruning_config"]["prune_steps"]
    steps_per_valid = exp_config["pruning_config"]["steps_per_valid"]
    steps_per_ckpt = exp_config["pruning_config"]["steps_per_ckpt"]
    calibration = exp_config["pruning_config"].get("calibration", False)
    steps_per_calibrate = exp_config["pruning_config"].get("steps_per_calibration", 60)
    update_interval = pruning_grad_samples
    lr = exp_config["pruning_config"]["lr"]
    lr_divider = exp_config["pruning_config"].get("lr_divider", 25)

    n_prune_channels_per_iter = exp_config["pruning_config"]["n_prune_channels_per_iter"]
    perc_prune_channels_per_iter = exp_config["pruning_config"].get("perc_prune_channels_per_iter", None)
    max_prune_importance_per_iter = exp_config["pruning_config"].get("max_prune_importance_per_iter", None)
    min_channels_per_group = exp_config["pruning_config"].get("min_channels_per_group", 16)
    clip_grad_norm_max = exp_config["pruning_config"].get("clip_grad_norm_max", 1e9)
    importance_metric = exp_config["pruning_config"]["importance_metric"]

    # Get shared ckpt_directory ready
    log_directory = os.path.join(log["directory"], exp_path)
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if not os.path.isdir(ckpt_directory):
        os.makedirs(ckpt_directory)
        os.chmod(ckpt_directory, 0o775)
    print("ckpt_directory: ", ckpt_directory, flush=True)

    model, n_iter, optimizer, prune_step, run_id = load_model_dict(ckpt_directory, log, lr, teacher_exp_path,
                                                                   teacher_network, teacher_network_config)

    prune_groups = CleanUMambaPrunableChannels(model, False)

    wandb_config = dict(exp_config["pruning_config"],
    **{
        "training_samples": training_samples,
        "pruning_grad_samples": pruning_grad_samples,
        "batch_size": batch_size,
        "prune_steps": prune_steps,
        "n_prune_channels_per_iter": n_prune_channels_per_iter,
        "perc_prune_channels_per_iter": perc_prune_channels_per_iter,
        "min_channels_per_group": min_channels_per_group,
        "importance_metric": importance_metric,
        "clip_grad_norm_max": clip_grad_norm_max,
        "lr": lr,
        "base_model": "CleanUMamba",
        "base_checkpoint": "/DNS-CleanUMamba-3N-high-2/checkpoint/250000.pkl",
    })

    wandb.init(
        project="Slim-Speech-Enhancement-Pruning",
        config=wandb_config,
        id=run_id,
        resume="allow"
    )

    dataloader = load_CleanNoisyPairDataset(**trainset_config,
                                            subset='training',
                                            batch_size=optimization["batch_size_per_gpu"],
                                            num_gpus=1,
                                            num_workers=4)

    # define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=lr,
        n_iter=training_samples / batch_size * pruning_repeats,
        iteration=n_iter,
        divider=lr_divider,
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

    calibrator_container = calibrator(exp_config["pruning_config"].get("calibration_ema", 1)) if calibration else None

    new_calibration = False
    while prune_step < prune_steps + 1:
        for i, (clean, noisy, min_max, fileid) in enumerate(dataloader):
            n_iter += 1
            # print(n_iter)
            st = get_state(n_iter, batch_size, training_samples, pruning_grad_samples, pruning_repeats, update_interval,
                           steps_per_valid, steps_per_ckpt, steps_per_calibrate)

            if st["calibrate"] and calibration:
                calibrator_container.gather(model, prune_groups, loss_fn, importance_metric, trainset_config["root"], batch_size, pruning_grad_samples, n_iter)
                new_calibration = True

            # Run forward pass
            loss, output_dic = loss_fn(model, (clean.cuda(), noisy.cuda()), mrstftloss=mrstftloss)

            if st["pruning"]:
                # normalize the loss over the amount of samples
                loss = loss * batch_size / pruning_grad_samples

            # Run backward pass accumulating ontop of the previous gradients
            loss.backward()

            # just keep accumulating gradients until we are ready to prune
            if st["pruning"] and not st["go_prune"]:
                continue

            log_file = {
                "prune_step": st["prune_step"],
                "total_n_iter": n_iter,
                "total_n_epoch": (n_iter * batch_size) / 60000,
                "total_set_seconds": n_iter * batch_size * 10,
                "prune_n_samples": st["prune_samples"],
                "prune_n_epoch": st["prune_epoch"],
                "prune_set_seconds": st["prune_samples"] * 10,
                "train_n_samples": st["train_samples"],
                "train_n_epoch": st["train_epoch"],
                "train_set_seconds": st["train_samples"] * 10,
            }

            if new_calibration:
                new_calibration = False
                log_file = calibrator_container.log(log_file)

            if st["go_prune"]:
                # Determine importance_cutoff
                prune_channels, params_pruned, importance_min_dict = get_prune_channels(prune_groups,
                                                                                        importance_metric,
                                                                                        n_prune_channels_per_iter,
                                                                                        perc_prune_channels_per_iter,
                                                                                        min_channels_per_group,
                                                                                        max_prune_importance_per_iter,
                                                                                        calibrator_container)

                # Prune channels
                for group in prune_groups:
                    idxs = [prune["index"] for prune in prune_channels if prune["group"] == group]
                    group.prune(idxs, optimizer)

                # Reset gradients
                model.zero_grad()

                # Reset the loss to recompute the back prop kernels or whatever
                del loss

                # # Reset the opimizer Done: properly prune weigths and grads in place and prune adam exp_avg and exp_avg_sq
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                print_prune_stats(model, prune_groups, prune_channels, params_pruned, st["prune_step"])
                log_file = add_prune_to_log(log_file, prune_groups, prune_channels, params_pruned, model, n_iter,
                                            importance_min_dict)

                prune_step += 1
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_max)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                log_file = add_train_to_log(log_file, loss.item(), grad_norm, output_dic, optimizer)

                if st["log"]:
                    print(f"Trained Loss at iter {n_iter} with {i * batch_size} samples: {loss.item()}")

            if st["log"]:
                log_file["n_pruned"] = 18752 - sum([group.n_channels for group in prune_groups])

                if st["valid"]:
                    if st["pruning"]:
                        log_file = add_validation_to_log(log_file, model, n_iter, testset_path=trainset_config[
                                                                                                   "root"] + "datasets/test_set/synthetic/no_reverb",
                                                         prefix="BeforeFinetune/")
                    else:
                        log_file = add_prune_to_log(log_file, prune_groups, None, None, model,
                                                    n_iter, None)
                        log_file = add_validation_to_log(log_file, model, n_iter, testset_path=trainset_config[
                                                                                                   "root"] + "datasets/test_set/synthetic/no_reverb")
                wandb.log(log_file)

            go_prune = False

            # Train until total_loss_samples
            if not st["training_done"]:
                continue

            # Save checkpoint if applicable
            if st["ckpt"]:
                checkpoint_name = '{}.pkl'.format(st["prune_step"])
                torch.save({
                    'model': model,
                    'optimizer': optimizer.state_dict(),
                    'n_iter': n_iter,
                    'prune_step': st["prune_step"],
                    'run_id': run_id,
                }, os.path.join(ckpt_directory, checkpoint_name))
                print(f"Saved checkpoint at iter {n_iter}")

                if "Test/stoi" in log_file and log_file['Test/stoi'] < 0.9:
                    print("Stopping early due to low STOI")
                    return

            if st["prune_step"] == prune_steps or (
                    "Prune/model_channel_count" in log_file and log_file["Prune/model_channel_count"] < 1000):
                return
            prune_step = st["prune_step"]


def test_toy_prune_example():
    # read more into this: https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/

    example_data = torch.randn(1, 4, 128)
    module = nn.Conv1d(4, 4, 3, bias=False)

    loss = torch.sum(module(example_data))
    loss.backward()

    del loss

    print(module.weight)
    prune_parameter_and_grad(module.weight, [1, 2], 0, optimizer=None)
    module.out_channels = 2
    print(module.weight)

    loss = torch.sum(module(example_data))
    loss.backward()


if __name__ == "__main__":
    # test_CleanUMamba_pruningV2()
    # test_toy_prune_example()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.json',
                        help='JSON file for training configuration')
    parser.add_argument('-t', '--teacher', type=str, default='configs/exp/models/DNS-CleanUMamba-base.json',
                        help='JSON file for base model model configuration')
    parser.add_argument('-e', '--experiment', type=str,
                        default='configs/exp/pruning/DNS-CleanUMamba-Pruning5NoFT_FilterParamTaylorCalibration.json',
                        help='JSON file for model configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    global config
    config = json.loads(data)

    with open(args.experiment) as f:
        data = f.read()
    exp_config = json.loads(data)

    with open(args.teacher) as f:
        data = f.read()
    teacher_config = json.loads(data)

    global teacher_network
    teacher_network = teacher_config["network"]
    global teacher_network_config
    teacher_network_config = teacher_config["network_config"]  # to define teacher network

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset
    exp_path = exp_config["exp_path"]
    teacher_exp_path = teacher_config["exp_path"]

    train_config = config["train_config"]  # training parameters
    pruning_pipeline(teacher_exp_path, **train_config)
