import logging
import os
import time

import torch
import wandb

from torchprofile import profile_macs

from src.network.network import Net
from src.util.dataset import load_CleanNoisyPairDataset
from src.util.denoise_eval import validate

from src.training.train_distributed import apply_gradient_allreduce
from src.util.util import find_max_epoch

from torchinfo import summary

def get_network(exp_path, network, network_config, teacher: bool, num_gpus: int, rank: int, group_name: str, log: dict):
    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)

    # Create tensorboard logger.
    log_directory = os.path.join(log["directory"], exp_path)

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            if teacher:
                # the directory for teacher model should exist
                raise ValueError("Teacher model directory does not exist")
            else:
                os.makedirs(ckpt_directory)
                os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # predefine model
    net = Net(network, network_config).cuda()

    model_summary = summary(net, input_size=(1, 1, int(16000)), depth=1,
                                  col_names=["input_size",
                                       "output_size",
                                       "num_params",
                                       "params_percent",
                                       "mult_adds"])

    wandb_model_summary = {
        'input_size': 16000,
        'total_params': model_summary.total_params,
        'total_param_bytes': model_summary.total_param_bytes,
        'total_mult_adds': model_summary.total_mult_adds,
        'total_output_bytes': model_summary.total_output_bytes,
        'summary': {f"{n:03d} & {layer.depth}-{layer.depth_index}: {layer.class_name}": layer.num_params for n, layer in
                    enumerate(model_summary.summary_list) if layer.depth == 1 or layer.depth == 2},
    }

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    return net, wandb_model_summary, ckpt_directory

def load_checkpoint(ckpt_iter: [str, int], net, optimizer, ckpt_directory, scaler=None, wandb_config=None, rank: int = 0):
    run_id = None

    time0 = time.time()
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)

    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])

            if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (
                ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')

            if rank == 0 and wandb_config is not None:
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

        if rank == 0 and wandb_config is not None:
            run_id = wandb.util.generate_id()

            # start a new wandb run to track this script
            wandb.init(
                project="Slim-Speech-Enhancement",
                config=wandb_config,
                id=run_id
            )

    return ckpt_iter, run_id, time0

def get_model_macs(model, inputs=torch.rand(1, 1, 16000)) -> int:
    inputs = inputs.cuda()
    return profile_macs(model, inputs)


def get_num_parameters(model, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

def add_validation_to_log(log_file, model, n_iter, test_factor=1, testset_path="/datasets/test_set/synthetic/no_reverb", prefix=""):
    if "Test/pesq_wb" in log_file:
        return log_file

    validation_result = validate(model,
                                 testset_path,
                                 test_factor=test_factor)
    pesq_wb = validation_result['Test/pesq_wb'] / validation_result['Test/count']
    pesq_nb = validation_result['Test/pesq_nb'] / validation_result['Test/count']
    stoi = validation_result['Test/stoi'] / validation_result['Test/count']

    log_file[f"Test/{prefix}pesq_wb"] = pesq_wb
    log_file[f"Test/{prefix}pesq_nb"] = pesq_nb
    log_file[f"Test/{prefix}stoi"] = stoi

    print("iteration: {} \tpesq_wb: {:.4f} \tpesq_nb: {:.4f} \tstoi: {:.4f}".format(
        n_iter, pesq_wb, pesq_nb, stoi), flush=True)

    return log_file

def add_prune_to_log(log_file, prune_groups, prune_channels, params_pruned, model, n_iter, importance_min_dict):
    total_channel_count = 0
    for group in prune_groups:
        log_file[f"Prune/model_channel_count/{group.name}"] = group.n_channels
        total_channel_count += group.n_channels
    log_file[f"Prune/model_channel_count"] = total_channel_count

    if prune_channels is not None:
        log_file[f"Prune/prune_count"] = len(prune_channels)

        for group in prune_groups:
            prune_count = sum([1 for prune in prune_channels if prune["group"] == group])
            log_file[f"Prune/prune_count/{group.name}"] = prune_count
            log_file[f"Prune/prune_parameters/{group.name}"] = prune_count * group.channel_importances()["n_parameters"]

    if params_pruned is not None:
        log_file[f"Prune/prune_parameters"] = params_pruned

    if importance_min_dict is not None:
        total_min_importances = sum([value for value in importance_min_dict.values()])
        for key, value in importance_min_dict.items():
            log_file[f"Prune/importance_min/{key}"] = value/total_min_importances
            log_file["Prune/importance_min"] = total_min_importances

    log_file[f"Prune/model_parameter_count"] = get_num_parameters(model)
    log_file[f"Prune/model_macs"] = get_model_macs(model)

    return log_file

def add_train_to_log(log_file, loss, grad_norm, output_dic, optimizer):
    log_file["Train/Train-Loss"] = loss
    log_file["Train/Gradient-Norm"] = grad_norm

    log_file["Train/learning-rate"] = optimizer.param_groups[0]["lr"]

    for key, value in output_dic.items():
        log_file[f"Train/Train-Loss/{key}"] = value

    return log_file

def print_prune_stats(model, prune_groups, prune_channels, params_pruned, prune_step):
    print(f"Pruned {len(prune_channels)} channels at step {prune_step} with {params_pruned} parameters")
    for group in prune_groups:
        prune_count = sum([1 for prune in prune_channels if prune["group"] == group])
        if prune_count > 0:
            print(f"\t{group.name} pruned {prune_count} channels")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")


def load_model_dict(ckpt_directory, log, lr, teacher_exp_path, teacher_network, teacher_network_config):
    # try loading the checkpoint
    try:
        files = os.listdir(ckpt_directory)
        if len(files) == 0:
            print(f"No checkpoints present in folder: {ckpt_directory}")
            raise Exception("No checkpoints present")
        max_ckpt = find_max_epoch(ckpt_directory)
        # load checkpoint file
        model_path = os.path.join(ckpt_directory, '{}.pkl'.format(max_ckpt))
        print(f"Loading checkpoint {model_path}")
        checkpoint = torch.load(model_path)

        model = checkpoint["model"]  # As the physical model is pruned we save the model not the state dict
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        n_iter = checkpoint["n_iter"]
        run_id = checkpoint["run_id"]
        prune_step = checkpoint["prune_step"]

        print("Checkpoint loaded")

    except:  # or load from the base model
        print(f"laoding teacher model from: {teacher_exp_path}")

        model, _, teacher_ckpt_directory = get_network(teacher_exp_path, teacher_network,
                                                       teacher_network_config, True,
                                                       num_gpus=1, rank=0, group_name="", log=log)

        load_checkpoint("max", model, None, teacher_ckpt_directory, scaler=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        n_iter = -1
        prune_step = 1

        run_id = wandb.util.generate_id()

        print("Base model loaded")
    return model, n_iter, optimizer, prune_step, run_id


def get_state(n_iter, batch_size, training_samples, grad_samples, pruning_repeats, update_interval, steps_per_valid,
              steps_per_ckpt, steps_per_calibrate):
    """Calculate the state of the training process

    first from iter 0-grad_samples/batch_size we accumulate gradients
    at grad_samples/batch_size we prune
    repeat this for pruning_repeats untill we are at grad_samples * pruning_repeats / batch_size
    than do training for training_samples * pruning_repeats/batch_size
    and repeat everything
    """

    assert training_samples % batch_size == 0, "training_samples must be a multiple of batch_size"
    assert grad_samples % batch_size == 0, "grad_samples must be a multiple of batch_size"
    assert (grad_samples + training_samples) * pruning_repeats % batch_size == 0, "(grad_samples + training_samples) * pruning_repeats must be a multiple of batch_size"

    assert steps_per_valid % pruning_repeats == 0, "steps_per_valid must be a multiple of pruning_repeats"

    iters_per_step = ((grad_samples + training_samples) * pruning_repeats // batch_size)
    step = n_iter // iters_per_step
    n_iter_folded = n_iter % ((grad_samples + training_samples) * pruning_repeats // batch_size)

    prune_step = step * pruning_repeats + min(n_iter_folded // (grad_samples // batch_size), pruning_repeats - 1)

    pruning = n_iter_folded < grad_samples * pruning_repeats // batch_size
    training = not pruning

    go_prune = n_iter_folded % (grad_samples // batch_size) == (grad_samples // batch_size) - 1 and pruning
    training_done = n_iter_folded == ((grad_samples + training_samples) * pruning_repeats // batch_size) - 1

    if pruning:
        prune_samples = prune_step * grad_samples + n_iter_folded * batch_size % grad_samples
    else:
        prune_samples = prune_step * grad_samples + grad_samples

    training_samples = (prune_step // pruning_repeats) * training_samples * pruning_repeats + max(0,
                                                                                                  n_iter_folded * batch_size - grad_samples * pruning_repeats)

    return {
        'pruning': pruning,
        'training': training,
        'go_prune': go_prune,
        'training_done': training_done,
        'log': (n_iter_folded * batch_size) % update_interval == update_interval - batch_size,
        'valid': (prune_step) % steps_per_valid == steps_per_valid - 1 and (go_prune or training_done),
        'ckpt': (prune_step) % steps_per_ckpt == steps_per_ckpt - 1 and training_done,
        'calibrate': (prune_step) % steps_per_calibrate == 0 and n_iter_folded == 0,
        "prune_step": prune_step,
        "prune_samples": prune_samples,
        "prune_epoch": prune_samples / 60_000,
        "train_samples": training_samples,
        "train_epoch": training_samples / 60_000,
    }

def run_forward(model, loss_fn, root="/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge", batch_size=1, n_samples=1, shuffle=False, backward=True):
    if root == "/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge":
        logging.warning("[pruning/util.py:run_forward()] replace the root with your own DNS-Challenge dataset path")

    dataloader = load_CleanNoisyPairDataset(root=root, batch_size=batch_size, num_workers=0, crop_length_sec=10, shuffle=shuffle)
    total_losses = []
    for i, (clean, noisy, min_max, fileid) in enumerate(dataloader):
        if backward:
            loss, output_dic = loss_fn(model, (clean.cuda(), noisy.cuda()))
            total_losses.append(loss.item())
            loss.backward()
        else:
            with torch.no_grad():
                loss, output_dic = loss_fn(model, (clean.cuda(), noisy.cuda()))
                total_losses.append(loss.item())
        if (i * batch_size > n_samples):
            break

    return total_losses

@torch.no_grad()
def prune_parameter_and_grad(weight, keep_idxs, axis, optimizer=None):
    assert weight.shape[axis] > max(keep_idxs), f"{weight.shape[axis]} > {max(keep_idxs)}"
    keep_idxs = torch.LongTensor(keep_idxs).to(weight.device).contiguous()

    pruned_weight_data = torch.index_select(weight.data, axis, keep_idxs)
    weight.data = pruned_weight_data

    if weight.grad is not None:
        pruned_grad_data = torch.index_select(weight.grad.data, axis, keep_idxs)
        weight.grad.data = pruned_grad_data

    if optimizer is not None and len(optimizer.state) > 0:
        for p in optimizer.param_groups[0]["params"]:
            if id(p) == id(weight):
                exp_avg = optimizer.state[p]["exp_avg"]
                exp_avg.data = torch.index_select(exp_avg.data, axis, keep_idxs)
                assert exp_avg.shape == weight.shape, f"{exp_avg.shape} == {weight.shape}"

                exp_avg_sq = optimizer.state[p]["exp_avg_sq"]
                exp_avg_sq.data = torch.index_select(exp_avg_sq.data, axis, keep_idxs)
                assert exp_avg_sq.shape == weight.shape, f"{exp_avg_sq.shape} == {weight.shape}"
