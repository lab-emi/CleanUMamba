# adapted from https://github.com/NVIDIA/CleanUNet/blob/main/util.py

# Original Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import numpy as np
from math import cos, pi, floor, sin
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.util.stft_loss import MultiResolutionSTFTLoss


def flatten(v):
    return [x for y in v for x in y]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path):
    """
    Find latest checkpoint
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            number = f[:-4]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch


def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])

        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True, end="; ")

        if keyword is not None:
            keyword_parameters = [p for name, p in net.named_parameters() if p.requires_grad and keyword in name]
            params = sum([np.prod(p.size()) for p in keyword_parameters])
            print("{} Parameters: {:.6f}M".format(
                keyword, params / 1e6), flush=True, end="; ")

        print(" ")


def print_small_weight(checkpoint, th):
    for name, tensor in checkpoint['model_state_dict'].items():
        if name.split('.')[-1] == 'weight':
            print(name)
            print(tensor.shape)
            mean_distance = torch.mean(torch.abs(tensor))
            print(torch.isclose(tensor, torch.zeros_like(tensor), atol=mean_distance * 0.01, rtol=0).sum())


####################### lr scheduler: Linear Warmup then Cosine Decay #############################

# Adapted from https://github.com/rosinality/vq-vae-2-pytorch

# Original Copyright 2019 Kim Seonghyeon
#  MIT License (https://opensource.org/licenses/MIT)


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cosine(start, end, proportion):
    cos_val = cos(pi * proportion) + 1
    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, cur_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = cur_iter

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class LinearWarmupCosineDecay:
    def __init__(
            self,
            optimizer,
            lr_max,
            n_iter,
            iteration=0,
            divider=25,
            warmup_proportion=0.3,
            phase=('linear', 'cosine'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {'linear': anneal_linear, 'cosine': anneal_cosine}

        cur_iter_phase1 = iteration
        cur_iter_phase2 = max(0, iteration - phase1)
        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, cur_iter_phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, cur_iter_phase2, phase_map[phase[1]]),
        ]

        if iteration < phase1:
            self.phase = 0
        else:
            self.phase = 1

    def step(self):
        lr = self.lr_phase[self.phase].step()

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            self.phase = 0

        return lr


####################### model util #############################

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)


@torch.no_grad()
def sampling(net, noisy_audio, split_sampling=False, block_size=1600):
    """
    Perform denoising (forward) step
    net (nn.Module): network
    noisy_audio (torch.Tensor): noisy audio (B, C, L)
    """

    if not split_sampling:
        with torch.no_grad():
            return net(noisy_audio)

    if noisy_audio.dim() == 2:
        noisy_audio = noisy_audio.unsqueeze(1)

    denoised_audio = torch.zeros_like(noisy_audio)
    state = None
    with torch.no_grad():
        for i in tqdm(range(0, noisy_audio.shape[2], block_size)):
            end = min(denoised_audio.shape[2], i + block_size)
            noisy_audio_block = noisy_audio[:, :, i:end]
            denoised_audio_block = net(noisy_audio_block)

            if denoised_audio_block.shape[1] == 1:  # if the 1 channel waveform is generated by the model
                denoised_audio[:, :, i:end] = denoised_audio_block
            else:  # if mu law class probabilities are predicted
                denoised_audio[:, :, i:end] = torch.argmax(denoised_audio_block, dim=1).unsqueeze(1)

    return denoised_audio


def loss_fn(net, X,
            cross_entropy=None,
            ell_p=1,
            ell_p_lambda=1,
            stft_lambda=1,
            mrstftloss=MultiResolutionSTFTLoss(
                sc_lambda=0.5,
                mag_lambda=0.5,
                band="high",
                hop_sizes=[50, 120, 240],
                win_lengths=[240, 600, 1200],
                fft_sizes=[512, 1024, 2048]
            ).cuda(),
            kd_p=1,
            min_max=(-1, 1),
            teacher_net=None,
            student_teacher_adapter_layers = None,
            **kwargs
            ):
    """
    Loss function following CleanUNet

    Parameters:
    net: network
    X: training data pair (clean audio, noisy_audio)
    ell_p: \ell_p norm (1 or 2) of the AE loss
    ell_p_lambda: factor of the AE loss
    stft_lambda: factor of the STFT loss
    mrstftloss: multi-resolution STFT loss function

    Returns:
    loss: value of objective function
    output_dic: values of each component of loss
    """

    assert type(X) == tuple and len(X) == 2

    clean_audio, noisy_audio = X
    B, C, L = clean_audio.shape
    output_dic = {}
    loss = 0.0

    # AE loss
    # noisy_audio = noisy_audio.transpose(1, 2)
    if teacher_net is None:
        denoised_audio = net(noisy_audio)
    else:
        denoised_audio, skip_connections = net(noisy_audio, return_skip_connections=True)

        # Compute the knowledge distillation loss following
        # "Understanding the Role of the Projector in Knowledge Distillation" by Roy Miles, Krystian Mikolajczyk
        # https://arxiv.org/abs/2303.11098
        # https://github.com/roymiles/Simple-Recipe-Distillation/blob/31e8477cfdd6bb6ec7cdc332915fd4496b6de7b6/imagenet/torchdistill/losses/single.py#L150

        with torch.no_grad():
            _, teacher_skip_connections = teacher_net(noisy_audio, return_skip_connections=True)

        # Run the student hidden state through the projection + BN and the teacher hidden state through the BN
        student_skip_connections = [ad["bn_s"](ad["embed"](con)) for ad, con in zip(student_teacher_adapter_layers, skip_connections)]
        teacher_skip_connections = [ad["bn_t"](con) for ad, con in zip(student_teacher_adapter_layers, teacher_skip_connections)]

        kd_losses = []
        for f_s_norm, f_t_norm in zip(student_skip_connections, teacher_skip_connections):
            c_diff = f_s_norm - f_t_norm
            c_diff = torch.abs(c_diff)
            c_diff = c_diff.pow(4.0)

            kd_losses.append(torch.log(c_diff.sum()) * kd_p)


        # kd_losses = [F.l1_loss(st, tc) for st, tc in zip(st_skip_connections, teacher_skip_connections)]

        output_dic["kd_losses"] = [l.data for l in kd_losses]
        kd_loss = torch.mean(torch.stack(kd_losses))
        loss += kd_loss
        output_dic["kd_loss"] = kd_loss.data


    # Quantization scales the noisy audio to the full range, so we need to rescale it back
    # min_val, max_val = min_max
    # min_val = torch.tensor(min_val).cuda()
    # max_val = torch.tensor(max_val).cuda()
    # min_val = torch.squeeze(min_val, 1)
    # max_val = torch.squeeze(max_val, 1)
    # denoised_audio = (denoised_audio + 1) * (max_val - min_val + 1e-6) / 2 + min_val

    # denoised_audio = denoised_audio.transpose(1, 2)

    ae_loss = 0
    ce_loss = 0
    if cross_entropy:
        logits = denoised_audio.squeeze(0).permute(1, 0)
        y = clean_audio.view(-1)
        ce_loss = nn.functional.cross_entropy(logits, y)
        output_dic["cross_entropy"] = ce_loss.data
    elif ell_p == 2:
        ae_loss = nn.MSELoss()(denoised_audio, clean_audio)
    elif ell_p == 1:
        ae_loss = F.l1_loss(denoised_audio, clean_audio)
    else:
        raise NotImplementedError

    loss += ce_loss
    loss += ae_loss * ell_p_lambda
    output_dic["reconstruct"] = ae_loss.data * ell_p_lambda

    if stft_lambda > 0:
        sc_loss, mag_loss = mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
        loss += (sc_loss + mag_loss) * stft_lambda
        output_dic["stft_sc"] = sc_loss.data * stft_lambda
        output_dic["stft_mag"] = mag_loss.data * stft_lambda

    return loss, output_dic
