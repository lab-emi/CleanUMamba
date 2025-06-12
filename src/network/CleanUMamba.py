# CleanUNet architecture
import copy
import math
import time
from functools import partial

import numpy as np
import torch
import torch.quantization
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights

from mamba_ssm.utils.generation import InferenceParams
from torchinfo import summary
import itertools

# from src.network.S4.MambaS4 import create_block_mamba_s4 # Will only be imported if mamba_s4 = true
from src.network.layers import Activation

from src.util.util import weight_scaling_init

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



class CleanUMamba(nn.Module):
    """ CleanUNet architecture. """

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 encoder_groups=1,
                 bypass_channels=0,
                 glu_activation="Sigmoid",
                 tsfm_n_layers=3,
                 tsfm_n_head=8,
                 tsfm_d_model=512,
                 tsfm_d_inner=2048,
                 fused_add_norm = False,
                 use_fast_path = False,
                 rms_norm = False,
                 mamba_s4=False,
                 LSTM=False,
                 mamba_v2=False,
                 residual_projection=False,
                 norm_epsilon: float = 1e-5,
                 normalize_input = True,
                 device=None,
                 dtype=None,
                 ):

        """
        Parameters:
        channels_input (int):   input channels
        channels_output (int):  output channels
        channels_H (int):       middle channels H that controls capacity
        max_H (int):            maximum H
        encoder_n_layers (int): number of encoder/decoder layers D
        kernel_size (int):      kernel size K
        stride (int):           stride S
        tsfm_n_layers (int):    number of self attention blocks N
        tsfm_n_head (int):      number of heads in each self attention block
        tsfm_d_model (int):     d_model of self attention
        tsfm_d_inner (int):     d_inner of self attention
        device (torch.device):  device to use
        dtype (torch.dtype):    dtype to use
        """

        super(CleanUMamba, self).__init__()

        assert glu_activation in ["Sigmoid", "ReLU", "SiLU", "GELU"], f"glu_activation={glu_activation} not supported"

        factory_kwargs = {"device": device, "dtype": dtype}

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        self.residual_projection = residual_projection

        self.normalize_input = normalize_input
        self.dtype = dtype

        # encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if self.residual_projection:
            self.residual_projection_layers = nn.ModuleList()

        for i in range(encoder_n_layers):
            ec_groups = encoder_groups[i] if isinstance(encoder_groups, list) else encoder_groups
            bp_channels = bypass_channels[i] if isinstance(bypass_channels, list) else bypass_channels

            encoder_i = nn.Sequential(
                nn.Conv1d(channels_input, channels_H, kernel_size, stride,groups=ec_groups if i > 0 else 1, **factory_kwargs),
                nn.ReLU(),
                nn.Conv1d(channels_H, bp_channels + (channels_H - bp_channels) * 2, 1, **factory_kwargs),
                Activation(glu_activation, bp_channels)
            )
            self.encoder.append(encoder_i)
            channels_input = channels_H

            if self.residual_projection:
                self.residual_projection_layers.append(nn.Conv1d(channels_H, channels_H, 1, **factory_kwargs))

            # Decoder
            decoder_i = nn.Sequential(
                nn.Conv1d(channels_H, bp_channels + (channels_H - bp_channels) * 2, 1, **factory_kwargs),
                Activation(glu_activation, bp_channels),
                nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride, **factory_kwargs),
            )

            if i > 0: # Add ReLU for all but the final layer
                decoder_i.append(nn.ReLU())

            self.decoder.insert(0, decoder_i)

            channels_output = channels_H

            # double H but keep below max_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)

        # self attention block
        self.tsfm_conv1 = nn.Conv1d(channels_output, tsfm_d_model, kernel_size=1, **factory_kwargs)

        ssm_cfg = {
            "d_state": tsfm_d_model // tsfm_n_head,
            "d_conv": 4,
            "expand": tsfm_d_inner // tsfm_d_model,
        }
        if mamba_v2:
            ssm_cfg["layer"] = "Mamba2"
            ssm_cfg["headdim"] = tsfm_d_model // tsfm_n_head
            ssm_cfg["use_mem_eff_path"] = False
        else:
            ssm_cfg["use_fast_path"] = use_fast_path


        self.rms_norm = rms_norm
        self.residual_in_fp32 = True
        self.fused_add_norm = fused_add_norm

        self.LSTM = LSTM
        if LSTM:
            self.tsfm_Mamba_layers = nn.LSTM(bidirectional=False,
                                             num_layers=tsfm_n_layers,
                                             hidden_size=tsfm_d_model,
                                             input_size=tsfm_d_model)
            self.norm_f = nn.Identity()
        else:
            cb = None

            if mamba_s4:
                from src.network.S4.MambaS4 import create_block_mamba_s4
                cb = create_block_mamba_s4
            else:
                cb = create_block

            self.tsfm_Mamba_layers = nn.ModuleList(
                [
                    cb(
                        tsfm_d_model,
#                        d_intermediate=0,  # No extra MLP layer (for mamaba 2)
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=self.residual_in_fp32,
                        fused_add_norm=self.fused_add_norm,
                        layer_idx=i,
                        **factory_kwargs,
                    )
                    for i in range(tsfm_n_layers)
                ]
            )
            self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
                tsfm_d_model, eps=norm_epsilon, **factory_kwargs
            )

        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, channels_output, kernel_size=1, **factory_kwargs)

        # weight scaling initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(layer)

        self.apply(
            partial(
                _init_weights,
                n_layer=tsfm_n_layers,
            )
        )

        # Streaming values
        self.total_time = 0
        self.cat_time = 0
        self.frames = 0
        self.input_std = 0
        self.pending = torch.zeros(self.channels_input, 0, dtype=self.dtype, device=device)
        self.frame_length = self.valid_length(1)
        self.inference_params = None
        self.encoder_decoder_state = {}


    def pad_signal(self, input):
        """padding zeroes to x so that denoised audio has the same length"""
        L = self.valid_length(input.shape[-1])
        x = F.pad(input, (0, L - input.shape[-1]))
        return x

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """

        D, K, S = self.encoder_n_layers, self.kernel_size, self.stride

        for _ in range(D):
            if length < K:
                length = 1
            else:
                length = 1 + np.ceil((length - K) / S)

        for _ in range(D):
            length = (length - 1) * S + K

        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.encoder_n_layers

    def forward(self, noisy_audio, return_skip_connections=False):
        # (B, L) -> (B, C, L)
        if len(noisy_audio.shape) == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B, C, L = noisy_audio.shape
        assert C == 1

        # normalization and padding
        if self.normalize_input:
            std = noisy_audio.std(dim=2, keepdim=True) + 1e-3
            noisy_audio /= std
        x = self.pad_signal(noisy_audio)

        # encoder
        skip_connections = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)

        if hasattr(self, 'residual_projection') and self.residual_projection:
            for i, residual_projection_layer in enumerate(self.residual_projection_layers):
                skip_connections[i] = residual_projection_layer(skip_connections[i])

        skip_connections = skip_connections[::-1]

        x = self.tsfm_conv1(x)  # C 1024 -> 512

        if hasattr(self, 'LSTM') and self.LSTM:
            x = x.permute(2, 0, 1)
            x, _ = self.tsfm_Mamba_layers(x)
            tsfm_out = x.permute(1, 2, 0)

        else:
            hidden_states = x.permute(0, 2, 1)

            # Mamab MixerModel
            residual = None
            for layer in self.tsfm_Mamba_layers:
                hidden_states, residual = layer(hidden_states, residual, inference_params=None)

            if not hasattr(self, 'fused_add_norm') or not self.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else x
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

            tsfm_out = hidden_states.permute(0, 2, 1)

        x = self.tsfm_conv2(tsfm_out)  # C 512 -> 1024

        # decoder
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x = x + skip_i[:, :, :x.shape[-1]]
            x = upsampling_block(x)

        if self.normalize_input:
            x = x[:, :, :L] * std

        if return_skip_connections:
            skip_connections.append(tsfm_out)
            return x, skip_connections
        return x

    def reset_time_per_frame(self):
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self):
        if self.frames == 0:
            return 0
        return self.total_time / self.frames

    def allocate_inference_cache_layer(self, layer, batch_size, dtype=None):
        """allocate_inference_cache function for pruned mamba head since expand might no longer be an integer after pruning"""

        device = layer.out_proj.weight.device
        conv_dtype = layer.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, int(layer.d_model * layer.expand), layer.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = layer.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, int(layer.d_model * layer.expand), layer.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            # i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            i: self.allocate_inference_cache_layer(layer.mixer, batch_size, dtype=dtype)
            for i, layer in enumerate(self.tsfm_Mamba_layers)
        }

    def flush(self):
        """
        Flush remaining audio by padding it with zero and initialize the previous
        status. Call this when you have no more input and want to get back the last
        chunk of audio.
        """
        self.encoder_decoder_state = {}
        pending_length = self.pending.shape[1]
        padding = torch.zeros(self.channels_input, self.frame_length, device=self.pending.device, dtype=self.dtype)
        out = self.feed(padding)
        return out[:, :pending_length]

    @torch.no_grad()
    def feed(self, noisy_input):

        # Set up state cache for Mamba
        if self.inference_params is None:
            inf_cache = self.allocate_inference_cache(1, 1, self.dtype)
            self.inference_params = InferenceParams(
                max_seqlen=1,
                max_batch_size=1,
                key_value_memory_dict=inf_cache,
                seqlen_offset = 1,
            )

        begin = time.time()
        total_stride = self.total_stride

        if noisy_input.dim() != 2:
            raise ValueError("input should be two dimensional.")
        C, _ = noisy_input.shape
        if C != 1:
            raise ValueError(f"Expected 1 channel, got {C}")

        self.pending = torch.cat([self.pending, noisy_input], dim=1)
        denoised_frames = []
        while self.pending.shape[1] >= self.frame_length:
            self.frames += 1

            frame = self.pending[:, :self.frame_length]

            if self.normalize_input:
                self.input_std = (frame.std(dim=1, keepdim=True) + 1e-3) / self.frames + (1 - 1 / self.frames) * self.input_std
                frame = frame / self.input_std

            out = self._denoise_frame(frame)
            out = out[:, :total_stride]

            if self.normalize_input:
                out *= self.input_std

            denoised_frames.append(out)
            self.pending = self.pending[:, total_stride:]

        self.total_time += time.time() - begin

        if denoised_frames:
            out = torch.cat(denoised_frames, 1)
        else:
            out = torch.zeros(C, 0, device=noisy_input.device)
        return out

    def _denoise_frame(self, frame):

        x = frame.unsqueeze(1)

        skip_connections = []
        total_stride = self.total_stride
        for i, encode in enumerate(self.encoder):
            total_stride //= self.stride
            length = x.shape[2]
            if 1 == self.encoder_n_layers - 1:
                x = encode(x)
            else:
                prev = self.encoder_decoder_state.get(f"enc{i}", None)
                if prev is not None:
                    offset = length - self.kernel_size - self.stride * ((length - self.kernel_size) // self.stride - prev.shape[-1])
                    x = x[..., offset:]

                x = encode(x)

                if prev is not None:
                    x = torch.cat([prev, x], -1)

                self.encoder_decoder_state[f"enc{i}"] = x[..., total_stride:]

            skip_connections.append(x)

        x = self.tsfm_conv1(x)

        hidden_states = x.permute(0, 2, 1)

        residual = None
        for layer in self.tsfm_Mamba_layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=self.inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        x = hidden_states.permute(0, 2, 1)
        x = self.tsfm_conv2(x)

        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x += skip_i[..., :x.shape[-1]]
            x = upsampling_block[2](upsampling_block[1](upsampling_block[0](x)))

            # Get the previous samples and save the current samples for next iteration
            prev = self.encoder_decoder_state.get(f"dec{i}", None)
            self.encoder_decoder_state[f"dec{i}"] = x[..., -self.stride:] - upsampling_block[2].bias.view(-1, 1)

            x = x[..., :-self.stride]
            if prev is not None:
                x[..., :self.stride] += prev

            # Relu if this is not the last layer
            if i != self.encoder_n_layers - 1:
                x = upsampling_block[3](x)

        return x[0]

    def load_pruned_state_dict(self, pruned_state_dict):

        def prune_model_from_state_dict_shapes(module, local_state_dict, prefix=''):
            """Prunes the size dimensions of the module parameters based on the state dict shapes"""

            persistent_buffers = {k: v for k, v in module._buffers.items() if
                                  k not in module._non_persistent_buffers_set}
            local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
            local_state = {k: v for k, v in local_name_params if v is not None}

            for name, param in local_state.items():
                key = prefix + name
                if key in local_state_dict:

                    # print(f"{prefix}{module.__class__.__name__}: {key} - {param.shape} > {local_state_dict[key].shape}")
                    # prune the local state to the shape of the state_dict shape
                    param.data = copy.deepcopy(local_state_dict[key].data)

                    # update the in_channels and out_channels of the module
                    layers = [nn.LayerNorm, nn.Conv1d, nn.ConvTranspose1d, nn.Linear]
                    if any([isinstance(module, layer) for layer in layers]):
                        weight = getattr(module, 'weight') if hasattr(module, 'weight') else module.get()

                        # Update module variables
                        if isinstance(module, nn.LayerNorm):
                            module.normalized_shape = weight.shape
                        if isinstance(module, nn.Conv1d):
                            # self.module.groups = pruned_weights.shape[0]
                            module.in_channels = weight.shape[1]
                            module.out_channels = weight.shape[0]
                            if module.groups > 1:
                                module.groups = weight.shape[0]
                        if isinstance(module, nn.ConvTranspose1d):
                            module.in_channels = weight.shape[0]
                            module.out_channels = weight.shape[1]
                        if isinstance(module, nn.Linear):
                            module.in_features = weight.shape[1]
                            module.out_features = weight.shape[0]

                else:
                    print(f"Error cant find {key} in {module}")

            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    prune_model_from_state_dict_shapes(child, child_state_dict, child_prefix)

            if module.__class__.__name__ == "Mamba":
                module.dt_rank = module.dt_proj.in_features
                module.d_state = (module.x_proj.out_features - module.dt_rank) // 2
                module.d_inner = module.x_proj.in_features
                module.d_model = module.in_proj.in_features
                module.expand = module.d_inner / module.d_model

        prune_model_from_state_dict_shapes(self, pruned_state_dict)
        del prune_model_from_state_dict_shapes

        self.load_state_dict(pruned_state_dict, strict=True)


def get_model_properties(model):
    configs = {}
    configs["__class__.__name__"] = model.__class__.__name__

    model_dict = model.__dict__
    for key, value in model_dict.items():
        if key[0] != "_":
            configs[key] = value

    if "_modules" in model_dict:
        configs["_modules"] = {key:get_model_properties(value) for key, value in model_dict["_modules"].items()}

    return configs


def test_CleanUMamba():

    model = CleanUMamba(device="cuda", normalize_input=False)
    x = torch.rand(1, 160000).to("cuda")


    with torch.no_grad():
        # Parallel execution of the model
        denoise_par = model(x).squeeze(1)[:, :160000]
        # Sequential execution of the model
        denoise = model.feed(x)
        denoise = torch.cat([denoise, model.flush()], 1)


    assert torch.allclose(denoise, denoise_par, atol=0.1)


    summary(model, input_size=(1, 1, int(766)), depth=2, col_names=["input_size",
                                                                    "output_size",
                                                                    "num_params",
                                                                    "params_percent",
                                                                    "kernel_size",
                                                                    "mult_adds",
                                                                    "trainable"])


if __name__ == "__main__":
    test_CleanUMamba()
