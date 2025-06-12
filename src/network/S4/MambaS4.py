import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
#from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba_simple import Block
from torch import Tensor

from einops import rearrange, repeat

from src.network.S4.S4_fuctions import LinearActivation, Activation, DropoutNd, kernel_registry, contract

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class FFTConv(nn.Module):
    """Implements an FFT Convolution around a convolution kernel.

    d_model (H): Model dimension (in CNN terminology, this would be "channels").
    l_max (L): The maximum kernel length. Set l_max=None to always use a global kernel.
    channels: Can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this; instead, increase d_model for larger models.
    bidirectional: If True, convolution kernel will be two-sided.
    activation: Activation after the full convolution.
    transposed, dropout, tie_dropout: More general model options, see SequenceModule.
    mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D. Other options can be found in the kernel registry.

    kernel_args: See the class .kernel.SSMKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode", "init", "dt_min", "dt_max", "lr"
    """

    def __init__(
        self,
        d_model,
        l_max=None,
        channels=1,
        swap_channels=False,
        bidirectional=False,
        activation='gelu', # Activation after layer
        transposed=True,
        dropout=0.0,
        tie_dropout=False,
        drop_kernel=0.0,
        mode='dplr',
        kernel=None,
        **kernel_args,  # Arguments passed into inner convolution kernel
    ):
        super().__init__()
        self.d_model = d_model
        self.L = self.l_max = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.swap_channels = swap_channels


        if activation is not None and activation.startswith('glu'):
            channels *= 2
        self.activation = Activation(activation, dim=1 if self.transposed else -1)

        self.D = nn.Parameter(torch.randn(channels, self.d_model))

        if self.bidirectional:
            channels *= 2

        # Inner convolution kernel
        if mode is not None:
            assert kernel is None, "Pass either mode or kernel but not both"
            # log.info(
            #     "Argument 'mode' is deprecated and renamed to 'kernel',"
            #     "and will be removed in a future version."
            # )
            kernel, mode = mode, kernel
        kernel_cls = kernel_registry[kernel]
        self.kernel = kernel_cls(
            d_model=self.d_model,
            l_max=self.l_max,
            channels=channels,
            **kernel_args,
        )

        self.kernel.kernel = "keops"

        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()

    def forward(self, x, state=None, rate=1.0, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B D L) if self.transposed else (B L D)
        """

        # Always work with (B D L) dimension in this module
        if not self.transposed: x = x.transpose(-1, -2)
        L = x.size(-1)      # list as we get tensors for some reason
        if isinstance(L, Tensor):
            L = L.item()

        # Compute SS Kernel
        l_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state =  self.kernel(L=l_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))
            # The above has an off-by-one in the reverse direction
            # This is a deliberate choice since the off-by-one should not affect any applications
            # This can be amended which may be very slightly slower
            # k = F.pad(k0, (0, L)) \
            #         + F.pad(k1[..., 1:].flip(-1), (L+1, 0)) \
            #         + F.pad(k1[..., :1], (0, l_kernel+L-1))

        # Kernel dropout
        k = self.drop_kernel(k)

        # In principle, we could pad to l_kernel+L-1 instead of l_kernel+L, but we choose the latter for
        # equational simplicity. Additionally, we have not experimented to compare the efficiency of the two.
        k_f = torch.fft.rfft(k, n=l_kernel+L) # (C H L)
        x_f = torch.fft.rfft(x, n=l_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', x_f, k_f)
        y = torch.fft.irfft(y_f, n=l_kernel+L)[..., :L] # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', x, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None


        # Reshape to flatten channels
        if self.swap_channels:
            y = rearrange(y, 'b c h l -> b (h c) l')
        else:
            y = rearrange(y, 'b c h l -> b (c h) l')

        y = self.drop(y)  # DropoutNd better with transposed=True

        if not self.transposed: y = y.transpose(-1, -2)
        y = self.activation(y)

        return y, next_state


    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, x, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        y, next_state = self.kernel.step(x, state) # (B C H)
        y = y + x.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model * self.channels

class S4Block(nn.Module):
    """General block design wrapping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - initializer: Initializer on final linear layer.
    - weight_norm: Weight normalization on final linear layer.
    - dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.
    """

    def __init__(
        self,
        d_model,
        bottleneck=None,
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_model

        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            # self.d_model = self.d_model // bottleneck
            self.d_model = int(d_model * bottleneck)

            self.input_linear = LinearActivation(
                self.d_model,
                self.d_state,
                transposed=False,
                activation=None,
                activate=False,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                activation=gate_act,
                activate=True,
            )
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model*self.channels,
                    self.d_model * gate,
                    transposed=False,
                    activation=None,
                    activate=False,
                )

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        self.layer = FFTConv(self.d_state, transposed=False, dropout=dropout, tie_dropout=tie_dropout, **layer_args)

        # Pointwise operations

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            print("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_state*gate if gate is not None else self.d_state,
                self.d_model,
                transposed=False,
                activation=final_act,
                activate=True,
            )



    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        if self.transposed: x = rearrange(x, 'b d ... -> b ... d')  # why not x = x.transpose(-1, -2)
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        y, state = self.layer(x, **kwargs)


        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')

        return y, state

    def setup_step(self, **kwargs):
        self.layer.setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state) # (B C H)
        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.layer.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model

class MambaS4(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        print(self.d_inner)

        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.ssm = S4Block(
            d_model=self.d_state,
            bottleneck=self.d_inner/self.d_state
            # transposed=False
        )



        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        y, last_state = self.ssm(x)
        if ssm_state is not None:
            ssm_state.copy_(last_state)

        y = y * F.silu(z)

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

def create_block_mamba_s4(
    d_model,
    d_intermediate=0, # Not used, but kept for compatibility with Mamba 2
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(MambaS4, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        # mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


if __name__ == "__main__":
    mamba_simple = Mamba(64).cuda()
    mamba_s4_simple = MambaS4(64).cuda()

    input = torch.rand((2, 128, 64)).cuda()
    out = mamba_simple(input)
    print(out.shape)

    out_s4 = mamba_s4_simple(input)
    print(out_s4.shape)

    mamba_s4_block = create_block_mamba_s4(64).cuda()
    hidden_states, residual = mamba_s4_block(input)
    print(hidden_states.shape)
    print(residual.shape)


