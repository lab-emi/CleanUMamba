from typing import Optional, Tuple

import torch
import torch.nn as nn

class Activation(nn.Module):
    r"""Applies the gated linear unit function, adjusted to use torch primitives
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.
    """

    def __init__(self, activation="Sigmoid", bypass_channels=0) -> None:
        super().__init__()
        assert activation in ['Sigmoid', 'ReLU', "SiLU", "GELU"], f'activation={activation}'

        self.bypass_channels = bypass_channels
        if activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # return input[:, :input.shape[1] // 2] * self.sigmoid(input[:, input.shape[1] // 2:])

        nX = self.bypass_channels
        if nX == 0:
            nAB = (input.shape[1]) // 2
            A, B = torch.split(input, [nAB, nAB], 1)  # torch.split is explicitly handled by torch_pruning
            return A * self.activation(B)


        nAB = (input.shape[1] - nX) // 2

        X, A, B = torch.split(input, [nX, nAB, nAB], 1)  # torch.split is explicitly handled by torch_pruning
        assert A.shape == B.shape, f'A.shape={A.shape}, B.shape={B.shape}'

        return torch.cat([X, A * self.activation(B)], 1)
