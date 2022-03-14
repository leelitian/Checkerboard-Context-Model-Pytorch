from typing import Any

import torch
import torch.nn as nn

from torch import Tensor


class CheckerboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


if __name__ == '__main__':

    # notice that the bias is 'True' in practice
    ckbd = CheckerboardMaskedConv2d(3, 3, kernel_size=5, padding=2, stride=1, bias=True)
    x = torch.rand((1, 3, 8, 8))

    print(ckbd(x))