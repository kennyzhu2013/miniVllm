import torch
import platform
from torch import nn
import torch.nn.functional as F


def compile(func):
    if platform.system() == "Windows":
        return func
    return torch.compile(func)


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    # 替代@torch.compile，因为Windows上不支持
    @compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
