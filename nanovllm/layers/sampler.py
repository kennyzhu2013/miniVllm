import torch
import platform
from torch import nn


def compile(func):
    if platform.system() == "Windows":
        return func
    return torch.compile(func)


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
