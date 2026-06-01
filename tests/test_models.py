import pytest
import torch

from stforecast.models import STID

B, L, N = 4, 12, 6


def _inputs():
    x = torch.randn(B, L, N, 3)
    return x, torch.randint(0, 288, (B,)), torch.randint(0, 7, (B,))
