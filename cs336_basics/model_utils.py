import torch
from torch import Tensor

from math import sqrt
from einops import rearrange, einsum

from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

def Silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax to the input tensor along the specified dimension.
    """
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
