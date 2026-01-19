import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .model import TransformerLM, MultiHeadSelfAttention, scaledDotProductAttention
from .optimize import AdamW, cross_entropy_loss, lr_scheduler, gradient_clipping