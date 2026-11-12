"""Optimizer construction for ViT training."""

from torch import nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """Create AdamW for the model's trainable parameters.

    Move the model to its training device before calling this function.
    Weight decay applies to all included parameters, including biases and
    normalization parameters.
    """
    return AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
