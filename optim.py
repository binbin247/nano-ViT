"""Optimizer construction for ViT training."""

import inspect

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
    Weight decay applies only to parameters with at least two dimensions.
    """
    param_dict = {pn:p for pn, p in model.named_parameters() if p.requires_grad}
    weight_decay_group = [p for p in param_dict.values() if p.dim() >= 2]
    weight_nodecay_group = [p for p in param_dict.values() if p.dim() < 2]

    optim_groups = [
        { "params" : weight_decay_group, "weight_decay" : weight_decay },
        { "params" : weight_nodecay_group, "weight_decay" : 0.0},
    ]

    fused_available = "fused" in inspect.signature(AdamW).parameters

    extra_args = (
        {"fused": all(p.is_cuda for p in param_dict.values())}
        if fused_available
        else {}
    )
    if extra_args.get("fused", False):
        print("Using fused AdamW.")

    return AdamW(
        optim_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        **extra_args
    )
