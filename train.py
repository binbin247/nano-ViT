

def get_lr(step, warmup_steps, total_steps, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * max(0.0, 1.0 - progress)