import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data_loader import get_data_loader
from model import ViT
from optim import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Train arguments
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Final learning rate after cosine decay")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Epochs of linear warmup; use 0 to disable")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--num-epoch", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory for last.pt and best.pt")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for curves, metrics, and run arguments")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a saved checkpoint")

    # ViT arguments (defaults match the former L/16 preset)
    parser.add_argument("--in-channels", type=int, required=True, help="Number of input image channels")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of output classes")
    parser.add_argument("--ebd-dim", type=int, default=1024)
    parser.add_argument("--n-head", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--ffn-dim", type=int, default=4096)
    parser.add_argument("--ffn-type", type=str, choices=("gelu", "swiglu"), default="gelu")
    parser.add_argument("--dropout-img-ebd", type=float, default=0.1)
    parser.add_argument("--dropout-att-weight", type=float, default=0.0)
    parser.add_argument("--dropout-att-out", type=float, default=0.1)
    parser.add_argument("--dropout-ffn-inner", type=float, default=0.1)
    parser.add_argument("--dropout-ffn-out", type=float, default=0.1)
    parser.add_argument("--ffn-bias", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    return args

def get_device(args: argparse.Namespace) -> torch.device:
    """Select the accelerator. cuda | mps | cpu"""

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.device.startswith("mps") and torch.backends.mps.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    return device


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Set the first update's LR, then advance warmup/cosine after each update.

    Warmup uses base_lr / warmup_steps through base_lr. Cosine decay reaches
    min_lr on the last update (when total_steps > 1).
    """

    if base_lr is None:
        base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = min_lr / base_lr
    decay_start = max(warmup_steps - 1, 0)
    decay_steps = max(total_steps - 1 - decay_start, 1)

    def lr_factor(step: int) -> float:
        # LambdaLR sets step=0 on construction, before the first optimizer step.
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    model: ViT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_losses: list[float],
    eval_accuracies: list[float],
    learning_rates: list[float],
    best_accuracy: float,
    schedule_config: dict[str, int | float],
    args: argparse.Namespace,
) -> None:
    """Save all state needed to resume a training run."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        obj = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "train_losses": train_losses,
            "eval_accuracies": eval_accuracies,
            "learning_rates": learning_rates,
            "best_accuracy": best_accuracy,
            "schedule_config": schedule_config,
            "args": vars(args).copy(),
        },
        f = checkpoint_path,
    )


def restore_checkpoint(
    checkpoint: dict,
    *,
    model: ViT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    """Restore model, optimizer, and scheduler from checkpoint data."""

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)


def load_checkpoint(
    path: str,
    *,
    model: ViT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> dict:
    """Load and restore a checkpoint created locally."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    restore_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return checkpoint


def save_run_metadata(args: argparse.Namespace) -> None:
    """Save the resolved command-line arguments for this run."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.txt").open("w", encoding="utf-8") as file:
        for name, value in sorted(vars(args).items()):
            file.write(f"{name}={value}\n")


def save_training_metrics(
    args: argparse.Namespace,
    train_losses: list[float],
    eval_accuracies: list[float],
    learning_rates: list[float],
) -> None:
    """Save epoch-level metrics in a machine-readable format."""

    output_dir = Path(args.output_dir)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_losses": train_losses,
                "eval_accuracies": eval_accuracies,
                "learning_rates": learning_rates,
            },
            file,
            indent=2,
        )


def train_one_epoch(
    model: ViT, 
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    # 梯度归零 -> forward -> 计算 loss -> backward -> 更新权重

    model.train()
    total_loss = 0.0
    total_samples = 0

    for imgs, labels in train_loader:
        # Tensor.to() returns a new tensor; assign it back to use the device copy.
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(imgs)
            loss = F.cross_entropy(input=logits, target=labels)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)

@torch.no_grad()
def evaluate(
    model: ViT,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    correct_count = 0
    total_count = 0

    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        predictions = torch.argmax(logits, dim=1)

        correct_count += (predictions == labels).sum().item()
        total_count += labels.size(0)

    return correct_count / max(total_count, 1)

def plot_training_history(
    train_losses: list[float],
    eval_accuracies: list[float],
    output_path: str = "training_curves.png",
    learning_rates: list[float] | None = None,
) -> None:
    """Save loss, accuracy, and optional epoch-end learning rate curves."""

    if len(train_losses) != len(eval_accuracies):
        raise ValueError("train_losses and eval_accuracies must have the same length")
    if learning_rates is not None and len(learning_rates) != len(train_losses):
        raise ValueError("learning_rates must have the same length as train_losses")
    if not train_losses:
        return

    # Import lazily so training utilities remain importable without matplotlib.
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    num_plots = 3 if learning_rates is not None else 2
    figure, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
    loss_axis, accuracy_axis = axes[:2]

    loss_axis.plot(epochs, train_losses, marker="o")
    loss_axis.set_title("Training loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True, alpha=0.3)

    accuracy_axis.plot(epochs, eval_accuracies, marker="o")
    accuracy_axis.set_title("Evaluation accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.grid(True, alpha=0.3)

    if learning_rates is not None:
        lr_axis = axes[2]
        lr_axis.plot(epochs, learning_rates, marker="o")
        lr_axis.set_title("Learning rate at epoch end")
        lr_axis.set_xlabel("Epoch")
        lr_axis.set_ylabel("Learning rate")
        lr_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        lr_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

def main() -> None :
    # input arguments
    args = parse_args()
    save_run_metadata(args)

    # set random seed
    torch.manual_seed(args.seed)

    # set device: cuda or mps or cpu
    device = get_device(args)

    # prepare data loader
    train_loader, val_loader = get_data_loader(args)

    steps_per_epoch = len(train_loader)
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

    if resume_checkpoint is None:
        schedule_config = {
            "total_steps": args.num_epoch * steps_per_epoch,
            "warmup_steps": int(args.warmup_epochs * steps_per_epoch),
            "min_lr": args.min_lr,
            "base_lr": args.lr,
            "steps_per_epoch": steps_per_epoch,
        }
    else:
        schedule_config = resume_checkpoint["schedule_config"].copy()

        saved_steps_per_epoch = int(schedule_config["steps_per_epoch"])
        if saved_steps_per_epoch != steps_per_epoch:
            raise ValueError(
                "Cannot resume with a different number of steps per epoch: "
                f"checkpoint={saved_steps_per_epoch}, current={steps_per_epoch}. "
                "Keep the batch size and dataset split unchanged."
            )

    # prepare model
    model = ViT(args).to(device)

    # prepare optimizer
    optimizer =  build_optimizer(
        model = model,
        lr = args.lr,
        weight_decay = args.weight_decay,
        betas = (args.beta1, args.beta2),
        eps = args.eps,
    )
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        total_steps=int(schedule_config["total_steps"]),
        warmup_steps=int(schedule_config["warmup_steps"]),
        min_lr=float(schedule_config["min_lr"]),
        base_lr=float(schedule_config["base_lr"]),
    )

    train_losses: list[float] = []
    eval_accuracies: list[float] = []
    learning_rates: list[float] = []
    start_epoch = 1
    best_accuracy = -float("inf")

    if resume_checkpoint is not None:
        restore_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = resume_checkpoint["epoch"] + 1
        train_losses = resume_checkpoint.get("train_losses", [])
        eval_accuracies = resume_checkpoint.get("eval_accuracies", [])
        learning_rates = resume_checkpoint.get("learning_rates", [])
        best_accuracy = resume_checkpoint.get("best_accuracy", -float("inf"))
        print(
            f"Resumed from {args.resume} at epoch "
            f"{resume_checkpoint['epoch']} using the saved scheduler."
        )

    if start_epoch > args.num_epoch:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}, but "
            f"--num-epoch is {args.num_epoch}."
        )

    for epoch in range(start_epoch, args.num_epoch + 1):
        train_loss = train_one_epoch(
            model = model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
        )
        eval_accuracy = evaluate(
            model = model,
            val_loader = val_loader,
            device = device,
        )

        train_losses.append(train_loss)
        eval_accuracies.append(eval_accuracy)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        print(
            f"Epoch {epoch:03d}/{args.num_epoch} | "
            f"train loss: {train_loss:.4f} | "
            f"val accuracy: {eval_accuracy:.4f}"
        )

        is_best = eval_accuracy >= best_accuracy
        if is_best:
            best_accuracy = eval_accuracy

        checkpoint_kwargs = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_losses": train_losses,
            "eval_accuracies": eval_accuracies,
            "learning_rates": learning_rates,
            "best_accuracy": best_accuracy,
            "schedule_config": schedule_config,
            "args": args,
        }
        save_checkpoint(
            str(Path(args.checkpoint_dir) / "last.pt"),
            **checkpoint_kwargs,
        )

        if is_best:
            save_checkpoint(
                str(Path(args.checkpoint_dir) / "best.pt"),
                **checkpoint_kwargs,
            )

    plot_training_history(
        train_losses=train_losses,
        eval_accuracies=eval_accuracies,
        learning_rates=learning_rates,
        output_path=str(Path(args.output_dir) / "training_curves.png"),
    )
    save_training_metrics(
        args=args,
        train_losses=train_losses,
        eval_accuracies=eval_accuracies,
        learning_rates=learning_rates,
    )


if __name__ == "__main__":
    main()
