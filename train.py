#!/usr/bin/env python3
"""Train the angle regression CNN on data/train and validate on data/val."""
from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_utils import (
    SEED,
    angle_target,
    atomic_json_dump,
    circular_error,
    decode_angle,
    load_rgb,
    parse_angle,
    png_names,
    round_angle,
    seed_everything,
)
from model import AngleCNN, EXPECTED_PARAMETER_COUNT, count_trainable_parameters

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
ARTIFACT_NAMES = (
    "best.pt",
    "last.pt",
    "history.json",
    "config.json",
)
EARLY_STOP_PATIENCE = 40
SCHEDULER_PATIENCE = 20
DEFAULT_THREADS = 16


class AngleDataset(Dataset):
    def __init__(self, directory: Path, names: list[str], augment: bool = False,
                 rotate: bool = True) -> None:
        self.directory = directory
        self.names = names
        self.augment = augment
        self.rotate = rotate

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        angle = parse_angle(Path(name))
        array = load_rgb(self.directory / name).astype(np.float32) / 255.0
        if self.augment:
            if self.rotate and random.random() < 0.5:
                delta = 15 * random.randint(1, 23)  # 15, 30, ..., 345
                # 1°/列角度轴：内容顺时针转 delta 度 == 列右移 delta（严格无损）。
                array = np.roll(array, delta, axis=1)
                angle = (angle + delta) % 360
            if random.random() < 0.5:
                array = array + np.random.normal(0.0, 0.02, array.shape).astype(np.float32)
            array = np.clip(array, 0.0, 1.0)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        target = torch.from_numpy(angle_target(angle))
        return tensor, target


class AngularMSELoss(nn.Module):
    """MSE between unit-normalized predictions and unit targets (= 2-2cos dtheta).

    Output norm is excluded from the objective: decode_angle normalizes anyway,
    so raw-norm errors are pure noise that used to dominate the loss (~90% of
    validation MSE) and mask fine angular progress."""

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(nn.functional.normalize(outputs, dim=-1), targets)


def metrics_from_outputs(outputs: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    continuous = decode_angle(outputs)
    errors = circular_error(continuous, angles)
    rounded = round_angle(continuous)
    return {
        "circular_mae": float(np.mean(errors)),
        "circular_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "circular_median": float(np.median(errors)),
        "integer_accuracy": float(np.mean(rounded == angles)),
        "within_1_degree": float(np.mean(errors <= 1.0)),
        "within_3_degrees": float(np.mean(errors <= 3.0)),
        "within_5_degrees": float(np.mean(errors <= 5.0)),
        "within_10_degrees": float(np.mean(errors <= 10.0)),
    }


class NamedDataset(Dataset):
    def __init__(self, dataset: AngleDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        features, target = self.dataset[index]
        name = self.dataset.names[index]
        return features, target, parse_angle(Path(name)), name


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total = 0.0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item() * len(features)
    return total / len(loader.dataset)


def eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    outputs: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets, batch_angles, _batch_names in loader:
            prediction = model(features.to(device))
            total += criterion(prediction, targets.to(device)).item() * len(features)
            outputs.append(prediction.cpu().numpy())
            angles.append(batch_angles.numpy().astype(np.float64))
    output_array = np.concatenate(outputs)
    angle_array = np.concatenate(angles)
    return total / len(loader.dataset), metrics_from_outputs(output_array, angle_array)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int,
                device: torch.device, generator: torch.Generator | None = None) -> DataLoader:
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )


def capture_rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_loader": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: object, train_generator: torch.Generator) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain resumable RNG state")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        train_generator.set_state(state["train_loader"].cpu())
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint RNG state is incomplete or incompatible") from error


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
                    epoch: int, best_val: float, bad_epochs: int, config: dict[str, Any],
                    history: list[dict[str, Any]], train_generator: torch.Generator) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_circular_mae": best_val,
        "early_stop_bad_epochs": bad_epochs,
        "config": config,
        "history": history,
        "rng_state": capture_rng_state(train_generator),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(checkpoint, temp_name)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def plot_loss_curves(output_dir: Path, history: list[dict[str, Any]]) -> None:
    """Plot train/val loss curves with matplotlib and save them to output_dir."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe, no display required
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train loss", color="#1f77b4")
    ax.plot(epochs, val_loss, label="val loss", color="#ff7f0e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE)")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "loss_curve.png"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=output_dir)
    os.close(fd)
    try:
        fig.savefig(temp_name, dpi=150, format="png")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(fig)
    print(f"saved loss curve: {path}")


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, device: torch.device | str = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"invalid checkpoint: {path}")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def validate_resume_config(checkpoint_config: object, config: dict[str, Any]) -> None:
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint config is missing or invalid")
    keys = (
        "version",
        "seed",
        "input_shape",
        "model",
        "trainable_parameters",
        "batch_size",
        "threads",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "early_stopping_patience",
        "loss",
        "augmentation",
        "train_files",
        "val_files",
        "input_representation",
        "conv_padding_mode",
    )
    mismatched = [key for key in keys if checkpoint_config.get(key) != config.get(key)]
    if mismatched:
        raise ValueError(f"checkpoint configuration mismatch for: {', '.join(mismatched)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "production_001")
    parser.add_argument("--resume", type=Path, nargs="?", const="__DEFAULT__", help="resume last checkpoint, or provide a checkpoint path")
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="CPU core count for torch (torch.set_num_threads)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-rotation", action="store_true",
                        help="disable rotation augmentation (noise only)")
    parser.add_argument("--smoke", action="store_true", help="run one epoch with the normal training path")
    return parser.parse_args()


def choose_device(value: str | None) -> torch.device:
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    return device


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise SystemExit("--epochs/--batch-size/--threads must be positive")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    train_names = png_names(TRAIN_DIR)
    val_names = png_names(VAL_DIR)
    if not train_names or not val_names:
        raise SystemExit(
            f"missing {TRAIN_DIR} or {VAL_DIR} PNG files; run prepare_data.py first"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume
    if resume_path is not None:
        resume_path = output_dir / "last.pt" if str(resume_path) == "__DEFAULT__" else resume_path.resolve()
        if resume_path.parent != output_dir:
            raise ValueError("--resume checkpoint must be inside --output-dir")
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    else:
        existing = [name for name in ARTIFACT_NAMES if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(
                f"{output_dir} already contains experiment artifacts ({', '.join(existing)}); "
                "use --resume or choose an empty --output-dir"
            )

    config: dict[str, Any] = {
        "version": 11,
        "input_shape": [3, 44, 360],
        "input_scaling": "RGB uint8 / 255",
        "input_representation": "polar_unwrap_rgb_360x44 (angle->x, 1 deg/column, clockwise, north at column 0; radius->y, inner at top)",
        "conv_padding_mode": "circular",
        "seed": args.seed,
        "threads": args.threads,
        "device": str(device),
        "model": "AngleCNN",
        "trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": {"name": "ReduceLROnPlateau", "patience": SCHEDULER_PATIENCE, "factor": 0.5, "min_lr": 1e-6},
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "loss": "MSE(normalize([raw_sin, raw_cos]), [target_sin, target_cos]) = 2-2cos(dtheta)",
        "augmentation": {
            "rgb_gaussian_noise": {"probability": 0.5, "sigma": 0.02,
                                   "masked_to_ring_alpha": False},
            "clockwise_rotation": {
                "probability": 0.0 if args.no_rotation else 0.5,
                "degrees": "15-multiples 15..345 (24 directions); lossless np.roll on the 1 deg/column angular axis",
            },
        },
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_files": train_names,
        "val_files": val_names,
    }
    model = AngleCNN().to(device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6)
    criterion = AngularMSELoss()
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = make_loader(
        AngleDataset(TRAIN_DIR, train_names, True, rotate=not args.no_rotation),
        args.batch_size,
        True,
        args.seed,
        device,
        train_generator,
    )
    val_loader = make_loader(
        NamedDataset(AngleDataset(VAL_DIR, val_names, False)),
        args.batch_size,
        False,
        args.seed,
        device,
    )
    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        validate_resume_config(checkpoint.get("config"), config)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val = float(checkpoint.get("best_val_circular_mae", best_val))
        bad_epochs = int(checkpoint.get("early_stop_bad_epochs", 0))
        history = list(checkpoint.get("history", []))
        if len(history) != start_epoch:
            raise ValueError("checkpoint epoch/history length is inconsistent")
        restore_rng_state(checkpoint.get("rng_state"), train_generator)

    atomic_json_dump(output_dir / "config.json", config)
    atomic_json_dump(output_dir / "history.json", {"epochs": history})
    max_epochs = start_epoch + 1 if args.smoke else args.epochs
    max_epochs = min(max_epochs, args.epochs)
    if bad_epochs >= EARLY_STOP_PATIENCE:
        print("checkpoint has already reached the early-stopping condition")
    for epoch in range(start_epoch, max_epochs if bad_epochs < EARLY_STOP_PATIENCE else start_epoch):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_loss(model, val_loader, criterion, device)
        scheduler.step(val_metrics["circular_mae"])
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()},
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        improved = val_metrics["circular_mae"] < best_val
        if improved:
            best_val = val_metrics["circular_mae"]
            bad_epochs = 0
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler, epoch, best_val, bad_epochs,
                config, history, train_generator,
            )
        else:
            bad_epochs += 1
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, epoch, best_val, bad_epochs,
            config, history, train_generator,
        )
        atomic_json_dump(output_dir / "history.json", {"epochs": history})
        print(f"epoch={epoch + 1}/{max_epochs} train_loss={train_loss:.6f} val_mae={val_metrics['circular_mae']:.3f}° val_rmse={val_metrics['circular_rmse']:.3f}°")
        if not args.smoke and bad_epochs >= EARLY_STOP_PATIENCE:
            print("early stopping")
            break

    if not (output_dir / "best.pt").exists():
        raise RuntimeError("best checkpoint was not produced")
    if history:
        best_record = min(history, key=lambda record: record["val_circular_mae"])
        # Settlement: reload best.pt and re-evaluate on the validation set so the
        # reported numbers are exactly those of the shipped checkpoint.
        load_checkpoint(output_dir / "best.pt", model, device=device)
        _, final_metrics = eval_loss(model, val_loader, criterion, device)
        summary: dict[str, Any] = {
            "epoch": int(best_record["epoch"]),
            "val_count": len(val_names),
            "best_val_circular_mae": best_record["val_circular_mae"],
            **{f"val_{k}": v for k, v in final_metrics.items()},
        }
        atomic_json_dump(output_dir / "summary.json", summary)
        print(f"best val_circular_mae={best_record['val_circular_mae']:.3f}° (epoch {best_record['epoch']})")
        print("final evaluation on best.pt (val set):")
        print(f"  val_circular_mae={final_metrics['circular_mae']:.3f}°  "
              f"val_circular_median={final_metrics['circular_median']:.3f}°")
        print(f"  within_1_degree={final_metrics['within_1_degree']:.2%}  "
              f"within_3_degrees={final_metrics['within_3_degrees']:.2%}  "
              f"within_5_degrees={final_metrics['within_5_degrees']:.2%}  "
              f"within_10_degrees={final_metrics['within_10_degrees']:.2%}")
        print(f"  integer_accuracy={final_metrics['integer_accuracy']:.2%}")
        plot_loss_curves(output_dir, history)


if __name__ == "__main__":
    main()
