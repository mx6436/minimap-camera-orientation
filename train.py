"""在 data/train 上训练角度回归 CNN，在 data/val 上验证。"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
from model import (
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_CHANNELS,
    DEFAULT_HEAD_GRID,
    DEFAULT_NORM,
    DEFAULT_RADIUS_POOL,
    EXPECTED_PARAMETER_COUNT,
    AngleCNN,
    choose_device,
    count_trainable_parameters,
    load_model,
)

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "production_001"
DEFAULT_CONFIG_PATH = ROOT / "train.toml"
DEFAULT_THREADS = 16
ARTIFACT_NAMES = ("best.pt", "config.json", "history.json", "summary.json")

CONFIG_DEFAULTS: dict[str, Any] = {
    "batch_size": 32,
    "epochs": 200,
    "seed": SEED,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler_patience": 8,
    "early_stop_patience": 25,
    "dropout": DEFAULT_DROPOUT,
    "norm_lambda": 0.0,
    "head_grid": list(DEFAULT_HEAD_GRID),
    "head_channels": DEFAULT_HEAD_CHANNELS,
    "radius_pool": DEFAULT_RADIUS_POOL,
    "norm": DEFAULT_NORM,
    "rotation": True,
}


class AngleDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        names: list[str],
        augment: bool = False,
        rotate: bool = True,
    ) -> None:
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
                delta = 15 * random.randint(1, 23)
                # 1°/列角度轴：内容顺时针转 delta 度 == 列右移 delta（严格无损）。
                array = np.roll(array, delta, axis=1)
                angle = (angle + delta) % 360
            if random.random() < 0.5:
                array = array + np.random.normal(0.0, 0.02, array.shape).astype(np.float32)
            array = np.clip(array, 0.0, 1.0)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        target = torch.from_numpy(angle_target(angle))
        return tensor, target


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    unknown = sorted(set(values) - set(CONFIG_DEFAULTS))
    if unknown:
        raise SystemExit(f"unknown config keys in {path}: {', '.join(unknown)}")
    config = {**CONFIG_DEFAULTS, **values}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise SystemExit("epochs/batch_size must be positive")
    if not 0.0 <= config["dropout"] < 1.0:
        raise SystemExit("dropout must be in [0, 1)")
    if config["norm_lambda"] < 0.0:
        raise SystemExit("norm_lambda must be non-negative")
    if len(config["head_grid"]) != 2 or min(config["head_grid"]) < 1:
        raise SystemExit("head_grid must be two positive integers")
    if config["head_channels"] < 0:
        raise SystemExit("head_channels must be non-negative (0 keeps the 256 trunk channels)")
    if config["radius_pool"] not in ("max", "avg"):
        raise SystemExit("radius_pool must be 'max' or 'avg'")
    if config["norm"] not in ("batch", "group"):
        raise SystemExit("norm must be 'batch' or 'group'")
    if config["lr"] <= 0.0 or config["weight_decay"] < 0.0:
        raise SystemExit("lr must be positive and weight_decay non-negative")
    if config["scheduler_patience"] < 1 or config["early_stop_patience"] < 1:
        raise SystemExit("scheduler_patience/early_stop_patience must be positive")


def _rank_data(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    return float(np.corrcoef(_rank_data(a), _rank_data(b))[0, 1])


def norm_metrics(outputs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """模长统计 + 原始 MSE 的精确加法分解。

    |v-y|^2 = (||v||-1)^2 + 2*||v||*(1-cos dtheta)；两项均非负，因此
    norm_err_share = mean((||v||-1)^2) / mean(|v-y|^2) 是精确分解。

    v 与单位目标向量的夹角等于解码后的循环误差，故 arccos 对齐角可兼作
    逐样本误差，供按模长表盘追踪。"""
    norms = np.linalg.norm(outputs, axis=-1)
    raw_mse = np.sum((outputs - targets) ** 2, axis=-1)
    total = float(np.mean(raw_mse))
    align = np.sum(outputs * targets, axis=-1) / np.maximum(norms, 1e-12)
    angular = np.degrees(np.arccos(np.clip(align, -1.0, 1.0)))
    low = norms <= np.percentile(norms, 20)
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_p5": float(np.percentile(norms, 5)),
        "norm_p95": float(np.percentile(norms, 95)),
        "raw_mse_total": total,
        "norm_err_share": float(np.mean((norms - 1.0) ** 2) / total) if total > 0 else 0.0,
        "spearman_norm_err": _spearman(norms, angular),
        "norm_low20_mae": float(np.mean(angular[low])),
        "norm_low20_gt10_share": float(np.mean(angular[low] > 10.0)),
    }


def metrics_from_outputs(outputs: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    continuous = decode_angle(outputs)
    errors = circular_error(continuous, angles)
    rounded = round_angle(continuous)
    return {
        "circular_mae": float(np.mean(errors)),
        "circular_rmse": float(np.sqrt(np.mean(errors**2))),
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


def combined_loss(outputs: torch.Tensor, targets: torch.Tensor, norm_lambda: float) -> torch.Tensor:
    """原始 sin/cos 上的 MSE，外加可选的二次模长锚。

    带锚时，方向对齐度为 c 的逐样本均衡模长为 r = (c + 2*lam) / (1 + 2*lam)：
    c = 1 时 r = 1，随 c 单调，完全不确定的样本下限为 2*lam/(1+2*lam)。"""
    mse = F.mse_loss(outputs, targets)
    if norm_lambda <= 0.0:
        return mse
    norms = torch.linalg.vector_norm(outputs, dim=-1)
    return mse + norm_lambda * ((norms - 1.0) ** 2).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    norm_lambda: float,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    samples = 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = combined_loss(model(features.to(device)), targets.to(device), norm_lambda)
        loss.backward()
        optimizer.step()
        batch_size = len(features)
        total += loss.item() * batch_size
        samples += batch_size
    return total / samples


def eval_loss(
    model: nn.Module, loader: DataLoader, norm_lambda: float, device: torch.device
) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    samples = 0
    outputs: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets, batch_angles, _batch_names in loader:
            prediction = model(features.to(device))
            batch_size = len(features)
            total += combined_loss(prediction, targets.to(device), norm_lambda).item() * batch_size
            samples += batch_size
            outputs.append(prediction.cpu().numpy())
            angles.append(batch_angles.numpy().astype(np.float64))
            targets_list.append(targets.numpy())
    output_array = np.concatenate(outputs)
    angle_array = np.concatenate(angles)
    metrics = metrics_from_outputs(output_array, angle_array)
    metrics.update(norm_metrics(output_array, np.concatenate(targets_list)))
    return total / samples, metrics


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
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


def save_checkpoint(path: Path, model: nn.Module, model_config: dict[str, Any]) -> None:
    checkpoint = {"model": model.state_dict(), "config": model_config}
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
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境也可用
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="训练配置 TOML（模型/损失/优化/增强）",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="auto、cpu 或 cuda")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="torch 的 CPU 线程数（torch.set_num_threads）",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="以正常训练路径只跑一个 epoch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        raise SystemExit(f"config file not found: {args.config}") from None
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"invalid TOML in {args.config}: {error}") from error
    torch.set_num_threads(args.threads)
    seed_everything(config["seed"])
    device = choose_device(args.device)
    train_names = png_names(TRAIN_DIR)
    val_names = png_names(VAL_DIR)
    if not train_names or not val_names:
        raise SystemExit(f"missing {TRAIN_DIR} or {VAL_DIR} PNG files; run prepare_data.py first")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in ARTIFACT_NAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"{output_dir} already contains experiment artifacts ({', '.join(existing)}); "
            "choose an empty --output-dir"
        )

    model_config = {
        "dropout": config["dropout"],
        "head_grid": list(config["head_grid"]),
        "head_channels": config["head_channels"] or None,
        "radius_pool": config["radius_pool"],
        "norm": config["norm"],
    }
    loss_description = "MSE([raw_sin, raw_cos], [target_sin, target_cos])"
    if config["norm_lambda"] > 0.0:
        loss_description += f" + {config['norm_lambda']:g}*(||v||-1)^2"
    record: dict[str, Any] = {
        "version": 15,
        "head_grid": list(config["head_grid"]),
        "head_channels": config["head_channels"] or None,
        "radius_pool": config["radius_pool"],
        "norm": config["norm"],
        "dropout": config["dropout"],
        "norm_lambda": config["norm_lambda"],
        "loss": loss_description,
        "input_shape": [3, 44, 360],
        "input_scaling": "RGB uint8 / 255",
        "input_representation": (
            "polar_unwrap_rgb_360x44 (angle->x, 1 deg/column, clockwise, north at column 0; "
            "radius->y, inner at top)"
        ),
        "conv_padding_mode": "circular",
        "seed": config["seed"],
        "threads": args.threads,
        "device": str(device),
        "model": "AngleCNN",
        "trainable_parameters": None,
        "batch_size": config["batch_size"],
        "max_epochs": config["epochs"],
        "optimizer": "AdamW",
        "learning_rate": config["lr"],
        "weight_decay": config["weight_decay"],
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "patience": config["scheduler_patience"],
            "factor": 0.5,
            "min_lr": 1e-6,
        },
        "early_stopping_patience": config["early_stop_patience"],
        "augmentation": {
            "rgb_gaussian_noise": {
                "probability": 0.5,
                "sigma": 0.02,
                "masked_to_ring_alpha": False,
            },
            "clockwise_rotation": {
                "probability": 0.5 if config["rotation"] else 0.0,
                "degrees": (
                    "15-multiples 15..345 (24 directions); "
                    "lossless np.roll on the 1 deg/column angular axis"
                ),
            },
        },
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_files": train_names,
        "val_files": val_names,
    }
    model = AngleCNN(
        dropout=config["dropout"],
        head_grid=tuple(config["head_grid"]),
        head_channels=config["head_channels"] or None,
        radius_pool=config["radius_pool"],
        norm=config["norm"],
    ).to(device)
    parameter_count = count_trainable_parameters(model)
    if model.is_default_architecture() and parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    record["trainable_parameters"] = parameter_count
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["scheduler_patience"],
        min_lr=1e-6,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(config["seed"])
    train_loader = make_loader(
        AngleDataset(TRAIN_DIR, train_names, True, rotate=config["rotation"]),
        config["batch_size"],
        True,
        config["seed"],
        device,
        train_generator,
    )
    val_loader = make_loader(
        NamedDataset(AngleDataset(VAL_DIR, val_names, False)),
        config["batch_size"],
        False,
        config["seed"],
        device,
    )

    best_val = float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    atomic_json_dump(output_dir / "config.json", record)
    atomic_json_dump(output_dir / "history.json", {"epochs": history})
    max_epochs = 1 if args.smoke else config["epochs"]
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config["norm_lambda"], device)
        val_loss, val_metrics = eval_loss(model, val_loader, config["norm_lambda"], device)
        scheduler.step(val_metrics["circular_mae"])
        record_epoch = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record_epoch)
        improved = val_metrics["circular_mae"] < best_val
        if improved:
            best_val = val_metrics["circular_mae"]
            bad_epochs = 0
            save_checkpoint(output_dir / "best.pt", model, model_config)
        else:
            bad_epochs += 1
        atomic_json_dump(output_dir / "history.json", {"epochs": history})
        print(
            f"epoch={epoch + 1}/{max_epochs} train_loss={train_loss:.6f} "
            f"val_mae={val_metrics['circular_mae']:.3f}° "
            f"val_rmse={val_metrics['circular_rmse']:.3f}°"
        )
        if not args.smoke and bad_epochs >= config["early_stop_patience"]:
            print("early stopping")
            break

    if not (output_dir / "best.pt").exists():
        raise RuntimeError("best checkpoint was not produced")
    best_record = min(history, key=lambda record: record["val_circular_mae"])
    # 结算：重新加载 best.pt 并在验证集上重新评估，确保汇报的数字就是
    # 交付 checkpoint 的数字。
    settled_model = load_model(output_dir / "best.pt", device=device)
    _, final_metrics = eval_loss(settled_model, val_loader, config["norm_lambda"], device)
    summary: dict[str, Any] = {
        "epoch": int(best_record["epoch"]),
        "val_count": len(val_names),
        "best_val_circular_mae": best_record["val_circular_mae"],
        **{f"val_{k}": v for k, v in final_metrics.items()},
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    print(
        f"best val_circular_mae={best_record['val_circular_mae']:.3f}° "
        f"(epoch {best_record['epoch']})"
    )
    print("final evaluation on best.pt (val set):")
    print(
        f"  val_circular_mae={final_metrics['circular_mae']:.3f}°  "
        f"val_circular_median={final_metrics['circular_median']:.3f}°"
    )
    print(
        f"  within_1_degree={final_metrics['within_1_degree']:.2%}  "
        f"within_3_degrees={final_metrics['within_3_degrees']:.2%}  "
        f"within_5_degrees={final_metrics['within_5_degrees']:.2%}  "
        f"within_10_degrees={final_metrics['within_10_degrees']:.2%}"
    )
    print(f"  integer_accuracy={final_metrics['integer_accuracy']:.2%}")
    plot_loss_curves(output_dir, history)


if __name__ == "__main__":
    main()
