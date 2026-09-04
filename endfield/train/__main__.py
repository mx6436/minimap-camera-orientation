"""训练 AzimuthNet 并在验证集上评估（`uv run train`，见 ADR 0004）。"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import torch

from endfield.data_utils import atomic_json_dump, png_names, seed_everything
from endfield.model import (
    EXPECTED_PARAMETER_COUNT,
    AzimuthNet,
    choose_device,
    count_trainable_parameters,
    load_model,
)
from endfield.train.artifacts import ARTIFACT_NAMES, plot_loss_curves, save_checkpoint
from endfield.train.config import load_config
from endfield.train.data import TRAIN_DIR, VAL_DIR, AngleDataset, make_loader, names_fingerprint
from endfield.train.engine import eval_loss, train_epoch
from endfield.train.record import build_record

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "production_001"
DEFAULT_CONFIG_PATH = REPO_ROOT / "train.toml"
DEFAULT_THREADS = 16


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

    model = AzimuthNet().to(device)
    track_metric = "expected_rmse"
    parameter_count = count_trainable_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    record = build_record(
        config,
        args.threads,
        device,
        len(train_names),
        len(val_names),
        names_fingerprint(train_names),
        names_fingerprint(val_names),
    )
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
        AngleDataset(TRAIN_DIR, train_names, augment=True),
        config["batch_size"],
        True,
        config["seed"],
        device,
        train_generator,
    )
    val_loader = make_loader(
        AngleDataset(VAL_DIR, val_names),
        config["batch_size"],
        False,
        config["seed"],
        device,
    )

    best_val = float("inf")
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    atomic_json_dump(output_dir / "record.json", record)
    atomic_json_dump(output_dir / "history.json", {"epochs": history})
    max_epochs = 1 if args.smoke else config["epochs"]
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = eval_loss(model, val_loader, device)
        scheduler.step(val_metrics[track_metric])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        improved = val_metrics[track_metric] < best_val
        if improved:
            best_val = val_metrics[track_metric]
            bad_epochs = 0
            save_checkpoint(output_dir / "best.pt", model)
        else:
            bad_epochs += 1
        atomic_json_dump(output_dir / "history.json", {"epochs": history})
        print(
            f"epoch={epoch + 1}/{max_epochs} train_loss={train_loss:.6f} "
            f"val_{track_metric}={val_metrics[track_metric]:.3f}°"
        )
        if not args.smoke and bad_epochs >= config["early_stop_patience"]:
            print("early stopping")
            break

    if not (output_dir / "best.pt").exists():
        raise RuntimeError("best checkpoint was not produced")
    best_entry = min(history, key=lambda entry: entry[f"val_{track_metric}"])
    # 结算：重新加载 best.pt 并在验证集上重新评估，确保汇报的数字就是
    # 交付 checkpoint 的数字。
    settled_model = load_model(output_dir / "best.pt", device=device)
    final_loss, final_metrics = eval_loss(settled_model, val_loader, device)
    summary: dict[str, Any] = {
        "epoch": int(best_entry["epoch"]),
        "val_count": len(val_names),
        f"best_val_{track_metric}": best_entry[f"val_{track_metric}"],
        "val_loss": final_loss,
        **{f"val_{k}": v for k, v in final_metrics.items()},
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    print(
        f"best val_{track_metric}={best_entry[f'val_{track_metric}']:.3f}° "
        f"(epoch {best_entry['epoch']})"
    )
    print("final evaluation on best.pt (val set):")
    print(f"  val_loss={final_loss:.4f}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in final_metrics.items()))
    plot_loss_curves(output_dir, history)


if __name__ == "__main__":
    main()
