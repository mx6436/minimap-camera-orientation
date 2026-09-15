"""训练 AzimuthNet 并在验证集上评估"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import torch

from endfield import preprocess_cache, run_dir, run_record
from endfield.atomic_io import atomic_json_dump, load_json
from endfield.data_utils import png_names, seed_everything
from endfield.dataset import PROCESSED_REF_DIR
from endfield.model import (
    AzimuthNet,
    choose_device,
    count_trainable_parameters,
    expected_parameter_count,
    load_model,
)
from endfield.train import artifacts as train_artifacts
from endfield.train.artifacts import plot_loss_curves, save_checkpoint
from endfield.train.config import load_config
from endfield.train.data import (
    AngleDataset,
    filter_reference_gap,
    make_loader,
    names_fingerprint,
    split_dirs,
)
from endfield.train.engine import eval_loss, train_epoch
from endfield.train.metrics import Metrics
from endfield.train.record import build_record
from placement import workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "train.toml"
# 8 物理核：SMT 线程对 conv 负载无增益反有争用
DEFAULT_THREADS = 8
# 会话级、不改变训练前提的字段：续训时放行，其余任一不一致即拒
_SESSION_KEYS = frozenset({"threads", "device", "max_epochs"})


def _tracked_metric(entry: dict[str, Any]) -> float:
    """逐 epoch 历史行 -> 跟踪指标；磁盘键到 typed 指标的转换只在 `Metrics` 一处。"""
    return getattr(Metrics.from_payload(entry), Metrics.TRACK_FIELD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="训练配置 TOML（模型/损失/优化/增强）",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="从 run 目录的 last.pt 接续训练")
    parser.add_argument("--device", default=None, help="auto、cpu 或 cuda")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="torch 的 CPU 线程数（torch.set_num_threads）",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="禁用 torch.compile",
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
    train_dir, val_dir = split_dirs(config["input_mode"])
    train_names = png_names(train_dir)
    val_names = png_names(val_dir)
    if not train_names or not val_names:
        raise SystemExit(f"missing {train_dir} or {val_dir} PNG files; run prepare-data first")
    assets_root = None
    if config["input_mode"] is run_record.InputMode.REF:
        assets_root = workspace.assets_root_from_provenance(
            preprocess_cache.read_stamp(PROCESSED_REF_DIR)
        )
    if config["max_ref_missing"] is not None:
        total = len(train_names)
        train_names = filter_reference_gap(train_names, train_dir, config["max_ref_missing"])
        print(
            f"max_ref_missing={config['max_ref_missing']:g}: "
            f"kept {len(train_names)}/{total} training samples"
        )
        if not train_names:
            raise SystemExit(
                f"max_ref_missing={config['max_ref_missing']:g} filtered out every "
                "training sample; raise the threshold"
            )

    output_dir = args.run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir.occupied(output_dir)
    if existing and not args.resume:
        raise FileExistsError(
            f"{output_dir} already contains experiment artifacts ({', '.join(existing)}); "
            "choose an empty --run-dir"
        )

    model = AzimuthNet(in_channels=run_record.input_channels(config["input_mode"])).to(device)
    # compile 只包前向；checkpoint/优化器用原模型，state_dict 键不带 _orig_mod. 前缀
    compile_enabled = config["compile"] and not args.no_compile
    runtime_model = torch.compile(model) if compile_enabled else model
    parameter_count = count_trainable_parameters(model)
    expected_count = expected_parameter_count(model.in_channels)
    if parameter_count != expected_count:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    record = build_record(
        config,
        args.threads,
        device,
        len(train_names),
        len(val_names),
        names_fingerprint(train_names),
        names_fingerprint(val_names),
        trainable_parameters=parameter_count,
        assets_root=assets_root,
    )
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
        AngleDataset(
            train_dir,
            train_names,
            noise_augment=config["noise_augment"],
            roll_augment=config["roll_augment"],
            input_mode=config["input_mode"],
        ),
        config["batch_size"],
        True,
        config["seed"],
        device,
        train_generator,
    )
    val_loader = make_loader(
        AngleDataset(val_dir, val_names, input_mode=config["input_mode"]),
        config["batch_size"],
        False,
        config["seed"],
        device,
    )

    best_val = float("inf")
    bad_epochs = 0
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    if args.resume:
        recorded = run_dir.load_record(output_dir)
        fresh = {k: v for k, v in record.metadata.items() if k not in _SESSION_KEYS}
        old = {k: recorded.metadata.get(k) for k in fresh}
        if (old, recorded.input_mode) != (fresh, record.input_mode):
            raise SystemExit(f"{output_dir}: train.toml 与 record.json 不一致；换配置请换新目录")
        state = train_artifacts.load_resume_state(run_dir.last_path(output_dir))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_val, bad_epochs = state["epoch"], state["best_val"], state["bad_epochs"]
        rows = load_json(run_dir.history_path(output_dir))["epochs"]
        history = [entry for entry in rows if entry["epoch"] <= start_epoch]
        run_dir.summary_path(output_dir).unlink(missing_ok=True)
    else:
        run_record.write(output_dir, record)
        atomic_json_dump(run_dir.history_path(output_dir), {"epochs": history})
    max_epochs = 1 if args.smoke else config["epochs"]
    for epoch in range(start_epoch, max_epochs):
        train_loss = train_epoch(
            runtime_model,
            train_loader,
            optimizer,
            device,
            config["target_sigma"],
            config["precision"],
        )
        val_loss, val_metrics = eval_loss(
            runtime_model,
            val_loader,
            device,
            config["target_sigma"],
            config["precision"],
        )
        tracked = getattr(val_metrics, Metrics.TRACK_FIELD)
        scheduler.step(tracked)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics.to_payload(),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        improved = tracked < best_val
        if improved:
            best_val = tracked
            bad_epochs = 0
            save_checkpoint(run_dir.checkpoint_path(output_dir), model)
        else:
            bad_epochs += 1
        atomic_json_dump(run_dir.history_path(output_dir), {"epochs": history})
        resume = {"epoch": epoch + 1, "best_val": best_val, "bad_epochs": bad_epochs}
        resume.update(model=model.state_dict(), optimizer=optimizer.state_dict())
        resume["scheduler"] = scheduler.state_dict()
        train_artifacts.save_resume_state(run_dir.last_path(output_dir), resume)
        print(
            f"epoch={epoch + 1}/{max_epochs} train_loss={train_loss:.6f} "
            f"val_{Metrics.TRACK_FIELD}={tracked:.3f}°"
        )
        if not args.smoke and bad_epochs >= config["early_stop_patience"]:
            print("early stopping")
            break

    checkpoint = run_dir.checkpoint_path(output_dir)
    if not checkpoint.exists():
        raise RuntimeError("best checkpoint was not produced")
    best_entry = min(history, key=_tracked_metric)
    # 结算：重新加载 best.pt 并在验证集上重新评估，确保汇报的数字就是
    # 交付 checkpoint 的数字。
    settled_model = load_model(checkpoint, device=device)
    # 与训练期验证同精度，数字才可比
    final_loss, final_metrics = eval_loss(
        settled_model, val_loader, device, config["target_sigma"], config["precision"]
    )
    run_dir.write_summary(
        output_dir,
        run_dir.TrainingSummary(
            epoch=int(best_entry["epoch"]),
            val_count=len(val_names),
            best_val_rms_error=_tracked_metric(best_entry),
            val_loss=final_loss,
            metrics=final_metrics,
        ),
    )
    print(
        f"best val_{Metrics.TRACK_FIELD}={_tracked_metric(best_entry):.3f}° "
        f"(epoch {best_entry['epoch']})"
    )
    print("final evaluation on best.pt (val set):")
    print(f"  val_loss={final_loss:.4f}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in final_metrics.to_payload().items()))
    plot_loss_curves(output_dir, history)


if __name__ == "__main__":
    main()
