#!/usr/bin/env python3
"""Data/augmentation ablation experiments. Training loop identical to train.py.

Usages:
  python3 ablate_data.py <fraction> <output_dir> [--aug all|rot|rot_noise|rot_photo|none] [--lr 1e-3] [--train-seed 42] [--stratify-seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from train import (
    SEED,
    NamedDataset,
    eval_loss,
    make_loader,
    seed_everything,
    train_epoch,
)
from data_utils import angle_target, atomic_json_dump, load_rgba, parse_angle, png_names

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 25


class OfflineDataset(Dataset):
    """Offline augmentation: pre-generate ALL rotation copies (including 0deg) once,
    cache as uint8; every epoch shuffles and sees every copy. Noise stays online."""

    def __init__(self, directory: Path, names: list[str], rot_step: int, noise: bool) -> None:
        self.noise = noise
        self.samples: list[tuple[np.ndarray, float]] = []
        for name in names:
            base = load_rgba(directory / name)
            base_angle = parse_angle(Path(name))
            for delta in range(0, 360, rot_step):
                arr = AblateDataset.rotate_rgba(base.astype(np.float32) / 255.0, float(delta))
                arr8 = np.ascontiguousarray((arr * 255.0).round().astype(np.uint8))
                self.samples.append((arr8, (base_angle + delta) % 360))
        print(f"offline pool: {len(names)} images x {360 // rot_step} rotations = {len(self.samples)} samples", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr8, angle = self.samples[index]
        array = arr8.astype(np.float32) / 255.0
        if self.noise:
            rgb = array[..., :3]
            alpha = array[..., 3:4]
            if random.random() < 0.5:
                rgb = rgb + np.random.normal(0.0, 0.02, rgb.shape).astype(np.float32) * (alpha > 0)
                rgb = np.clip(rgb, 0.0, 1.0)
                rgb[alpha[..., 0] == 0] = 0.0
                array = np.concatenate([rgb, alpha], axis=-1)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        target = torch.from_numpy(angle_target(angle))
        return tensor, target


class AblateDataset(Dataset):
    """Same augmentation logic as train.AngleDataset, with per-family switches
    and configurable rotation step (90 = np.rot90 path; 30/15 = PIL BICUBIC)."""

    def __init__(self, directory: Path, names: list[str], rot: bool, photo: bool, noise: bool, rot_step: int = 90) -> None:
        self.directory = directory
        self.names = names
        self.rot = rot
        self.rot_step = rot_step
        self.photo = photo
        self.noise = noise

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def rotate_rgba(array: np.ndarray, delta: float) -> np.ndarray:
        """Rotate an HxWx4 float [0,1] array clockwise by delta degrees.
        Multiples of 90 use lossless np.rot90; other angles use PIL (clockwise = -angle,
        since PIL rotate is counter-clockwise; transparent fill keeps the ring intact)."""
        if abs(delta % 90) < 1e-9:
            return np.rot90(array, k=-int(round(delta / 90)) % 4, axes=(0, 1)).copy()
        img = Image.fromarray((array * 255.0).astype(np.uint8), "RGBA")
        img = img.rotate(-delta, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0, 0))
        return np.asarray(img).astype(np.float32) / 255.0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        angle = parse_angle(Path(name))
        array = load_rgba(self.directory / name).astype(np.float32) / 255.0
        if self.rot:
            if random.random() < 0.5:
                if self.rot_step == 90:
                    delta = 90 * random.choice((1, 2, 3))
                else:
                    delta = self.rot_step * random.randint(1, 360 // self.rot_step - 1)
                array = self.rotate_rgba(array, delta)
                angle = (angle + delta) % 360
        rgb = array[..., :3]
        alpha = array[..., 3:4]
        if self.photo:
            if random.random() < 0.5:
                rgb *= random.uniform(0.85, 1.15)
            if random.random() < 0.5:
                mean = rgb[alpha[..., 0] > 0].mean() if np.any(alpha[..., 0] > 0) else 0.0
                rgb = (rgb - mean) * random.uniform(0.85, 1.15) + mean
        if self.noise:
            if random.random() < 0.5:
                rgb += np.random.normal(0.0, 0.02, rgb.shape).astype(np.float32) * (alpha > 0)
        if self.photo or self.noise:
            rgb = np.clip(rgb, 0.0, 1.0)
            rgb[alpha[..., 0] == 0] = 0.0
            array = np.concatenate([rgb, alpha], axis=-1)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        target = torch.from_numpy(angle_target(angle))
        return tensor, target


def stratified_subset(names: list[str], fraction: float, seed: int = 42) -> list[str]:
    """30-degree-bin stratified subset; larger fractions nest smaller ones."""
    bins: dict[int, list[str]] = defaultdict(list)
    for name in names:
        bins[parse_angle(Path(name)) // 30].append(name)
    rng = random.Random(seed)
    for bucket in bins.values():
        rng.shuffle(bucket)
    selected: list[str] = []
    for bucket in bins.values():
        count = max(1, round(len(bucket) * fraction)) if bucket else 0
        selected.extend(bucket[:count])
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fraction", type=float)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-seed", type=int, default=SEED, help="RNG seed for training")
    parser.add_argument("--stratify-seed", type=int, default=42, help="RNG seed for subset selection")
    parser.add_argument("--aug", choices=["all", "rot", "rot_noise", "rot_photo", "none"], default="all",
                        help="augmentation recipe (default all = rot+photo+noise, same as train.py)")
    parser.add_argument("--rot-step", type=int, choices=[90, 30, 15], default=90,
                        help="rotation augmentation step in degrees (default 90 = np.rot90 path)")
    parser.add_argument("--aug-mode", choices=["online", "offline"], default="online",
                        help="online: random rotation per access (default); offline: pre-generate all rotation copies")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default 1e-3)")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="epoch cap (default 200)")
    args = parser.parse_args()

    seed_everything(args.train_seed)
    device = torch.device("cpu")
    train_names = png_names(TRAIN_DIR)
    val_names = png_names(VAL_DIR)
    subset_names = stratified_subset(train_names, args.fraction, args.stratify_seed)
    print(f"train full={len(train_names)} subset={len(subset_names)} val={len(val_names)} "
          f"aug={args.aug} lr={args.lr}", flush=True)

    rot = args.aug in ("all", "rot", "rot_noise", "rot_photo")
    photo = args.aug in ("all", "rot_photo")
    noise = args.aug in ("all", "rot_noise")
    train_generator = torch.Generator()
    train_generator.manual_seed(args.train_seed)
    if args.aug_mode == "offline":
        if photo:
            raise SystemExit("offline mode does not support photo augmentation")
        if not rot:
            raise SystemExit("offline mode requires rotation augmentation")
        train_loader = make_loader(
            OfflineDataset(TRAIN_DIR, subset_names, args.rot_step, noise),
            BATCH_SIZE, True, args.train_seed, 0, device, train_generator,
        )
    else:
        train_loader = make_loader(
            AblateDataset(TRAIN_DIR, subset_names, rot=rot, photo=photo, noise=noise, rot_step=args.rot_step),
            BATCH_SIZE, True, args.train_seed, 0, device, train_generator,
        )
    val_loader = make_loader(NamedDataset(AblateDataset(VAL_DIR, val_names, rot=False, photo=False, noise=False)),
                             BATCH_SIZE, False, SEED, 0, device)

    from model import AngleCNN, EXPECTED_PARAMETER_COUNT, count_trainable_parameters
    model = AngleCNN()
    assert count_trainable_parameters(model) == EXPECTED_PARAMETER_COUNT
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    criterion = nn.MSELoss()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "ablation",
        "fraction": args.fraction,
        "train_seed": args.train_seed,
        "stratify_seed": args.stratify_seed,
        "aug_mode": args.aug_mode,
        "aug": args.aug,
        "rot_step": args.rot_step,
        "rot": rot, "photo": photo, "noise": noise,
        "train_count": len(subset_names),
        "val_count": len(val_names),
        "hyperparameters": {"optimizer": "AdamW", "learning_rate": args.lr, "weight_decay": 1e-4,
                            "batch_size": BATCH_SIZE, "max_epochs": args.max_epochs,
                            "scheduler": {"name": "ReduceLROnPlateau", "patience": 8, "factor": 0.5, "min_lr": 1e-6},
                            "early_stopping_patience": EARLY_STOP_PATIENCE},
    }
    atomic_json_dump(out_dir / "config.json", config)

    best_val = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(args.max_epochs):
        epoch_t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_loss(model, val_loader, criterion, device)
        scheduler.step(val_metrics["circular_mae"])
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                  **{f"val_{k}": v for k, v in val_metrics.items()},
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        improved = val_metrics["circular_mae"] < best_val
        if improved:
            best_val = val_metrics["circular_mae"]
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")
        else:
            bad_epochs += 1
        atomic_json_dump(out_dir / "history.json", {"epochs": history})
        if (epoch + 1) % 10 == 0 or improved:
            print(f"epoch={epoch + 1}/{args.max_epochs} train_loss={train_loss:.4f} val_mae={val_metrics['circular_mae']:.2f} "
                  f"lr={optimizer.param_groups[0]['lr']:.1e} ({time.time() - epoch_t0:.1f}s/epoch, {time.time() - t0:.0f}s total)", flush=True)
        if bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break

    best_record = min(history, key=lambda r: r["val_circular_mae"])
    summary = {"fraction": args.fraction, "aug_mode": args.aug_mode, "aug": args.aug, "rot_step": args.rot_step, "lr": args.lr,
               "best_val_mae": best_record["val_circular_mae"], "best_epoch": best_record["epoch"],
               "last_epoch": len(history),
               "val_median": best_record["val_circular_median"],
               "within_5": best_record["val_within_5_degrees"],
               "within_10": best_record["val_within_10_degrees"]}
    atomic_json_dump(out_dir / "summary.json", summary)
    print(f"DONE {summary}", flush=True)


if __name__ == "__main__":
    main()
