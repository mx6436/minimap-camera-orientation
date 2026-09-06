"""run 产物的落盘：checkpoint 与 loss 曲线图，均经原子替换写入。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from endfield.data_utils import atomic_path
from endfield.model import ARCH_VERSION

ARTIFACT_NAMES = ("best.pt", "record.json", "history.json", "summary.json")


def save_checkpoint(path: Path, model: nn.Module) -> None:
    checkpoint = {"model": model.state_dict(), "arch": ARCH_VERSION}
    with atomic_path(path) as temp:
        torch.save(checkpoint, temp)


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
    ax.set_ylabel("loss (KL)")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "loss_curve.png"
    try:
        with atomic_path(path) as temp:
            fig.savefig(temp, dpi=150, format="png")
    finally:
        plt.close(fig)
    print(f"saved loss curve: {path}")
