"""对单张原始截图 PNG 预测摄像机角度。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import polar
from data_utils import decode_angle
from model import choose_device, load_model

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "runs" / "production_001" / "best.pt"
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device=device)

    frame = polar.load_source_rgb(args.image)
    cx, cy, r_in, r_out = polar.scaled_roi(frame.shape[:2])
    array = polar.unwrap(frame, cx, cy, r_in, r_out).astype(np.float32) / 255.0
    features = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(features).cpu().numpy()
    angle = float(decode_angle(output)[0])
    print(f"angle: {angle:.6f}")


if __name__ == "__main__":
    main()
