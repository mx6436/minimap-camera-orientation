"""对单张原始截图 PNG 预测摄像机角度。"""

from __future__ import annotations

import argparse
from pathlib import Path

import endfield.polar as polar
from endfield.model import choose_device, load_model, predict_angle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_model(args.run_dir / "best.pt", device=device)

    frame = polar.load_source_bgr(args.image)
    cx, cy, r_in, r_out = polar.scaled_roi(frame.shape[:2])
    angle, _ = predict_angle(model, polar.unwrap(frame, cx, cy, r_in, r_out))
    print(f"angle: {angle:.6f}")


if __name__ == "__main__":
    main()
