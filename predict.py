"""对单张原始截图 PNG 预测摄像机角度。"""

from __future__ import annotations

import argparse
from pathlib import Path

import endfield.polar as polar
from endfield import preprocess
from endfield.live import to_base_frame
from endfield.model import choose_device, load_model, predict_angle
from endfield.ref import observed_roi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_model(args.run_dir / "best.pt", device=device)

    frame = polar.load_source_bgr(args.image)
    strip = preprocess.observed_strip(observed_roi(to_base_frame(frame)))
    angle, _ = predict_angle(model, strip)
    print(f"angle: {angle:.6f}")


if __name__ == "__main__":
    main()
