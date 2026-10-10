"""Generate fixtures for the C++ preprocess test (cpp/tests/fixture_test.cpp).

Each case stores raw inputs and two expected outputs:
  def_*  endfield/preprocess.py (torch definition, source of truth)
  ort_*  preprocess.onnx (freshly exported from the definition, or --onnx) run by onnxruntime

Case groups:
  conf_*  the 8 conformance scenarios (endfield/conformance.builtin_scenarios)
  synth_* random positions / scales (incl. out of bounds) on a synthetic 4-channel asset with varying alpha
  real_*  (optional, --maaend) real 720p minimap ROIs from MaaEnd test screenshots on a real zone asset
  rand_*  (optional, --maaend) random positions / scales on the real zone asset

Run from the repo root:
  uv run python cpp/tools/gen_fixtures.py --out cpp/build/fixtures [--maaend F:/Project/Golang/MaaEnd]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from endfield import conformance as cf  # noqa: E402
from endfield import preprocess as pp  # noqa: E402

# MaaEnd agent/cpp-algo/source/MapLocator/MapTypes.h kDefaultMinimapRoi (x, y, w, h) at 720p
MINIMAP_ROI = (49, 51, pp.ROI_W, pp.ROI_H)
# MaaEnd zone asset and positions seen in a Wuling_Base AutoCollect run
REAL_ASSET = "assets/resource/image/MapLocator/Wuling/Base.png"
REAL_POSITIONS = [(941.2, 1779.21), (941.19, 1779.21), (1010.5, 1650.25), (520.0, 900.0)]
REAL_SHOTS = "tests/MaaEndTestset/ADB/Official_CN"


class Writer:
    def __init__(self, out: Path, onnx_path: Path):
        import onnxruntime as ort

        self.out = out
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.shared: dict[int, str] = {}
        self.cases: list[str] = []

    def _shared_asset(self, asset: np.ndarray) -> str:
        key = id(asset)
        if key not in self.shared:
            name = f"asset{len(self.shared):02d}_{asset.shape[1]}x{asset.shape[0]}"
            (self.out / "shared").mkdir(parents=True, exist_ok=True)
            asset.tofile(self.out / "shared" / f"{name}.bin")
            self.shared[key] = name
        return self.shared[key]

    def case(self, name, minimap, asset, x, y, scale, tags, shared=False):
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        if not shared:
            asset = np.ascontiguousarray(cf.normalize_asset(asset))
        minimap = np.ascontiguousarray(minimap)
        def_obs, def_ref = pp.strip_pair(minimap, asset, x, y, scale)
        f32 = np.float32
        feeds = {
            "minimap": minimap[None],
            "asset": asset[None],
            "x": np.array(x, f32),
            "y": np.array(y, f32),
            "scale": np.array(scale, f32),
        }
        ort_obs, ort_ref = self.session.run(None, feeds)
        minimap.tofile(d / "minimap.bin")
        asset_ref = self._shared_asset(asset) if shared else ""
        if not shared:
            asset.tofile(d / "asset.bin")
        def_obs.tofile(d / "def_observed.bin")
        def_ref.tofile(d / "def_reference.bin")
        ort_obs[0].tofile(d / "ort_observed.bin")
        ort_ref[0].tofile(d / "ort_reference.bin")
        meta = {
            "name": name,
            "asset_h": int(asset.shape[0]),
            "asset_w": int(asset.shape[1]),
            "x": float(x),
            "y": float(y),
            "scale": float(scale),
            "asset_ref": asset_ref,
            "tags": list(tags),
        }
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.cases.append(name)


def random_positions(rng, h, w, count, margin=80):
    for _ in range(count):
        x = float(rng.uniform(-margin, w + margin))
        y = float(rng.uniform(-margin, h + margin))
        scale = float(rng.choice([1.0, 15.0 / 16.0, 0.75, 1.25, float(rng.uniform(0.5, 1.5))]))
        yield x, y, scale


def imread(path: Path, flags):
    import cv2

    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "cpp" / "build" / "fixtures")
    ap.add_argument("--onnx", type=Path, help="preprocess.onnx to compare against (default: export from the definition)")
    ap.add_argument("--maaend", type=Path, help="MaaEnd checkout for real minimap ROIs and zone asset")
    ap.add_argument("--synth", type=int, default=120, help="random cases on the synthetic asset")
    ap.add_argument("--random", type=int, default=200, help="random cases on the real asset (needs --maaend)")
    ap.add_argument("--seed", type=int, default=20261010)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    # the test also reads definition_hash back from <out>/preprocess.onnx (MaaEnd's predictor gate)
    onnx_path = a.onnx
    if onnx_path is None:
        onnx_path = pp.export_onnx(a.out / "preprocess.onnx")
    elif onnx_path.resolve() != (a.out / "preprocess.onnx").resolve():
        shutil.copyfile(onnx_path, a.out / "preprocess.onnx")
    w = Writer(a.out, onnx_path)
    rng = np.random.default_rng(a.seed)

    for sc in cf.builtin_scenarios():
        w.case(f"conf_{sc.name}", sc.minimap, sc.asset, sc.x, sc.y, sc.scale, ("conformance", *sc.tags))

    # synthetic: conformance texture as minimap, 4-channel asset with smooth alpha in [0, 255]
    synth_mm = cf.builtin_scenarios()[0].minimap
    synth_asset = np.ascontiguousarray(
        np.dstack([rng.integers(0, 256, (600, 520, 3), dtype=np.uint8), rng.integers(0, 256, (600, 520), dtype=np.uint8)])
    )
    for i, (x, y, s) in enumerate(random_positions(rng, *synth_asset.shape[:2], a.synth)):
        mm = synth_mm if i % 2 == 0 else rng.integers(0, 256, (pp.ROI_H, pp.ROI_W, 3), dtype=np.uint8)
        w.case(f"synth_{i:03d}", mm, synth_asset, x, y, s, ("synthetic",), shared=True)

    if a.maaend:
        import cv2

        asset = np.ascontiguousarray(cf.normalize_asset(imread(a.maaend / REAL_ASSET, cv2.IMREAD_UNCHANGED)))
        rx, ry, rw, rh = MINIMAP_ROI
        rois = []
        for shot in sorted((a.maaend / REAL_SHOTS).glob("*.png")):
            img = imread(shot, cv2.IMREAD_COLOR)
            if img is not None and img.shape[:2] == (720, 1280):
                rois.append(img[ry : ry + rh, rx : rx + rw])
            if len(rois) == 8:
                break
        for i, roi in enumerate(rois):
            x, y = REAL_POSITIONS[i % len(REAL_POSITIONS)]
            w.case(f"real_{i:02d}", roi, asset, x, y, 1.0, ("real",), shared=True)
        for i, (x, y, s) in enumerate(random_positions(rng, *asset.shape[:2], a.random)):
            mm = rois[0] if i % 2 == 0 else rng.integers(0, 256, (rh, rw, 3), dtype=np.uint8)
            w.case(f"rand_{i:03d}", mm, asset, x, y, s, ("random",), shared=True)

    index = {"definition_hash": pp.definition_hash(), "onnx": str(onnx_path), "cases": w.cases}
    (a.out / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"definition_hash={pp.definition_hash()}  onnx={onnx_path}")
    print(f"wrote {len(w.cases)} cases to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
