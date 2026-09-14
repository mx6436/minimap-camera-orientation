"""#23 corrected evaluation: per-scheme `observed` <-> `ref` pair consistency.

DEPRECATED (#44): #23 已闭合；本脚本依赖的旧 cv2 前处理 API（`endfield.polar.unwrap`、
`endfield.ref.reference_crop` 等）已在 #25 删除，不再可运行，仅作方法与结论存档
（见 prototype/README.md）；原始目录已更新为 `data/train_raw` / `data/val_raw` 并集，
但实现不再维护。

Cross-scheme strip deltas are expected — different definitions must differ; they cannot
rank schemes. What must hold is that each scheme's `ref` stays *paired* with its own
`observed` strip (same world content, same strip geometry). This script measures, per real
accepted sample and per scheme (current cv2, clean_ideal, clean_cv2align, replica):

- `zero_ncc`: NCC between obs luma and ref luma over fully-present pixels (alpha == 255)
  at zero shift — registration quality;
- `peak_ncc` / `shift`: best integer shift (x circular, y radial) and NCC there;
- `resid`: mean |obs - ref| over present pixels, and over the alpha transition band;
- `zero_alpha_resid`: max |ref - obs| where alpha == 0 (must be ~0: missing reference
  copies the observation);
- `valid_frac`: fraction of the strip with alpha == 255.

Aggregates are paired against the current scheme on the same samples, and broken down by
fractional center distance |x - round(x)| (the subpixel realignment hypothesis) and by
scale class / OOB. Writes locally (gitignored): out/pairing.jsonl, out/pairing_summary.json,
out/pairing_visuals/*.png.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from endfield.data_utils import union_png_samples
from endfield.locate import accept, load_records, record_scale, zone_asset_path
from endfield.polar import INNER_R, OUTER_R, load_source_bgr, unwrap
from endfield.ref import (
    MAP_ASSETS_ROOT,
    ROI_POLE,
    compose_observed_backdrop,
    composite_on_black,
    load_reference_image,
    observed_roi,
    reference_alpha_plane,
    reference_crop,
)
from prototype.compare_real_samples import crop_oob, ort_strips
from prototype.preprocess_variants import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = (ROOT / "data" / "train_raw", ROOT / "data" / "val_raw")
LOCATE_PATH = ROOT / "data" / "locator" / "locate.jsonl"
SCHEMES = ("current", *VARIANTS)
PAD, MAX_SHIFT, MIN_PIXELS = 5, 4, 400


def raw_samples() -> dict[str, Path]:
    """两侧原始目录并集（目录即划分；脚本废弃，仅保持路径引用与现行契约一致）。"""
    return union_png_samples(RAW_DIRS)


def current_strips(roi, black, alpha_plane, x, y, scale):
    bgr_crop = reference_crop(black, x, y, scale)
    alpha_crop = reference_crop(alpha_plane, x, y, scale)
    composed = compose_observed_backdrop(bgr_crop, alpha_crop, roi)
    reference = np.dstack(
        [
            unwrap(composed, *ROI_POLE, INNER_R, OUTER_R),
            unwrap(alpha_crop, *ROI_POLE, INNER_R, OUTER_R),
        ]
    )
    return unwrap(roi, *ROI_POLE, INNER_R, OUTER_R), reference


def pair_metrics(obs: np.ndarray, ref: np.ndarray) -> dict:
    """`observed` (42,360,3) vs `ref` (42,360,4): NCC / shift / residuals."""
    luma_obs = obs.astype(np.float32).mean(axis=2)
    luma_ref = ref[..., :3].astype(np.float32).mean(axis=2)
    alpha = ref[..., 3]
    present = alpha == 255
    band = (alpha > 0) & (alpha < 255)
    zero = alpha == 0

    def ncc(selector_obs, selector_ref):
        a = selector_obs - selector_obs.mean()
        b = selector_ref - selector_ref.mean()
        denominator = float(np.sqrt(float((a * a).sum()) * float((b * b).sum())))
        return None if denominator <= 0 else float((a * b).sum()) / denominator

    base_obs = luma_obs[PAD : 42 - PAD]
    zero_ncc = None
    peak_ncc = None
    peak_shift = (0, 0)
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        rows = slice(PAD + dy, 42 - PAD + dy)
        if rows.start < 0 or rows.stop > 42:
            continue
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            shifted = np.roll(luma_ref, -dx, axis=1)[rows]
            shifted_mask = np.roll(present, -dx, axis=1)[rows]
            if int(shifted_mask.sum()) < MIN_PIXELS:
                continue
            value = ncc(base_obs[shifted_mask], shifted[shifted_mask])
            if value is None:
                continue
            if dx == 0 and dy == 0:
                zero_ncc = value
            if peak_ncc is None or value > peak_ncc:
                peak_ncc, peak_shift = value, (dx, dy)

    residual = np.abs(obs.astype(np.int16) - ref[..., :3].astype(np.int16)).mean(axis=2)
    return {
        "zero_ncc": zero_ncc,
        "peak_ncc": peak_ncc,
        "shift": list(peak_shift),
        "valid_frac": float(present.mean()),
        "resid_present": float(residual[present].mean()) if present.any() else None,
        "resid_band": float(residual[band].mean()) if band.any() else None,
        "zero_alpha_max": int(residual[zero].max()) if zero.any() else None,
    }


def render(path: Path, obs_current, ref_current, obs_variant, ref_variant, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("obs current", obs_current),
        ("ref current", ref_current[..., :3]),
        (
            "|obs-ref| current",
            np.abs(obs_current.astype(np.int16) - ref_current[..., :3].astype(np.int16)).max(
                axis=2
            ),
        ),
        ("obs variant", obs_variant),
        ("ref variant", ref_variant[..., :3]),
        (
            "|obs-ref| variant",
            np.abs(obs_variant.astype(np.int16) - ref_variant[..., :3].astype(np.int16)).max(
                axis=2
            ),
        ),
    ]
    fig, axes = plt.subplots(6, 1, figsize=(12, 9))
    fig.suptitle(title, fontsize=9)
    for ax, (label, image) in zip(axes, rows, strict=True):
        cmap = "hot" if label.startswith("|") else None
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax.set_ylabel(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, default=ROOT / "prototype" / "drafts")
    parser.add_argument("--out", type=Path, default=ROOT / "prototype" / "out")
    parser.add_argument("--limit", type=int, default=0, help="0 = all accepted samples")
    parser.add_argument("--visuals", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    import onnxruntime as ort

    records = load_records(LOCATE_PATH)
    samples = raw_samples()
    raw_names = set(samples)
    accepted = sorted(
        name for name, record in records.items() if accept(record)[0] and name in raw_names
    )
    if args.limit:
        rng = np.random.default_rng(args.seed)
        accepted = [accepted[i] for i in rng.choice(len(accepted), size=args.limit, replace=False)]

    zones = sorted({str(records[n].get("zone", "")) for n in accepted})
    assets = {z: load_reference_image(zone_asset_path(z, MAP_ASSETS_ROOT)) for z in zones}
    black_cache = {z: composite_on_black(a) for z, a in assets.items()}
    alpha_cache = {z: reference_alpha_plane(a) for z, a in assets.items()}
    sessions = {
        variant: ort.InferenceSession(
            str(args.drafts / f"preprocess_{variant}.onnx"), providers=["CPUExecutionProvider"]
        )
        for variant in VARIANTS
    }
    print(f"samples: {len(accepted)}")

    rows = []
    gains: list[tuple[float, str]] = []
    for index, name in enumerate(accepted, 1):
        record = records[name]
        zone = str(record.get("zone", ""))
        asset = assets[zone]
        roi = observed_roi(load_source_bgr(samples[name]))
        x, y, scale = float(record["x"]), float(record["y"]), record_scale(record)
        current = current_strips(roi, black_cache[zone], alpha_cache[zone], x, y, scale)
        strips = {"current": current}
        for variant in VARIANTS:
            strips[variant] = ort_strips(sessions[variant], roi, asset, x, y, scale)

        row = {
            "name": name,
            "zone": zone,
            "scale": scale,
            "oob": crop_oob(asset.shape, x, y, scale),
            "frac_dist": float(np.hypot(x - round(x), y - round(y))),
            "metrics": {scheme: pair_metrics(*strips[scheme]) for scheme in SCHEMES},
        }
        rows.append(row)
        gain = (row["metrics"]["clean_ideal"]["zero_ncc"] or -1) - (
            row["metrics"]["current"]["zero_ncc"] or -1
        )
        gains.append((gain, name))
        if index % 500 == 0 or index == len(accepted):
            print(f"[{index}/{len(accepted)}] {name}")

    out_path = args.out / "pairing.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: dict = {
        "samples": len(rows),
        "schemes": list(SCHEMES),
        "scheme_metrics": {},
        "paired_vs_current": {},
        "by_frac_dist": {},
        "by_scale": {},
        "by_zone_top": {},
        "shift_hist": {},
    }
    for scheme in SCHEMES:
        values = {
            key: [r["metrics"][scheme][key] for r in rows if r["metrics"][scheme][key] is not None]
            for key in ("zero_ncc", "peak_ncc", "resid_present", "resid_band", "valid_frac")
        }
        summary["scheme_metrics"][scheme] = {
            key: {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            }
            for key, vals in values.items()
        }
        summary["shift_hist"][scheme] = dict(
            Counter(
                f"{r['metrics'][scheme]['shift'][0]},{r['metrics'][scheme]['shift'][1]}"
                for r in rows
            )
        )
        shifts = [tuple(r["metrics"][scheme]["shift"]) for r in rows]
        summary["scheme_metrics"][scheme]["shift_zero_frac"] = float(
            sum(1 for s in shifts if s == (0, 0)) / len(shifts)
        )
        max_zero = [
            r["metrics"][scheme]["zero_alpha_max"]
            for r in rows
            if r["metrics"][scheme]["zero_alpha_max"] is not None
        ]
        summary["scheme_metrics"][scheme]["zero_alpha_max_of_max"] = (
            int(max(max_zero)) if max_zero else None
        )

    for scheme in SCHEMES:
        if scheme == "current":
            continue
        deltas = {
            "zero_ncc": [
                r["metrics"][scheme]["zero_ncc"] - r["metrics"]["current"]["zero_ncc"]
                for r in rows
                if r["metrics"][scheme]["zero_ncc"] is not None
                and r["metrics"]["current"]["zero_ncc"] is not None
            ],
            "resid_present": [
                r["metrics"][scheme]["resid_present"] - r["metrics"]["current"]["resid_present"]
                for r in rows
                if r["metrics"][scheme]["resid_present"] is not None
            ],
        }
        summary["paired_vs_current"][scheme] = {
            key: {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "better_frac": float(np.mean(np.asarray(vals) > 0))
                if key == "zero_ncc"
                else float(np.mean(np.asarray(vals) < 0)),
            }
            for key, vals in deltas.items()
        }

    def bucket(bounds, value):
        for low, high in zip(bounds[:-1], bounds[1:], strict=True):
            if low <= value < high:
                return f"[{low},{high})"
        return f">={bounds[-1]}"

    for dimension, selector in (
        ("by_frac_dist", lambda r: bucket((0.0, 0.1, 0.25, 0.4, 0.5), r["frac_dist"])),
        ("by_scale", lambda r: "scale1" if r["scale"] == 1.0 else "scaled"),
    ):
        grouped: dict[str, list] = defaultdict(list)
        for row in rows:
            grouped[selector(row)].append(row)
        for bucket_name, bucket_rows in sorted(grouped.items()):
            entry = {}
            for scheme in ("clean_ideal", "clean_cv2align", "replica"):
                deltas = [
                    r["metrics"][scheme]["zero_ncc"] - r["metrics"]["current"]["zero_ncc"]
                    for r in bucket_rows
                    if r["metrics"][scheme]["zero_ncc"] is not None
                    and r["metrics"]["current"]["zero_ncc"] is not None
                ]
                entry[scheme] = {
                    "n": len(deltas),
                    "mean_delta": float(np.mean(deltas)) if deltas else None,
                    "better_frac": float(np.mean(np.asarray(deltas) > 0)) if deltas else None,
                }
            summary[dimension][bucket_name] = entry

    zone_counts = Counter(r["zone"] for r in rows)
    for zone, _ in zone_counts.most_common(8):
        zone_rows = [r for r in rows if r["zone"] == zone]
        entry = {"n": len(zone_rows)}
        for scheme in ("current", "clean_ideal", "clean_cv2align", "replica"):
            pairs = [
                (r["metrics"][scheme]["zero_ncc"], r["metrics"]["current"]["zero_ncc"])
                for r in zone_rows
                if r["metrics"][scheme]["zero_ncc"] is not None
                and r["metrics"]["current"]["zero_ncc"] is not None
            ]
            entry[scheme] = {
                "n": len(pairs),
                "mean_zero_ncc": float(np.mean([p[0] for p in pairs])) if pairs else None,
                "mean_delta_vs_current": (
                    0.0 if scheme == "current" else float(np.mean([p[0] - p[1] for p in pairs]))
                )
                if pairs
                else None,
            }
        summary["by_zone_top"][zone] = entry

    summary["visuals"] = {
        "gain": [name for _, name in sorted(gains, reverse=True)[: args.visuals]],
        "loss": [name for _, name in sorted(gains)[: args.visuals]],
    }
    (args.out / "pairing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary: {args.out / 'pairing_summary.json'}")

    for title, names in summary["visuals"].items():
        for name in names:
            record = records[name]
            zone = str(record.get("zone", ""))
            asset = assets[zone]
            roi = observed_roi(load_source_bgr(samples[name]))
            x, y, scale = float(record["x"]), float(record["y"]), record_scale(record)
            obs_cur, ref_cur = current_strips(
                roi, black_cache[zone], alpha_cache[zone], x, y, scale
            )
            obs_var, ref_var = ort_strips(sessions["clean_ideal"], roi, asset, x, y, scale)
            render(
                args.out / "pairing_visuals" / f"{title}__{name.removesuffix('.png')}.png",
                obs_cur,
                ref_cur,
                obs_var,
                ref_var,
                f"{title}  {name}  zone={zone}  x={x:.2f} y={y:.2f} scale={scale:.4f}",
            )
    print(f"visuals: {args.out / 'pairing_visuals'}")


if __name__ == "__main__":
    main()
