"""#23 prototype: current cv2 vs clean vs replica preprocess strips on real samples.

Throwaway. Run:

    uv run python -m prototype.export_drafts
    uv run python -m prototype.compare_real_samples [--limit N | --all] [--timing N]

Compares, per real accepted sample (data/raw + data/locator/locate.jsonl + MapLocator
assets), the current cv2 path (`endfield.ref`, with the per-zone black composite cached;
byte-identity with `ref_strip` is asserted on a few samples) against the draft ONNX
variants under ORT 1.19.2, and against torch eager on a subset.

Outputs (local, data-derived; not committed): out/samples.jsonl, out/summary.json,
out/visuals/*.png (worst samples for the headline comparisons).
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from endfield.data_utils import png_names
from endfield.locate import accept, load_records, zone_asset_path
from endfield.polar import INNER_R, OUTER_R, load_source_bgr, unwrap
from endfield.ref import (
    MAP_ASSETS_ROOT,
    ROI_H,
    ROI_POLE,
    ROI_W,
    compose_observed_backdrop,
    composite_on_black,
    load_reference_image,
    observed_roi,
    ref_strip,
    reference_alpha_plane,
    reference_crop,
    reference_gap_fraction,
    zone_scale,
)
from prototype.preprocess_variants import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LOCATE_PATH = ROOT / "data" / "locator" / "locate.jsonl"

# (base, variant) graphs compared per sample; headline = base is the current cv2 path
COMPARISONS = {
    "clean_ideal_vs_current": ("current", "clean_ideal"),
    "clean_cv2align_vs_current": ("current", "clean_cv2align"),
    "replica_vs_current": ("current", "replica"),
    "clean_cv2align_vs_clean_ideal": ("clean_ideal", "clean_cv2align"),
}
HEADLINE = ("clean_ideal_vs_current", "clean_cv2align_vs_current", "replica_vs_current")
# strips in a comparison: observed 3ch / ref BGR 3ch / ref alpha 1ch / reference 4ch
GROUPS = {
    "observed": lambda obs, ref: obs,
    "ref_bgr": lambda obs, ref: ref[..., :3],
    "ref_alpha": lambda obs, ref: ref[..., 3:4],
    "reference": lambda obs, ref: ref,
}


def current_strips(
    roi: np.ndarray, black: np.ndarray, alpha_plane: np.ndarray, x: float, y: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Byte-identical to endfield.ref.ref_strip with the per-zone black composite cached."""
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


def channel_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return {
        "max": int(diff.max()),
        "mean": float(diff.mean()),
        "p99": float(np.percentile(diff, 99)),
        "frac_any": float((diff > 0).mean()),
        "frac_gt1": float((diff > 1).mean()),
    }


def new_stat() -> dict:
    return {"sample_max": [], "pixel_total": 0, "pixel_any": 0, "pixel_gt1": 0, "channel_max": None}


def record(stats: dict, key: str, actual: np.ndarray, expected: np.ndarray, row: dict) -> dict:
    metrics = channel_metrics(actual, expected)
    row["metrics"][key] = metrics
    entry = stats[key]
    entry["sample_max"].append(metrics["max"])
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    entry["pixel_total"] += diff.size
    entry["pixel_any"] += int((diff > 0).sum())
    entry["pixel_gt1"] += int((diff > 1).sum())
    channel_max = diff.reshape(-1, diff.shape[-1]).max(axis=0)
    entry["channel_max"] = (
        channel_max
        if entry["channel_max"] is None
        else np.maximum(entry["channel_max"], channel_max)
    )
    return metrics


def load_assets(zones: list[str], cache: dict) -> None:
    for zone in zones:
        if zone in cache:
            continue
        asset = load_reference_image(zone_asset_path(zone, MAP_ASSETS_ROOT))
        cache[zone] = (asset, composite_on_black(asset), reference_alpha_plane(asset))


def crop_oob(asset_shape: tuple[int, int], x: float, y: float, scale: float) -> bool:
    width, height = round(ROI_W * scale), round(ROI_H * scale)
    x0 = round(x) - width // 2
    y0 = round(y) - height // 2
    return x0 < 0 or y0 < 0 or x0 + width > asset_shape[1] or y0 + height > asset_shape[0]


def choose_samples(
    records: dict[str, dict], raw_names: set[str], assets: dict, limit: int, seed: int
) -> list[str]:
    accepted = sorted(
        name for name, record in records.items() if accept(record)[0] and name in raw_names
    )
    if limit <= 0 or limit >= len(accepted):
        return accepted
    rng = np.random.default_rng(seed)
    by_zone: dict[str, list[str]] = defaultdict(list)
    for name in accepted:
        by_zone[str(records[name].get("zone", ""))].append(name)
    chosen: list[str] = []
    for _zone, names in sorted(by_zone.items()):
        keep = max(2, round(limit * len(names) / len(accepted)))
        idx = rng.choice(len(names), size=min(keep, len(names)), replace=False)
        chosen.extend(names[i] for i in idx)
    # include out-of-bounds crops, but capped so they do not dominate the distribution
    oob = [
        name
        for name in accepted
        if crop_oob(
            assets[str(records[name].get("zone", ""))][0].shape,
            float(records[name]["x"]),
            float(records[name]["y"]),
            zone_scale(str(records[name].get("zone", ""))),
        )
    ]
    if len(oob) > limit // 4:
        oob = [oob[i] for i in rng.choice(len(oob), size=limit // 4, replace=False)]
    return sorted(set(chosen) | set(oob))


def ort_strips(session, roi: np.ndarray, asset: np.ndarray, x: float, y: float, scale: float):
    outputs = session.run(
        None,
        {
            "minimap": roi[None],
            "asset": asset[None],
            "x": np.array(x, dtype=np.float32),
            "y": np.array(y, dtype=np.float32),
            "scale": np.array(scale, dtype=np.float32),
        },
    )
    return outputs[0][0], outputs[1][0]


def render_worst(
    path: Path,
    base_obs: np.ndarray,
    base_ref: np.ndarray,
    var_obs: np.ndarray,
    var_ref: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        (base_obs, var_obs, "observed"),
        (base_ref[..., :3], var_ref[..., :3], "ref BGR"),
        (base_ref[..., 3], var_ref[..., 3], "ref alpha"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 7))
    fig.suptitle(title, fontsize=9)
    for row, (base, var, label) in enumerate(rows):
        diff = np.abs(base.astype(np.int16) - var.astype(np.int16))
        diff_display = diff.max(axis=2) if diff.ndim == 3 else diff
        for col, image in enumerate((base, var, diff_display)):
            cmap = "gray" if image.ndim == 2 and col < 2 else ("hot" if col == 2 else None)
            axes[row][col].imshow(
                image, cmap=cmap, vmin=0, vmax=255 if col < 2 else max(1, int(diff_display.max()))
            )
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
        axes[row][0].set_title(f"{label} base", fontsize=8)
        axes[row][1].set_title(f"{label} variant", fontsize=8)
        axes[row][2].set_title(f"|diff| max={int(diff_display.max())}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize_stats(key: str, stat: dict) -> dict:
    maxima = np.asarray(stat["sample_max"])
    return {
        "samples": len(maxima),
        "samples_with_diff": int((maxima > 0).sum()),
        "samples_gt1": int((maxima > 1).sum()),
        "sample_max_mean": float(maxima.mean()),
        "sample_max_p99": float(np.percentile(maxima, 99)),
        "sample_max_max": int(maxima.max()),
        "pixel_any_fraction": stat["pixel_any"] / stat["pixel_total"],
        "pixel_gt1_fraction": stat["pixel_gt1"] / stat["pixel_total"],
        "channel_max": [int(v) for v in stat["channel_max"]],
    }


def breakdown(path: Path, keys: list[str], group: str) -> dict:
    """Slice per-sample metrics by scale class / OOB / zone (reads the JSONL once)."""
    out: dict = {
        "by_scale": defaultdict(list),
        "by_oob": defaultdict(list),
        "by_zone": defaultdict(list),
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for key in keys:
            metrics = row["metrics"].get(f"{key}.{group}")
            if metrics is None:
                continue
            scale_class = "scale1" if row["scale"] == 1.0 else "scaled"
            out["by_scale"][scale_class].append((key, metrics["max"]))
            out["by_oob"]["oob" if row["oob"] else "inside"].append((key, metrics["max"]))
            out["by_zone"][row["zone"]].append((key, metrics["max"]))
    result: dict = {}
    for dimension, buckets in out.items():
        result[dimension] = {}
        for bucket, entries in buckets.items():
            per_key: dict[str, list[int]] = defaultdict(list)
            for key, value in entries:
                per_key[key].append(value)
            result[dimension][bucket] = {
                key: {"n": len(values), "max": int(max(values)), "mean": float(np.mean(values))}
                for key, values in per_key.items()
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, default=ROOT / "prototype" / "drafts")
    parser.add_argument("--out", type=Path, default=ROOT / "prototype" / "out")
    parser.add_argument("--limit", type=int, default=600, help="0 = all accepted samples")
    parser.add_argument("--timing", type=int, default=120, help="samples for torch timing")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--visuals", type=int, default=3, help="worst samples per comparison")
    args = parser.parse_args()

    import onnxruntime as ort

    records = load_records(LOCATE_PATH)
    raw_names = set(png_names(RAW_DIR))
    accepted_zones = sorted(
        {str(r.get("zone", "")) for n, r in records.items() if accept(r)[0] and n in raw_names}
    )
    assets: dict[str, tuple] = {}
    load_assets(accepted_zones, assets)
    names = choose_samples(records, raw_names, assets, args.limit, args.seed)
    np.random.default_rng(args.seed).shuffle(names)
    print(f"samples: {len(names)} of {sum(1 for r in records.values() if accept(r)[0])} accepted")

    for name in names[:5]:  # current path self-check
        record_ = records[name]
        zone = str(record_.get("zone", ""))
        asset, black, alpha_plane = assets[zone]
        roi = observed_roi(load_source_bgr(RAW_DIR / name))
        x, y, scale = float(record_["x"]), float(record_["y"]), zone_scale(zone)
        obs, ref = current_strips(roi, black, alpha_plane, x, y, scale)
        full = ref_strip(roi, asset, x, y, scale)
        assert np.array_equal(obs, full[..., :3]) and np.array_equal(ref, full[..., 3:]), name
    print("current path self-check: byte-identical to ref_strip")

    sessions = {
        variant: ort.InferenceSession(
            str(args.drafts / f"preprocess_{variant}.onnx"), providers=["CPUExecutionProvider"]
        )
        for variant in VARIANTS
    }
    torch_models = {variant: factory().eval() for variant, factory in VARIANTS.items()}
    warm_roi = np.zeros((ROI_H, ROI_W, 3), np.uint8)
    warm_asset = np.zeros((140, 160, 4), np.uint8)
    for session in sessions.values():  # warm up before timing
        ort_strips(session, warm_roi, warm_asset, 80.0, 70.0, 1.0)
    timing_names = set(names[: args.timing])
    engine_keys = [f"{variant}_ort_vs_torch" for variant in VARIANTS]

    stats: dict[str, dict] = defaultdict(new_stat)
    gap_delta: dict[str, list[float]] = defaultdict(list)
    worst: dict[str, list[tuple[float, str]]] = defaultdict(list)
    timing: dict[str, list[float]] = defaultdict(list)
    args.out.mkdir(parents=True, exist_ok=True)
    samples_path = args.out / "samples.jsonl"
    handle = samples_path.open("w", encoding="utf-8")

    for index, name in enumerate(names, 1):
        record_ = records[name]
        zone = str(record_.get("zone", ""))
        asset, black, alpha_plane = assets[zone]
        roi = observed_roi(load_source_bgr(RAW_DIR / name))
        x, y, scale = float(record_["x"]), float(record_["y"]), zone_scale(zone)

        t0 = time.perf_counter()
        obs_cur, ref_cur = current_strips(roi, black, alpha_plane, x, y, scale)
        t_current = (time.perf_counter() - t0) * 1e3

        ort_out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for variant, session in sessions.items():
            t1 = time.perf_counter()
            ort_out[variant] = ort_strips(session, roi, asset, x, y, scale)
            timing[f"{variant}_ort_ms"].append((time.perf_counter() - t1) * 1e3)

        torch_out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if name in timing_names:
            with torch.no_grad():
                for variant, model in torch_models.items():
                    t2 = time.perf_counter()
                    out = model(
                        torch.from_numpy(roi[None]),
                        torch.from_numpy(asset[None]),
                        torch.tensor(x),
                        torch.tensor(y),
                        torch.tensor(scale),
                    )
                    timing[f"{variant}_torch_ms"].append((time.perf_counter() - t2) * 1e3)
                    torch_out[variant] = (out[0][0].numpy(), out[1][0].numpy())
            timing["current_cached_ms"].append(t_current)
            t3 = time.perf_counter()
            ref_strip(roi, asset, x, y, scale)
            timing["current_ref_strip_ms"].append((time.perf_counter() - t3) * 1e3)

        strips: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "current": (obs_cur, ref_cur),
            **ort_out,
        }
        row: dict = {
            "name": name,
            "zone": zone,
            "scale": scale,
            "oob": crop_oob(asset.shape, x, y, scale),
            "metrics": {},
        }

        for key, (base, variant) in COMPARISONS.items():
            for group, extract in GROUPS.items():
                record(
                    stats, f"{key}.{group}", extract(*strips[variant]), extract(*strips[base]), row
                )
            score = max(row["metrics"][f"{key}.{group}"]["max"] for group in GROUPS)
            heap = worst[key]
            if len(heap) < args.visuals:
                heapq.heappush(heap, (float(score), name))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (float(score), name))

        for key in engine_keys:
            variant = key[: -len("_ort_vs_torch")]
            if variant not in torch_out:
                continue
            for group, extract in GROUPS.items():
                record(
                    stats,
                    f"{key}.{group}",
                    extract(*ort_out[variant]),
                    extract(*torch_out[variant]),
                    row,
                )

        row["gap"] = {}
        for variant in ("clean_ideal", "clean_cv2align", "replica"):
            gap_cur = reference_gap_fraction(ref_cur)
            gap_var = reference_gap_fraction(ort_out[variant][1])
            row["gap"][variant] = {
                "current": gap_cur,
                "variant": gap_var,
                "delta": abs(gap_cur - gap_var),
            }
            gap_delta[variant].append(abs(gap_cur - gap_var))

        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if index % 200 == 0 or index == len(names):
            print(f"[{index}/{len(names)}] {name}")
    handle.close()

    summary: dict = {
        "samples": len(names),
        "limit": args.limit,
        "seed": args.seed,
        "zones": dict(sorted(Counter(str(records[n].get("zone", "")) for n in names).items())),
        "ort_version": ort.__version__,
        "drafts": {
            variant: str(args.drafts / f"preprocess_{variant}.onnx") for variant in VARIANTS
        },
        "timing_ms": {
            key: {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "n": len(values),
            }
            for key, values in sorted(timing.items())
        },
        "comparisons": {
            key: {
                group: summarize_stats(f"{key}.{group}", stats[f"{key}.{group}"])
                for group in GROUPS
                if f"{key}.{group}" in stats
            }
            for key in list(COMPARISONS) + engine_keys
        },
        "gap_fraction_delta": {
            variant: {
                "samples": len(values),
                "max": float(max(values, default=0.0)),
                "mean": float(np.mean(values)) if values else 0.0,
                "over_tolerance_0.01": int(sum(1 for value in values if value > 0.01)),
            }
            for variant, values in gap_delta.items()
        },
        "worst": {
            key: [name for _, name in sorted(heap, reverse=True)] for key, heap in worst.items()
        },
        "breakdown_ref_bgr": breakdown(samples_path, list(COMPARISONS), "ref_bgr"),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary: {args.out / 'summary.json'}")

    for key in HEADLINE:
        base, variant = COMPARISONS[key]
        for score, name in sorted(worst[key], reverse=True):
            record_ = records[name]
            zone = str(record_.get("zone", ""))
            asset, black, alpha_plane = assets[zone]
            roi = observed_roi(load_source_bgr(RAW_DIR / name))
            x, y, scale = float(record_["x"]), float(record_["y"]), zone_scale(zone)
            if base == "current":
                base_obs, base_ref = current_strips(roi, black, alpha_plane, x, y, scale)
            else:
                base_obs, base_ref = ort_strips(sessions[base], roi, asset, x, y, scale)
            var_obs, var_ref = ort_strips(sessions[variant], roi, asset, x, y, scale)
            render_worst(
                args.out / "visuals" / f"{key}__{name.removesuffix('.png')}.png",
                base_obs,
                base_ref,
                var_obs,
                var_ref,
                f"{key}  {name}  zone={zone}  scale={scale:.4f}  score={score:.0f}",
            )
    print(f"visuals: {args.out / 'visuals'}")


if __name__ == "__main__":
    main()
