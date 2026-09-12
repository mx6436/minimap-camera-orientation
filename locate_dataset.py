"""对 data/raw 全量跑 MapLocator 批量定位，产出 data/locator/locate.jsonl。

用法:
    uv run locate_dataset.py [--jobs 4] [--limit N] [--no-retry-failed]

定位 CLI 与资源默认取 gitignored 的 local/maplocator/（来源与重建见该目录的
README.local.md）；本脚本只引用仓库内路径。

断点续跑：已成功的样本跳过；失败项默认重跑（结果覆盖旧记录）。重复运行幂等。
产物 `locate.jsonl` 每行一条记录，另有 `summary.json` 汇总（失败分类、调用次数
分布、locConf 分布、按命名族成功率）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from endfield.locate import (
    accept,
    load_records,
    merge_records,
    pending_names,
    run_cli,
    split_shards,
    summarize,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "locator"
OUT_PATH = OUT_DIR / "locate.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
DEFAULT_CLI = ROOT / "local" / "maplocator" / "bin" / "map-locate"
DEFAULT_RESOURCE_DIR = ROOT / "local" / "maplocator" / "resource"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI, help="map-locate 可执行文件")
    parser.add_argument(
        "--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR, help="MapLocator 资源目录"
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="产物 JSONL 路径")
    parser.add_argument("--jobs", type=int, default=4, help="并行定位进程数")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本（0 = 全部）")
    parser.add_argument("--no-retry-failed", action="store_true", help="已有记录（含失败）全部跳过")
    return parser.parse_args()


def run_shards(
    cli: Path,
    resource_dir: Path,
    todo: list[str],
    raw_dir: Path,
    jobs: int,
    parts_dir: Path,
) -> list[dict]:
    """把待定位清单切成 jobs 片并行跑，返回各片产物的合并记录。"""
    shards = split_shards(todo, jobs)
    parts_dir.mkdir(parents=True, exist_ok=True)

    def run_one(index: int, shard: list[str]) -> Path:
        part = parts_dir / f"part_{index:02d}.jsonl"
        paths = [str(raw_dir / name) for name in shard]

        def progress(count: int, total: int) -> None:
            print(f"  shard {index}: {count}/{total}", file=sys.stderr)

        start = time.monotonic()
        count = run_cli([cli], resource_dir, paths, part, progress=progress)
        print(f"shard {index}: {count} images in {time.monotonic() - start:.1f}s", file=sys.stderr)
        return part

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        part_paths = list(pool.map(lambda item: run_one(*item), enumerate(shards)))

    records: list[dict] = []
    for path in part_paths:
        records.extend(load_records(path).values())
    return records


def print_summary(summary: dict) -> None:
    print(
        f"total={summary['total']} ok={summary['ok']} accepted={summary['accepted']} "
        f"failed={summary['failed']}"
    )
    if summary["excluded_by_reason"]:
        print(f"  excluded: {summary['excluded_by_reason']}")
    print(f"  attempts: {summary['attempts']}")
    locconf = summary["locconf"]
    if locconf:
        print(
            f"  locConf: median={locconf['median']:.3f} p25={locconf['p25']:.3f} "
            f"p75={locconf['p75']:.3f} below0.85={locconf['below_high_conf']} "
            f"below0.55={locconf['below_loc_threshold']} held={summary['held']}"
        )
    print("  families:")
    for family, counts in summary["families"].items():
        rate = counts["accepted"] / counts["total"] if counts["total"] else 0.0
        print(
            f"    {family}: ok={counts['ok']}/{counts['total']} "
            f"accepted={counts['accepted']} ({rate:.1%})"
        )


def main() -> None:
    args = parse_args()
    if not args.cli.exists():
        raise SystemExit(f"定位 CLI 不存在: {args.cli}（见 local/maplocator/README.local.md）")
    if not args.resource_dir.is_dir():
        raise SystemExit(f"资源目录不存在: {args.resource_dir}")

    names = sorted(path.name for path in RAW_DIR.glob("*.png"))
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit(f"没有样本: {RAW_DIR}")

    done = load_records(args.out)
    todo = pending_names(names, done, retry_failed=not args.no_retry_failed)
    print(f"raw={len(names)} done={len(done)} pending={len(todo)} jobs={args.jobs}")

    if todo:
        new_records = run_shards(
            args.cli, args.resource_dir, todo, RAW_DIR, args.jobs, args.out.parent / "parts"
        )
        merged = merge_records(done.values(), new_records)
        for record in merged:
            record["accepted"], record["accept_reason"] = accept(record)
        write_jsonl(args.out, merged)

    summary = summarize(load_records(args.out).values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_summary(summary)
    print(f"-> {args.out}")
    print(f"-> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
