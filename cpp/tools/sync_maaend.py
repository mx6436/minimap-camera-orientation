"""Sync the C++ preprocess port (cpp/src/) into a MaaEnd checkout, or check that it is in sync.

cpp/src/ is the only copy that gets edited. MaaEnd carries the same three files verbatim under
agent/cpp-algo/source/MapLocator/; the only MaaEnd-specific include (<MaaUtils/NoWarningCV.hpp>)
is stubbed by cpp/compat/ for the standalone build here.

Run from the repo root:
  uv run python cpp/tools/sync_maaend.py --maaend F:/Project/Golang/MaaEnd [--check]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "cpp" / "src"
FILES = (
    "CameraOrientationPreprocess.h",
    "CameraOrientationPreprocess.cpp",
    "CameraOrientationAzimuthTable.inc",
)
MAAEND_DIR = Path("agent/cpp-algo/source/MapLocator")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--maaend", type=Path, required=True, help="MaaEnd checkout")
    ap.add_argument("--check", action="store_true", help="only report differences, do not copy")
    a = ap.parse_args()

    dest = a.maaend / MAAEND_DIR
    if not dest.is_dir():
        print(f"{dest} not found; is {a.maaend} a MaaEnd checkout?")
        return 2

    stale = []
    for name in FILES:
        ours = (SRC / name).read_bytes()
        target = dest / name
        if target.exists() and target.read_bytes() == ours:
            continue
        stale.append(name)
        if not a.check:
            target.write_bytes(ours)
            print(f"wrote {target}")

    if a.check and stale:
        print(f"out of sync with {dest}: {', '.join(stale)}")
        print("rerun without --check to copy cpp/src/ into MaaEnd")
        return 1
    if not stale:
        print(f"{dest} in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
