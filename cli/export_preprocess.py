"""导出 `preprocess.onnx`（前处理定义模块的唯一导出入口）。

用法：

    uv run export-preprocess --out runs/<name>/bundle/preprocess.onnx

图契约（输入名、动态维、输出角色、几何与合成语义）见 `endfield/preprocess.py`
与导出图的 metadata；`manifest.json`（#28）消费 `definition_hash()`。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from endfield import preprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("preprocess.onnx"), help="输出文件路径")
    args = parser.parse_args(argv)
    path = preprocess.export_onnx(args.out)
    print(f"exported: {path} (definition_hash={preprocess.definition_hash()[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
