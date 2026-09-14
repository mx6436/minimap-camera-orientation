"""校验结论（Finding）：结构断言、bundle 自检与 conformance 共用的报告单元。

独立成 module 是为了让 `endfield/bundle.py` 与 `endfield/conformance.py` 都能持有它，
而不会出现 bundle → conformance 的反向依赖；`export-onnx` 也要对不属于任何 bundle 的图
报结论。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """一条校验结论：level 取 error / warning / info，code 是稳定的短标记。"""

    level: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}
