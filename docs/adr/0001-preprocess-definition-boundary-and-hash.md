# 前处理定义的边界与哈希口径

本仓的前处理定义（`endfield/preprocess.py`）以整帧为输入：整帧到观测 ROI 的裁剪几何（含全透明像素处理）属于定义的一部分，训练、数据生成与 live 都经它生成条带。交付的 `preprocess.onnx` 仍以观测 ROI 为 `minimap` 输入，帧到 ROI 的裁剪由 MaaEnd 侧完成——把裁剪移入图会变更输入契约并需要跨仓协调，不在本次范围内。`definition_hash` 取该定义文件内容的 sha256，因此落入定义的字节关键变换必须全部收在这一个文件里。

## Considered Options

- **多文件哈希（`preprocess.py` + `polar.py` + `ref.py`）**：会让展示几何（`scaled_roi`、`imread_png`）的改动也触发数据失效，且把「哪些文件算定义」重新变成散在调用点的约定。
- **规格常量哈希（只哈希几何常量）**：会漏掉采样逻辑的代码变更，比单文件更不安全。

## Consequences

- 定义收口后 `definition_hash` 变化：既有 `processed*` 缓存与旧 bundle 的图 metadata / manifest 会一次性判定不同源，需重生成数据、重导出 bundle。这是有意的——定义的字节来源变了。
- `observed_roi` 保持公开：它是整帧与 ROI 输入图之间的 seam，conformance 在这一层校验交付图；但管线消费方只使用 frame 级入口。
