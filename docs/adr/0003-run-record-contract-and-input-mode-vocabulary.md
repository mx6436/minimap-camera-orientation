# 运行档案的单一契约与输入模式词汇

运行档案（`record.json`）是训练、实机推理与工件导出共用的唯一运行契约：其消费字段
schema 与输入模式（`polar` / `ref`）词汇——合法模式、模式→输入通道数、ref 的资产根
字段、旧档案默认——由共享的 `endfield/run_record.py` 持有（torch-free）。训练经它写，
实机与交付经它读；checkpoint 输入通道与档案的一致性在该 interface 上核对，实机、
导出与 conformance 不再各自推导。该 module 不持有交付 bundle 词汇（三图文件名、
manifest、role→mode）与数据目录布局；实机与交付不得再出现第二份模式表。

## Considered Options

- **`version` 门控旧默认**（`version` 缺失即报错）：`version` 至今无人读取，使其承重
  会把 write-only 文档变成兼容语义；缺 `input_mode` 字段本身已足以判定 polar 时期产物。
- **保留实机侧 `RunConfig` 投影**：会保留第二份「模式 → 路径 / 通道」推导面；收敛后
  删除，实机直接消费运行档案。
- **把交付 role 映射一并收进该 module**：role（`polar` / `polar_with_ref`）与图文件布局
  属交付 bundle 词汇，应随交付 module 收口，不进运行档案。

## Consequences

- 旧 run（无 `input_mode`）继续可用，默认语义由该 module 单点给出。
- 新增输入模式或改字段名时，训练、实机、导出与 conformance 的改动收敛在一个 module
  与其测试；消费方不再各自解析 `record.json`。
- 磁盘格式不变，`version` 保持 31；原子 JSON 落盘能力独立为 torch-free 的
  `endfield/atomic_io.py`，使运行档案 module 不拖入 torch。
