# 底图定位独立为 `placement/` 包，张量编码留在 `endfield/`

本仓分两个顶层包：`endfield/` 持有模型、训练、前处理定义与交付校验；`placement/` 持有
底图定位（CONTEXT.md 词条）——对 MapLocator 定位记录与本地工作台资产的消费面，产出参考
条带对。依赖方向单向：`placement → endfield`（取前处理定义与帧解码），反向不得依赖。
缝划在**条带域与张量域之间**：`placement` 只出 `(观测条带, 参考条带)`，7 通道张量编码
`[obs.BGR, ref.BGR, ref.A]` 与参考缺失占比留在 `endfield` 一侧。

## Considered Options

- **原地加深（在 `endfield/locate.py` 里加一个类型）**：定位记录的解读仍与模型、训练、
  交付混在同一个包里，两个关注点没有 seam 可验证。
- **`endfield/reference/` 子包**：给这个面起了名字，但 `endfield/` 仍同时拥有「从画面学
  角度」与「消费 MaaEnd 定位资产」两件事，依赖方向不可断言。
- **编码跟着采样走去 `placement`**：`endfield/train/data.py` 读参考数据要用拼接与缺失
  占比，于是 `endfield.train → placement → endfield.preprocess` 成环。
- **训练侧的参考读取整体搬进 `placement`**：把模型输入面的一半搬出 `endfield`，缝划错位置。

## Consequences

- `endfield/live.py` 删除：`ref_pair_at` 归 `placement/sample.py`，`to_base_frame` 归
  `endfield/polar.py`，`MissingZoneAsset` 归 `placement/sample.py`。
- `endfield/model.py` 与 `endfield/train/record.py` 直连前处理定义取 `IMG_H`/`IMG_W`，
  `polar.py` 不再转发几何常量——它是「帧解码 + 基准缩放 + 展示几何」的家，仍属 ADR 0001
  划在定义哈希之外的展示面。
- `REF_SUBDIR` 与 `data/` 树归 `endfield/dataset.py`：它是数据集布局，训练与数据集生成
  共用，不属于定位面，也不属于运行档案（ADR 0003）。
- `endfield/preprocess.py` 位置与字节都不变，`definition_hash` 随之不变：既有 `processed*`
  缓存与 bundle 仍同源，搬迁无需重生成数据。
- 8 个 CLI 命令收进顶层 `cli/` 包，两个包成为纯库：`endfield/conformance.py` 不再 import
  顶层脚本 `export_onnx`，库与脚本之间的知识环随之消失。
- 未来若有人提议把 `placement` 合回 `endfield`、或把张量编码挪到采样侧，先读这条。
