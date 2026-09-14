"""底图定位：MapLocator 定位记录与本地工作台资产的消费面。

产出参考条带对（`sample.ReferenceSampler`）；条带域之上的张量编码在 `endfield` 侧。
依赖方向单向：本包依赖 `endfield`（前处理定义与帧解码），反向不得依赖（ADR 0004）。
"""
