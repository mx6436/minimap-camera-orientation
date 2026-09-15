# 数据准备按所有权拆分：模式无关核在 `endfield/prepare.py`，ref 输入解析在 `placement/ref_inputs.py`

数据准备（CONTEXT.md 词条）从 `cli/prepare_data.py` 拆开，按所有权落三处：模式无关的核
（`PrepareInputs` 输入侧值、`StripPair` 条带对、`PrepareReport`、生成与落盘、缓存戳、
有效划分的计算与划分视图链接）进 `endfield/prepare.py`；ref 的输入解析（`accept` 门、
资产存在性、坐标一致性过滤的编排、落戳指纹条目）进 `placement/ref_inputs.py`；
`cli/prepare_data.py` 只留 argparse、接线与报告打印。缝划在 render 回调上：`prepare()`
只认「样本名 + 观测 ROI → 条带对」，不知道底图定位，也不 import `placement`；`mode`
留在 interface 上承载「`ref` ⟺ 参考条带非 None」这条不变量。`dataset.py` 收
`raw_samples` / `directory_split` 与 `DatasetLayout`（数据准备的产物布局与训练读取
共用同一个值）。

这样切的理由是依赖方向：ref 管线要读定位记录、过入选门、查资产、跑坐标一致性过滤、
用参考采样，全是 `placement` 的知识面；把它们搬进 `endfield/` 即成包级环
`endfield → placement → endfield`，正是 ADR 0004 拒过的同型成环。

## Considered Options

- **整体搬进 `endfield/prepare.py`**（架构评审初稿的写法）：ref 输入解析要消费定位记录、
  资产与采样，搬进去即与 `placement → endfield` 成环，作废 ADR 0004 的包级方向。否。
- **`cli/` 内部拆分**（`cli/prepare_data.py` 只剩 argparse，管线进 `cli/dataprep.py`）：
  改动最小、方向不变，但库语义继续住在命令包里，函数级测试仍只能 `import cli.*`；
  把「文件凌乱」换成「文件名好看一点的凌乱」。否。
- **新顶层包 `dataprep/`**：与 `placement/` 平级（定位 / 准备 / 训练三段各一包），DAG
  干净；代价是改写 ADR 0004「跨包编排一律在 `cli/`」的结论，且同一理由会依次催生
  live overlay、导出实现的下一个包。留待布局语法整体重估时再议。
- **把 placement 侧的适配器作为参数注入 `endfield/prepare.py`**：方向保持，但 `polar`
  侧要为不存在的东西传参，interface 变大而深度不变。否。
- **两个模式各持一份生成循环**：重复的落盘与缓存逻辑会立刻长回来。否。

## Consequences

- `endfield/preprocess.py` 位置与字节都不变：`definition_hash` 不变，既有 `processed*`
  缓存与旧 bundle 仍同源，搬迁无需重生成数据、无需重导出。
- `cli/` 的语义面消失：`tests/test_prepare_data.py`（改名 `tests/test_prepare.py`）与
  `tests/test_ref.py` 不再 import `cli.prepare_data`；`tests/test_definition_ownership.py`
  的 `SCANNED_PACKAGES` 不变（不新增顶层包）。
- 有效划分的计算（两侧原始名单求交、`train + val == processed` 校验、链接）从
  `run_polar` / `run_ref` 两份收进 `prepare()` 一处；`report.train` / `report.val` 是
  求交后的结果。
- stdout 不构成契约（人读、无解析方）；命令行为逐字节不变：产物 PNG、符号链接、
  `.preprocess.json` 戳、退出码与报错口径。
- `docs/agents/engineering.md` 的「管线」bullets 与「模块归属」表要同步；CONTEXT.md 已
  新增「数据准备」词条。
- 未来若有人提议把 `placement/ref_inputs.py` 合回 `endfield/`、或在 `cli/` 里重建管线，
  先读这条。
