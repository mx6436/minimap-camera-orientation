# 交付 bundle 词汇与结构自检收口到 `endfield/bundle.py`，验收剖面由 conformance 注入

交付角色词汇（角色 → 图文件名 / 输入模式）、manifest 字段 schema 与「manifest ↔ 文件
哈希 ↔ 图 metadata ↔ ORT 可加载」的结构自检收在一个 torch-free 的 `endfield/bundle.py`：
`cli/export_artifact.py` 与 `cli/verify_artifact.py` 各自持有的 role 表、图文件名表与
manifest 一致性实现随之删除。验收剖面的取值（定义哈希、ORT 版本、容差剖面、fixture
清单）由 `endfield/conformance.py` 构造为 `ManifestProfile` 注入，bundle 不 import
`conformance` 也不 import `preprocess`（后者的 torch 依赖会拖进校验路径）。图文件名是
跨仓契约（MaaEnd 交付布局），manifest 字段 schema 属本仓。通道数仍由
`endfield/run_record.py` 单点定义（ADR 0003），bundle 只持角色 → 输入模式。

## Considered Options

- **role 表与一致性检查留在导出侧，verify 侧各持一份**（原状）：同一份 manifest ↔ 工件
  契约有两套实现，角色 → 文件名/模式有四份表；改一次 bundle 形状要动四个文件。
- **验收剖面的取值一并收进 bundle**：manifest 记录的是产出方工具链的验收剖面，取值属
  验证方；收进 bundle 会让它变成第二个 conformance，并让导出侧失去剖面来源。
- **把 ONNX 算子级断言（`check_preprocess_model` / `check_classifier_model`）一并收进
  bundle**：那是图的契约而非 bundle 的契约，且 `export-onnx` 要在没有 bundle 时用它；
  与「图契约 ↔ 数值 conformance」的切分是另一件事，不在本条内。
- **让 bundle 直接 import `preprocess` / `conformance` 取定义哈希与 ORT 版本**：会拖入
  torch 与 fixture 机制，`verify-artifact` 的校验路径随之变重；故改为注入。
- **`Finding` 随 bundle 一起定义**：`export-onnx` 要对不属于任何 bundle 的图报结论，会让
  后续的图契约模块反过来依赖 bundle；故独立为 `endfield/findings.py`。

## Consequences

- manifest 字段名、交付角色表、图文件名不再有第二份；`cli/export_onnx.py` 的
  `OUTPUT_NAMES` 删除，默认交付文件名由 `bundle.graph_file(role_for_mode(mode))` 给出。
- `cli/export_artifact.check_bundle` 与 `verify_bundle` 的 manifest 一致性口径合并为
  `bundle.check_structure`；`verify_bundle` 成为「bundle 一致性 → 算子级断言 → 数值比对」
  三段。带 manifest 的 bundle 因此多出文件 sha256 与图 metadata 互证（校验更强，代价是
  三张图各多一次 ORT 加载）。
- `verify-artifact` 的草稿路径（无 manifest → 只报 warning、只要求 preprocess）保留，
  由 `check_structure(require_manifest=False)` 表达。
- `run` 产物的布局（`record.json` / `summary.json`）不进 bundle：`build_manifest` 经
  `load_run` 回调取得运行档案与训练汇总，编排层仍是唯一知道 run 目录形状的地方
  （ADR 0004）。
- 本条是 ADR 0003 的补全而非修订：0003 对 `run_record` 的结论不变（该 module 不持有交付
  bundle 词汇），本条只点名它要求收口的那个交付 module。
