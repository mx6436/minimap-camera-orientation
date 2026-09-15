# run 产物契约收口到 `endfield/run_dir.py`，交付侧直连其 `load_run`

run 目录的形状（`best.pt` / `record.json` / `summary.json` / `history.json` 的路径与占用标记）与
`summary.json` 的字段 schema 此前没有单点持有者：训练写侧、实机与三个导出命令各自拼路径，交付侧按字符串
键消费训练汇总。现收进 torch-free 的 `endfield/run_dir.py`：路径常量、`TrainingSummary`（含 typed 验证
指标）、`write_summary` / `load_run` / `occupied`。ADR 0005 曾让 `bundle.build_manifest` 经调用方注入的
`load_run` 回调取 run 事实；现改为直接 import `run_dir.load_run`——布局仍不进 bundle（知识在
`run_dir.py`），bundle 只是少了一个参数、不再需要调用方提供读取回调。`record.json` 的 schema 仍由
`endfield/run_record.py` 单点持有，`run_dir.py` 只是它的调用方。

## Considered Options

- **扩展 `endfield/run_record.py`**：它自我限定为「不持有交付 bundle 词汇与数据目录布局」，把「一个目录」
  的契约塞进「一个文件」的模块会让名字失真。否。
- **`endfield/train/run_dir.py`**：库层的 `conformance` 要读 run 目录，会变成依赖训练子包。否。
- **`cli/run_dir.py`（编排层）**：`endfield/conformance.py` 在库层拼 `best.pt`，命令包不能成为库的依赖
  （ADR 0004）。否。
- **保留 `load_run` 注入、只把返回类型 typed**：接口多一个参数与一个 `Callable`，而 ADR 0005 那条注入的
  理由（bundle 不 import torch）对 torch-free 的 `run_dir.py` 不成立。否。
- **`summary.json` 合并进 `record.json`**：档案是训练写定的复现前提，也是「run 目录是否被占用」的凭据；
  合并会让它在训练过程中被改写。否。
- **全字段严格读取 `summary.json`**：诊断指标集随 `distribution_metrics` 演化，旧 run 会因此再也导不出
  bundle。改为交付契约字段严、诊断指标宽、未知键保留。

## Consequences

- `cli/train.py` 的 `ARTIFACT_NAMES` 与 `best.pt` / `summary.json` 字面量、`cli/export_artifact.load_run`、
  `bundle.build_manifest` 的 `load_run` 参数一并删除；`build_manifest` 的签名少一个参数。
- `endfield/train/metrics.py` 的 `distribution_metrics` 返回 typed `Metrics`，`val_` 前缀的磁盘键由该模块
  的 `to_payload()` 产生；`cli/train.py` 的跟踪指标不再是裸字符串。
- `endfield/train/artifacts.py` 的 `save_checkpoint` 留在原地（import torch），只从 `run_dir.py` 取路径；
  `run_dir.py` 必须保持 torch-free，`verify-artifact` 的校验路径依赖这一点。
- `loss_curve.png` 不进路径常量与占用集合：`plot_loss_curves` 自行拼路径，它的缺失不影响 run 占用判定。
- ADR 0005 的结论「编排层仍是唯一知道 run 目录形状的地方」由本条修订为「`endfield/run_dir.py` 是唯一
  知道 run 目录形状的地方，编排层与交付侧都只是调用方」；其「布局不进 bundle」与「bundle 不 import
  torch」两条不变。
- 无磁盘格式变更：既有 run 目录与 bundle 无需重生成，`definition_hash` 不变。
