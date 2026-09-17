# 工程笔记

改本仓代码前读这里：跨模块契约、不变量与模块归属。人类入门看 README，领域词汇看 CONTEXT.md，决策看 `docs/adr/`。

## 管线

```
data/{train_raw,hard_raw,val_raw} ──locate-dataset(ref)──> data/locator
        │
        └──prepare-data[polar|ref]──> data/processed{,_ref} ──train──> runs/<name>
                                                                       │
                                                      export-artifact ─┴──> bundle ──> MaaEnd
```

- `data/train_raw` / `data/hard_raw` / `data/val_raw` 是仅有的三个人工维护目录，脚本只读；划分由「训练侧 / 验证侧」表达（`train_raw` 与 `hard_raw` 同为训练侧，样本落在哪个训练目录不改变并集），跨目录同名硬报错。标签 `_r<角度>.png`，允许一位小数。
- `prepare-data` 一条命令完成 raw → 模型输入 → 划分；`data/train` / `data/val`（及 `_ref`）是 `processed*` 的符号链接视图，每次运行重建并校验悬空链接。
- `processed*` 挂 `.preprocess.json` 缓存戳（定义哈希 + 图版本 + 输入指纹）：定义变更 / 输入增删 / `--force` 触发重生成。polar 的指纹是两侧样本名并集；ref 另含消费的 `zone`/`x`/`y`/`scale`，样本在目录间移动不改变并集、不触发重算。
- 数据准备的实现（生成 + 有效划分 + 划分视图链接）收在 `endfield/prepare.py`，输入侧的模式差异由调用方注入「输入侧值 + 渲染回调」两个适配器：polar 的在 `endfield/prepare.py`，ref 的输入解析与渲染在 `placement/ref_inputs.py`；`cli/prepare_data.py` 只做接线与打印（ADR 0006）。产物布局（`DatasetLayout`）与两侧名单求交的纯函数在 `endfield/dataset.py`，训练读取共用同一份目录口径。
- ref 额外依赖定位产物 `data/locator/`（`locate-dataset` 增量维护，可断点续跑）与 gitignored 的本地工作台 `local/maplocator/`。入选门：`status==0 && !isHeld && locConf>=0.55`；坐标一致性过滤拒绝的样本计入 skipped。

## 前处理定义

- `endfield/preprocess.py` 是整帧 → 观测 ROI → 条带的唯一实现（CONTEXT.md「前处理定义」）：极坐标展开几何、参考采样与条带域合成、采样与取整约定都收在这里。训练数据、实机输入与交付的 `preprocess.onnx` 都经它，MaaEnd 侧只消费图。
- 交付图的输入是观测 ROI，帧 → ROI 的裁剪由 MaaEnd 侧完成；图内做窗口优先裁剪（ADR 0001）。
- `definition_hash` 是该文件内容 sha256：改动它 = 既有 `processed*` 缓存与旧 bundle 一次性判不同源，需重生成数据、重导出。定义相关的字节关键变换必须全部收在这一个文件里。
- `endfield/polar.py` 不转发几何常量：需要 `IMG_H`/`IMG_W` 等常量的调用方直接从前处理定义取，`polar.py` 只持帧解码、基准缩放与展示几何。
- `tests/test_definition_ownership.py` 守卫这条边界（扫描面含各顶层包，并写死定义文件路径）。

## 底图定位

- `placement/placement.py` 的 `Placement` 是定位记录里被消费的那部分事实（`zone`/`x`/`y`/`scale`，CONTEXT.md「底图定位」）的唯一构造点：缺项与类型不符硬报错；定位失败的记录照常构造（zone 为空），拿不到资产在 `asset_path()` 返回 None。
- `placement/sample.py` 的 `ReferenceSampler.strips(observed_roi, placement)` 是唯一采样入口：底图按资产路径只读一次、只转一次 float32，训练的「每 zone 复用」与实机的「每帧复用」是同一个实现。采样语义仍在定义模块。
- 条带域之上（7 通道 `[obs.BGR, ref.BGR, ref.A]` 与参考缺失占比）在 `endfield/input_encoding.py`，训练读取与实机共用（ADR 0004）。

## 运行档案与输入模式

- `endfield/run_record.py` 单点持有 `record.json` 的 schema 与输入模式词汇：合法模式、模式 → 通道数（`polar` 3 / `ref` 7）、ref 资产根字段、旧档案默认（缺 `input_mode` 按 polar）。训练写、live / 导出 / conformance 读；不得出现第二份模式表（ADR 0003）。
- run 目录的形状（`best.pt` / `record.json` / `summary.json` / `history.json` 的路径与占用标记）与 `summary.json` 的字段 schema 收在 torch-free 的 `endfield/run_dir.py`：训练写、交付读、实机取 checkpoint 都经它，`bundle.build_manifest` 直连 `run_dir.load_run`（ADR 0007 修订 ADR 0005 的注入结论）。交付契约字段（`epoch` / `val_count` / rms 误差 / 期望绝对误差）缺即报错，诊断指标缺失容忍，未知键保留；`record.json` 的 schema 仍在 `run_record.py`。
- ref 资产根来自数据缓存戳（`processed_ref` 的 provenance），训练写入 `record.json`（`ref_reference_assets_root`），实机按它加载底图；换根触发数据重生成（ADR 0002）。
- `max_ref_missing`（train.toml，0~1）在读取训练集时排除环内 `ref.A<255` 占比严格大于阈值的样本，只影响训练集。

## 训练

- 入口 `uv run train --run-dir <dir>`；全部参数在 `train.toml`，每个键都有代码内默认值，未知键硬报错。
- `hard_weight`（`train.toml`，代码内默认 5.0）是困难样本的全局权重：`data/hard_raw` 的样本在训练损失里按 `Σw·KL / Σw` 计（`w = hard_weight`，其余 1），语义等价于把该样本复制成 N 份。名单与训练划分求交后才生效（定位门与 `max_ref_missing` 的剔除同样适用，全部落空时打印显式警示）；权重只进训练损失，val 损失与全部 val 指标无权。档案记 `hard_weight` / `hard_count` / `hard_names_sha256`（ADR 0009）。

## 交付与 conformance

- `endfield/bundle.py` 持有交付 bundle 词汇与契约（ADR 0005）：交付角色（`preprocess` / `polar` / `polar_with_ref`）→ 图文件名与输入模式、本仓当前交付集合（`DELIVERED_ROLES` = `preprocess` + `polar_with_ref`；`polar` 退出交付，ADR 0008）、`manifest.json` 字段 schema、`build_manifest`（run 事实经 `run_dir.load_run`，ADR 0007）与 `check_structure`（manifest ↔ 文件哈希 ↔ 图 metadata ↔ ORT 可加载）。通道数仍由 `endfield/run_record.py` 单点定义；图文件名是跨仓契约，manifest 字段 schema 属本仓。
- `endfield/conformance.py` 持有验收剖面的取值（`profile()`：定义哈希、ORT pin、容差剖面、fixture 清单）并注入 `check_structure`；它另做算子级图断言与数值比对。导出侧与校验侧不再各持一份 manifest 一致性实现。
- bundle = `preprocess.onnx` + `polar_with_ref.onnx` + `manifest.json`；交付集合外的 run 在 `build_manifest` 即被拒（图不落盘），`check_structure` 按当前集合要求 manifest 声明齐全。重复导出（同 run + 同定义 + 同工具链）图与 manifest 逐字节一致。
- `verify_bundle` 分三段：bundle 一致性 → 算子级图断言 → 逐 fixture 数值比对；无 manifest 的草稿 bundle 只报 warning 且只要求 `preprocess`（`check_structure(require_manifest=False)`）。
- 导出侧结构自检失败退出码 1（图与 manifest 仍落盘）。conformance 判定：ORT 1.19.2 跑图（`pyproject.toml` 固定，与 MaaEnd 运行时同版本；版本不符直接判 error、证据作废），条带 uint8 容差 ±1 LSB、pmf 1e-4。放宽阈值需证据，失败先回票定位（图 / 定义 / 环境）。
- 交付布局 `assets/resource/model/map/cameraorientation/`（两图；`manifest.json` 是训练侧交付凭据，不进 MaaEnd）。人工拷入步骤见 README「拷入 MaaEnd」。

## 实机

- `live` 从 run 的 `record.json` 取输入模式：polar 每帧经定义模块展开；ref 常驻 `map-locate --stream` 子进程（定位在独立线程，显示循环不阻塞），由 `ReferenceSampler.strips` 出条带对、`assemble_ref_pair` 拼 7 通道。定位不可用（失败 / held / 低分 / 资产缺失）时 overlay 显示等待态。`--snapshot` 在首个有效定位后存一张 overlay 并退出。
- 需要 gamescope 会话；ref 需要本地工作台 CLI 支持 `--stream`。

## 模块归属

库侧两个顶层包：`endfield/`（模型与训练侧）与 `placement/`（底图定位，即对 MapLocator 定位记录与本地工作台资产的消费面）。依赖方向单向 `placement → endfield`（ADR 0004）；`cli/` 是第三个顶层包，只放命令入口。

| 模块 | 职责 |
| --- | --- |
| `endfield/preprocess.py` | 前处理定义（唯一实现） |
| `endfield/polar.py` | 帧解码、基准缩放与展示几何 |
| `endfield/input_encoding.py` | 张量编码：7 通道配对与参考缺失占比 |
| `endfield/dataset.py` | 数据集布局（`DatasetLayout`、两侧名单求交）与 `data/` 树 |
| `endfield/prepare.py` | 数据准备：渲染回调缝、产物与缓存戳、有效划分与视图链接 |
| `endfield/train/` | 训练循环与配置 |
| `endfield/model.py` | 网络与导出 wrapper |
| `endfield/run_record.py` | 运行档案与输入模式词汇 |
| `endfield/run_dir.py` | run 产物契约：路径、占用标记、训练汇总 schema 与读写 |
| `endfield/bundle.py` | 交付 bundle：角色词汇、manifest schema 与结构自检 |
| `endfield/findings.py` | 校验结论（`Finding`）：bundle 自检与 conformance 共用 |
| `endfield/conformance.py` | 结构断言、验收剖面、定义哈希、参考实现与数值比对 |
| `placement/records.py` | `locate.jsonl` 读写与集合操作 |
| `placement/placement.py` | 底图定位（`Placement`）、入选门与失败分类 |
| `placement/sample.py` | `ReferenceSampler`：底图定位 → 条带对（底图复用） |
| `placement/ref_inputs.py` | ref 输入侧：入选门、资产存在性、坐标过滤与指纹条目 → 输入侧值 |
| `placement/locator.py` | `map-locate` 进程驱动（批量一轮 / `--stream`） |
| `placement/workspace.py` | 本地工作台路径推导与 provenance |
| `placement/coord_filter.py` | 标注坐标一致性过滤（zone 判据与阈值判据） |
| `cli/` | 编排面：`prepare-data`、`locate-dataset`、`train`、`live`、`export-{onnx,preprocess,artifact}`、`verify-artifact` |
| `tests/` | pytest，含定义 ownership 守卫 |

`endfield/` 与 `placement/` 是纯库：import 它们不会带出 argparse 或子进程副作用；跨包的编排一律在 `cli/`。`endfield/model.py` 持有交付图的外层 wrapper 与权重折叠，conformance 因此不再反过来 import 顶层脚本。`endfield/prepare.py` 不 import `placement`：ref 的输入侧解析与渲染经注入的适配器接线，保持 `placement → endfield` 单向（ADR 0006）。
