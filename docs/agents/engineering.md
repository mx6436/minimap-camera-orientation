# 工程笔记

改本仓代码前读这里：跨模块契约、不变量与模块归属。人类入门看 README，领域词汇看 CONTEXT.md，决策看 `docs/adr/`。

## 管线

```
data/{train_raw,val_raw} ──locate_dataset.py(ref)──> data/locator
        │
        └──prepare_data.py[polar|ref]──> data/processed{,_ref} ──train──> runs/<name>
                                                                          │
                                            export_artifact.py ───────────┴──> bundle ──> MaaEnd
```

- `data/train_raw` / `data/val_raw` 是仅有的两个人工维护目录，脚本只读；划分由样本所在目录表达，跨侧同名硬报错。标签 `_r<角度>.png`，允许一位小数。
- `prepare_data.py` 一条命令完成 raw → 模型输入 → 划分；`data/train` / `data/val`（及 `_ref`）是 `processed*` 的符号链接视图，每次运行重建并校验悬空链接。
- `processed*` 挂 `.preprocess.json` 缓存戳（定义哈希 + 图版本 + 输入指纹）：定义变更 / 输入增删 / `--force` 触发重生成。polar 的指纹是两侧样本名并集；ref 另含消费的 `zone`/`x`/`y`/`scale`，样本在目录间移动不改变并集、不触发重算。
- ref 额外依赖定位产物 `data/locator/`（`locate_dataset.py` 增量维护，可断点续跑）与 gitignored 的本地工作台 `local/maplocator/`。入选门：`status==0 && !isHeld && locConf>=0.55`；坐标一致性过滤拒绝的样本计入 skipped。

## 前处理定义

- `endfield/preprocess.py` 是整帧 → 观测 ROI → 条带的唯一实现（CONTEXT.md「前处理定义」）：极坐标展开几何、参考采样与条带域合成、采样与取整约定都收在这里。训练数据、实机输入与交付的 `preprocess.onnx` 都经它，MaaEnd 侧只消费图。
- 交付图的输入是观测 ROI，帧 → ROI 的裁剪由 MaaEnd 侧完成；图内做窗口优先裁剪（ADR 0001）。
- `definition_hash` 是该文件内容 sha256：改动它 = 既有 `processed*` 缓存与旧 bundle 一次性判不同源，需重生成数据、重导出。定义相关的字节关键变换必须全部收在这一个文件里。
- `tests/test_definition_ownership.py` 守卫这条边界。

## 运行档案与输入模式

- `endfield/run_record.py` 单点持有 `record.json` 的 schema 与输入模式词汇：合法模式、模式 → 通道数（`polar` 3 / `ref` 7）、ref 资产根字段、旧档案默认（缺 `input_mode` 按 polar）。训练写、live / 导出 / conformance 读；不得出现第二份模式表（ADR 0003）。
- ref 资产根来自数据缓存戳（`processed_ref` 的 provenance），训练写入 `record.json`（`ref_reference_assets_root`），实机按它加载底图；换根触发数据重生成（ADR 0002）。
- `max_ref_missing`（train.toml，0~1）在读取训练集时排除环内 `ref.A<255` 占比严格大于阈值的样本，只影响训练集。

## 训练

- 入口 `uv run train --run-dir <dir>`；全部参数在 `train.toml`，每个键都有代码内默认值，未知键硬报错。

## 交付与 conformance

- bundle = `preprocess.onnx` + `polar.onnx` + `polar_with_ref.onnx` + `manifest.json`；polar 与 ref 分类器来自不同 run，调用处必填。重复导出（同 run + 同定义 + 同工具链）图与 manifest 逐字节一致。
- `manifest.json` 记 git commit / definition hash / 每图 sha256 / 指标 / fixture 清单 / 容差剖面；导出后结构自检（manifest ↔ 图 metadata ↔ 文件哈希互证、ORT 1.19.2 可加载），失败退出码 1。
- conformance 判定：ORT 1.19.2 跑图（`pyproject.toml` 固定，与 MaaEnd 运行时同版本；版本不符直接判 error、证据作废），条带 uint8 容差 ±1 LSB、pmf 1e-4。放宽阈值需证据，失败先回票定位（图 / 定义 / 环境）。
- 交付布局 `assets/resource/model/map/cameraorientation/`（三图；`manifest.json` 是训练侧交付凭据，不进 MaaEnd）。人工拷入步骤见 README「拷入 MaaEnd」。

## 实机

- `live.py` 从 run 的 `record.json` 取输入模式：polar 每帧经定义模块展开；ref 常驻 `map-locate --stream` 子进程（定位在独立线程，显示循环不阻塞），按 `(zone, x, y, scale)` 裁参考、合成 7 通道。定位不可用（失败 / held / 低分 / 资产缺失）时 overlay 显示等待态。`--snapshot` 在首个有效定位后存一张 overlay 并退出。
- 需要 gamescope 会话；ref 需要本地工作台 CLI 支持 `--stream`。

## 模块归属

| 模块 | 职责 |
| --- | --- |
| `endfield/preprocess.py` | 前处理定义（唯一实现） |
| `endfield/train/` | 训练循环与配置 |
| `endfield/model.py` | 网络与导出 wrapper |
| `endfield/run_record.py` | 运行档案与输入模式词汇 |
| `endfield/conformance.py` | 结构断言、定义哈希、参考实现 |
| `endfield/coord_filter.py` | 坐标一致性过滤（上游换算，不拟合参数） |
| `endfield/maplocator.py` | 本地工作台路径推导 |
| `export_artifact.py` / `export_onnx.py` / `export_preprocess.py` | 交付导出 |
| `verify_artifact.py` | conformance 校验入口 |
| `live.py` | 实机推理与 overlay |
| `tests/` | pytest，含定义 ownership 守卫 |
