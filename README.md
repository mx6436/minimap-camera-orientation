# 小地图摄像机角度识别

本项目训练一个小型 PyTorch 模型，从《明日方舟：终末地》的小地图截图中预测**摄像机角度**，即训练样本文件名后缀 `_r<角度>.png` 所编码的角度。角度以度为单位，取值范围 `[0, 360)`。

领域术语与核心约束（小地图世界锚定、方向指示器、视野扇形、箭头等）见 [CONTEXT.md](./CONTEXT.md)。

模型输入是**极坐标展开**：把小地图环形区域展开为 360x42 BGR 图像——角度映射到 x 轴（1°/列，正北为第 0 列，顺时针为正），半径映射到 y 轴（内径在上）。位于中心圆内的箭头被排除在输入之外；0/360 接缝以循环卷积（circular padding）连通。

**参考配对**输入（模式名 `ref`）在同一展开几何上增加一路参考：用 MapLocator 定位到的 zone 底图在 `(x, y)` 处裁出与观测同视野的 118x120 参考图（按 zone 尺度比，见下），观测与参考各自极坐标展开后并列拼接为 7 通道 `[obs.BGR, ref.BGR, ref.A]`（不预先相减）：`ref.BGR` 是参考裁剪的合成 BGR（观测背底合成，参考缺失处逐像素 copy 观测），`ref.A` 是底图资产的原始 alpha（0 = 参考缺失）。两种模式的数据、训练与实机推理路径都可用，模式由 `train.toml` / run `record.json` 选择。

## 工作流

依赖由 [uv](https://docs.astral.sh/uv/) 管理。先运行一次 `uv sync` 创建 `.venv` 并安装锁定依赖，之后通过 `uv run` 执行各脚本：

```bash
uv run locate_dataset.py                  # ref 前置：对 data/raw 批量定位（可断点续跑）
uv run prepare_data.py                    # polar：极坐标展开 + 清单切分
uv run prepare_data.py --mode ref         # ref：观测/参考各自展开 + 清单切分
uv run train --run-dir runs/<name>        # 输入由 train.toml 的 input_mode 选择
uv run predict.py data/raw/<screenshot>.png --run-dir runs/<name>
```

测试通过 pytest 运行：`uv run pytest`。

`prepare_data.py` 是唯一的前处理脚本，一条命令完成 raw → 模型输入 → 划分（polar 或 ref）。验证集成员由 `data/val_manifest.json` 直接指定（val = 清单 ∩ processed，清单引用不存在的文件名则报错；train = 其余全部），清单由人维护，是运行脚本的前置条件。角度标签支持一位小数（如 `_r210.9.png`），训练目标保留浮点精度。

每种模式各自清空并重写自己的 processed / train / val 目录；只有 `data/raw` 与 `data/val_manifest.json` 永不被脚本改动。ref 模式的定位产物单独维护在 `data/locator/`（不随 `prepare_data.py` 清空）。

训练入口是控制台命令 `uv run train`，只负责训练：读取训练/验证目录，从不复制、移动或划分图像。全部训练参数集中在根目录 [`train.toml`](./train.toml)：每个键都有代码内默认值，文件明示当前基线，未知键硬报错。CLI 只保留调用管道：`--config`（默认 `train.toml`）、`--run-dir`（必填，run 产物目录）、`--device`（auto/cpu/cuda）、`--threads`（CPU 线程，默认 8）与 `--smoke`（正常路径只跑一个 epoch，用于验证流程，不能替代完整训练）。

`train.toml` 的 `input_mode` 选择训练数据：`"polar"`（默认）读 `data/train`、`data/val`；`"ref"` 读 `data/train_ref`、`data/val_ref`。`map_assets_root` 指向 ref 使用的 MapLocator 底图资产目录（默认本地 MaaEnd 工作副本，需与 `prepare_data.py --mode ref` 一致），写在 run 的 `record.json`（`ref_reference_assets_root`），供实机推理读取。ref 模式还可选 `max_ref_missing`（0~1）：训练集在读取时排除环内 `ref.A<255` 占比**严格大于**阈值的样本（等于阈值保留），val 不变；阈值一并写入 `record.json`。

`predict.py` 接受恰好一张原始截图 PNG（任意分辨率，按 720p 基准等比缩放 ROI 后极坐标展开），不要求预先裁剪。`--run-dir` 必填，模型读取其中的 `best.pt`；设备可用 `--device` 指定。**predict 只实现 polar 输入**，ref 的单图推理路径尚未落地（实机路径见下节的 `live.py`）。

`export_onnx.py` 把 run 的 `best.pt` 导出为 ONNX 交付格式（默认 `<run-dir>/cameraorientation.onnx`），输入布局与训练契约一致：polar 为 `[1,42,360,3]` 观测条带，ref 为 `[1,42,360,7]` `[obs.BGR, ref.BGR, ref.A]` 参考配对条带；模式从 run 的 `record.json` 读取（旧 pair v2 record 映射为 ref）。`/255` 折入首层卷积，图内只留 HWC→CHW 转置与 softmax，输出 `[1,360]` 概率质量函数。

```bash
uv run export_onnx.py --run-dir runs/<name>                        # 输出 runs/<name>/cameraorientation.onnx
uv run export_onnx.py --run-dir runs/<name> --output /tmp/model.onnx
```

`live.py` 对运行中的游戏做实时推理：从 `--run-dir` 的 `record.json` 读取 `input_mode`（旧 record 无此字段时按 polar 兼容；ref 定名之前的 pair v2 record 映射为 ref），polar 每帧直接极坐标展开；ref 起 `map-locate --stream` 常驻子进程做流式定位（定位在独立线程，显示循环不阻塞），按定位 `(zone, x, y)` 裁参考底图、合成参考后拼 `[obs.BGR, ref.BGR, ref.A]` 7 通道张量，再喂模型；定位不可用（失败 / held / 低分 / 资产缺失）时 overlay 显示等待态。overlay 展示圆盘、当前模型输入（极坐标展开 / ref 的 obs 与 ref 两路）与 360 bin 概率曲线；`--snapshot <path>` 在拿到首个有效定位后保存一张 overlay 并退出（实机 smoke 取证用）。ref 实机推理依赖 gitignored 的 `local/maplocator/`（含 `--stream` 的迭代二进制，见其 `README.local.md`）。

## 工件校验（conformance）

`verify_artifact.py` 对交付 bundle 做一致性校验：ORT 1.19.2 跑图，逐 fixture 与参考实现（定义模块；定义模块落地前为现行 cv2 前处理）比对并应用容差，输出通过/失败与差异明细。判定口径见 map 的「模型级等价」：不承诺与 C++ 逐位一致，uint8 条带按 ±1 LSB 预期。

环境固定为 Python 3.12 + dev 依赖 `onnxruntime==1.19.2`（与 MaaEnd 运行时同版本，`pyproject.toml` 固定）；运行时版本不一致直接判 error（证据作废）。

### 用法

```bash
uv run verify_artifact.py --bundle runs/<name>/bundle          # 结构 + 全量内置 fixtures
uv run verify_artifact.py --bundle <dir> --run-dir runs/<name> # 追加分类器 ↔ checkpoint 数值比对
uv run verify_artifact.py --bundle <dir> --fixture-dir <dir>   # 用本地真实样本 fixtures
uv run verify_artifact.py --bundle <dir> --report report.json  # 落 JSON 证据
uv run verify_artifact.py --dump-fixtures conformance/fixtures # 物化内置 fixtures
```

- 退出码：0 = 通过（允许 warning），1 = 存在 error 或超容差比对。
- `--require` 缺省由 manifest 的 `graphs` 决定；无 manifest 的草稿 bundle 只要求 `preprocess`。
- `--run-dir` 提供后，分类器图与 checkpoint 逐 fixture 比对 pmf（torch 侧 `ExportWrapper`）；manifest 的 `graphs.<role>.run_dir` 优先于 CLI。

### bundle 与 manifest

bundle 目录（对应 MaaEnd 交付布局）含 `preprocess.onnx`、`polar.onnx`、`polar_with_ref.onnx` 与 `manifest.json`（由 `export_artifact.py` 产出）：

```json
{
  "schema_version": 1,
  "git_commit": "<训练/导出时的 commit>",
  "definition_hash": "<endfield.conformance.definition_hash() 的 sha256>",
  "ort_version": "1.19.2",
  "graphs": {
    "preprocess": {"file": "preprocess.onnx",
                   "outputs": {"observed": "obs", "reference": "ref"}},
    "polar": {"file": "polar.onnx", "run_dir": "runs/<polar_run>"},
    "polar_with_ref": {"file": "polar_with_ref.onnx", "run_dir": "runs/<ref_run>"}
  },
  "fixtures": ["polar_basic", "..."],
  "tolerances": {"strips_uint8": 1, "pmf_float32": 1e-4, "gap_fraction": 0.01}
}
```

- `graphs` 列出的图必须存在，缺一即 error；它也是缺省 `--require` 集。
- `outputs` 声明输出角色；缺省按图输出顺序（第一 = observed，第二 = reference）。
- `definition_hash` 与当前定义源码不一致 = bundle 与定义不同源，error（需重导出）。
- `fixtures` 缺省为内置 6 个场景；`tolerances` 覆盖默认剖面。

### fixtures

fixture 是「输入场景」，期望输出在比对时由参考实现实时计算，不落 golden。两种来源：内置场景（`endfield/conformance.py`）与 `--fixture-dir` 目录下的 `*.npz`（`--dump-fixtures` 的产物或本地真实样本，字段同名）。

| 字段 | 含义 |
| --- | --- |
| `name` / `description` / `tags` | 标识与覆盖类别 |
| `minimap` | 118x120 BGR uint8 观测 ROI |
| `asset` | HxWx3/4 uint8 底图；3 通道视为完全不透明 |
| `x` / `y` / `scale` | MapLocator 定位坐标与 `ZoneTemplateScale` |

内置 6 个场景覆盖：极坐标（`polar_basic`）、参考配对（`ref_pair_basic`）、裁剪越界（`crop_out_of_bounds`）、非 1:1 zone（`zone_non_1to1`，scale=15/16）、参考缺失（`ref_missing_alpha0`）、资产 3 通道（`asset_rgb_3ch`）。

### 结构与数值断言

- `preprocess.onnx`：存在 `GridSample`，且 opset 18 下 `mode="bilinear"` / `padding_mode="border"` / `align_corners=0`；`X`/`grid` 为 float32（uint8 需图内 Cast）；`asset` 的 H/W 为动态维；输出条带 uint8、形状 42x360x3/4；无 contrib 域节点。
- `polar.onnx` / `polar_with_ref.onnx`：输入 uint8 42x360x3/7，输出 float32 [1,360]，无 contrib 域节点。
- 数值：每个输出报 `max_abs` / `mean_abs` / `p99_abs` / `diff_fraction`，**通过条件 = `max_abs <= limits[profile]`**（形状或 dtype 不符直接失败）。缺口占比（`ref.A < 255`）另报绝对误差；分类器与 checkpoint 另报 `angle_error_deg`（argmax 环差，仅供证据，不作门限）。

默认容差剖面：

| 剖面 | 适用 | 缺省上限 | 依据 |
| --- | --- | --- | --- |
| `strips_uint8` | 条带输出 | 1 | ±1 LSB 口径（#22 研究） |
| `pmf_float32` | 分类器 pmf | 1e-4 | float32 导出等价 |
| `gap_fraction` | 缺口占比 | 0.01 | 1 个百分点 |

阈值只在 manifest 的 `tolerances` 覆盖；覆盖要有证据（#23 对拍分布），失败先回票定位（图/定义/环境），不得为通过而放宽。

### 定义模块交接

`verify_artifact.py` 的参考侧当前适配现行 cv2 前处理（`endfield/polar.py` + `endfield/ref.py`），`definition_hash()` 也取这两份源码。定义模块（#25）落地后，把 `endfield/conformance.py` 的 `reference_strips()` / `definition_hash()` 指过去（唯一实现）、删除 cv2 路径；校验侧接口与 fixtures 不变。

## 数据定位（MapLocator 批量）

`locate_dataset.py` 对 `data/raw` 全量截图逐张运行 MapLocator，产出 ref 前处理所需的定位产物与汇总：

```bash
uv run locate_dataset.py                 # 默认 4 个并行进程；已成功样本跳过，可断点续跑
```

- `data/locator/locate.jsonl`：每行一条定位记录（schema 见下表）。
- `data/locator/summary.json`：成败计数、失败分类、调用次数分布、locConf 分布、按命名族成功率。

定位 CLI 与资源放在 gitignored 的 `local/maplocator/`（本机工作台落点，来源与重建见该目录的 `README.local.md`）；仓库内脚本只引用该目录，不引用仓库外路径。

`locate.jsonl` 字段：

| 字段 | 含义 |
| --- | --- |
| `name` | 原始截图文件名（样本标识） |
| `status` | MapLocator 状态：`0` Success / `1` TrackingLost / `2` ScreenBlocked / `3` Teleported / `4` YoloFailed / `5` NotInitialized；CLI 级失败为 `-1` 读图失败、`-2` 小地图 ROI 越界 |
| `message` | 状态原文；失败分类见 `summary.json` 的 `excluded_by_reason` |
| `zone` | 定位到的 MapLocator zone（如 `Wuling_Base`、`ValleyIV_L6_109`；tier zone 的 x/y 为切片坐标） |
| `x`, `y` | zone 图上的像素坐标 |
| `rot` | MapLocator 输出的箭头朝向，**不是**摄像机角度 |
| `locConf` | 匹配分数（原始值，未加工） |
| `isHeld` | 全局搜索没有过线峰、放行裸峰的标记 |
| `latencyMs` / `elapsedMs` | 单次 locate 内部耗时 / 单图端到端耗时 |
| `attempts` | 该图实际 locate 调用次数（1 或 3，冷启动共识） |
| `accepted` / `accept_reason` | 入选门：`status==0` 且 `!isHeld` 且 `locConf >= 0.55`；否则为 `held` / `below_loc_threshold` / 失败类别 |

重复运行幂等：已成功样本跳过，失败项重跑覆盖；held 与低分记录保留在产物中但 `accepted=false`。

### 参考配对前处理（ref）

`uv run prepare_data.py --mode ref` 消费上节的定位产物，把观测与参考各自展开后落盘为两路：

- **姿态来源**：`locate.jsonl` 中 `accepted=true` 的记录；`(zone, x, y)` 一律取自 MapLocator 输出，不从文件名解析（文件名只提供样本标识与角度标签 `r`）。
- **样本范围**：定位失败 / held / 低分（`accepted=false`）与 zone 资产缺失的样本跳过并计数，不算错误。
- **参考底图**：按 `zone` 反解资产路径（`{P}_Base → {P}/Base.png`、`{P}_L{n}_{m} → {P}/Lv{int(n):03d}Tier{m}.png`、其它 → 任意子目录下 stem 同名文件）；tier zone 的 `(x,y)` 就是切片自身像素空间（实测与观测小地图 1:1，直接裁切片，无需仿射）。
- **参考裁剪**：与观测同一视野（`endfield/polar.py` 的 `ROI_CENTER`，尺寸 118x120，中心 `(x,y)`），按 zone 的尺度比缩放：绝大多数 zone 是 1:1 直接裁；`ValleyIV_Base` 的底图相对观测缩放过 6.7%，按 MapLocator 的 `ZoneTemplateScale`（15/16）裁 `ROI*15/16` 再缩回 118x120。越界处外侧填 0（黑），不失败。
- **观测流**：原始截图按 `polar.unwrap` 展开，输出 42x360x3 BGR，与 polar 模式的 `data/processed` 同源同几何。
- **参考流**：`ref.A` 为资产原始连续 alpha（不二值化、不设阈值），裁剪越界与资产 `alpha<255` 统一为「参考缺失」，`ref.A = 0`。`ref.BGR` 为观测背底合成 `black_ref + obs_roi*(1 - alpha/255)`（在 118x120 ROI 域、`unwrap` 之前，四舍五入回 uint8，>255 饱和）：alpha==0 处逐像素等于观测（缺失处 copy 观测）、alpha==255 处等于黑底合成 `rgb*alpha/255`。
- **产物布局**（两路分别落盘）：`data/processed_ref/<name>.png` 为观测流（42x360x3 BGR），`data/processed_ref/ref/<name>.png` 为参考流（42x360x4 BGRA，B/G/R = 参考 BGR，A = 原始 alpha）；`data/train_ref`、`data/val_ref` 是同一布局的符号链接视图，`ref/` 子树一并链接。
- **模型输入**：两路按通道拼接为 42x360x7；训练侧由 `train.toml` 的 `input_mode = "ref"` 选择数据根；`record.json` 记 `ref_reference_assets_root`；`live.py` 的 ref 推理路径与之共用 `endfield/ref.py` 的编码，与 `prepare_data.py` 的产物逐字节一致。
- **确定性**：重复运行产物逐字节一致。
- **训练样本过滤（可选）**：`train.toml` 的 `max_ref_missing`（0~1）在读取训练集时排除环内 `ref.A<255` 占比**严格大于**阈值的样本（等于阈值保留），只影响训练集，val 不变；`prepare_data.py` 始终全量落盘，过滤不改磁盘数据。

## 数据目录

除 `data/raw` 与 `data/val_manifest.json` 外，以下内容均为脚本输出，每次运行 `prepare_data.py` 时清空重写：

- `data/raw`：原始截图；任何脚本都不会修改它。
- `data/val_manifest.json`：清单切分的验证集成员清单，由人维护，脚本只读。
- `data/processed`：polar 处理输出（全部样本的极坐标展开）。
- `data/train` / `data/val`：polar 划分后的训练/验证图像副本（磁盘上不做增强）。
- `data/processed_ref`：ref 处理输出（`accepted=true` 样本的观测流与 `ref/` 参考流）。
- `data/train_ref` / `data/val_ref`：ref 划分后的训练/验证图像副本（含 `ref/` 子树）。
- `runs/`：checkpoint 与 JSON 实验结果。
- `data/locator/`：MapLocator 定位产物（`locate.jsonl`、`summary.json`），由 `locate_dataset.py` 增量维护（不随 `prepare_data.py` 清空）。
