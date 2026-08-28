# 小地图摄像机角度识别（RGBA Angle CNN）

本项目训练一个小型 PyTorch CNN，从《明日方舟：终末地》的小地图环形截图中预测**摄像机角度**，即训练样本文件名后缀 `_r<角度>.png` 所编码的角度。角度以度为单位，取值范围 `[0, 360)`，公开预测结果四舍五入到整度。

领域术语与核心约束（小地图世界锚定、方向指示器、视野扇形、箭头等）见 [CONTEXT.md](./CONTEXT.md)。

## 工作流

依赖由 [uv](https://docs.astral.sh/uv/) 管理。先运行一次 `uv sync` 创建 `.venv` 并安装锁定依赖，之后通过 `uv run` 执行各脚本：

```bash
uv run prepare_data.py --input rgba
uv run train.py --output-dir runs/experiment_001
uv run predict.py data/val/<one-val-file>.png --checkpoint runs/experiment_001/best.pt
```

`prepare_data.py` 是唯一的前处理脚本，一条命令完成 raw → 处理 → 划分：

```bash
uv run prepare_data.py --input rgba               # 环形裁剪 + 随机切分
uv run prepare_data.py --input rgba --split manifest   # 环形裁剪 + 清单切分
uv run prepare_data.py --input polar              # 极坐标展开 + 随机切分
```

`--input` 必填，选择处理格式：`rgba`（环形裁剪 112x112 RGBA，关键目的是**滤除箭头、保留视野扇形**：环形掩膜的内径孔洞剔除位于中心圆内的箭头，只保留携带真实信号（视野扇形）与背景（地形、图标）的环形区域）或 `polar`（极坐标展开 360x44 RGB，展开约定见 [CONTEXT.md](./CONTEXT.md) “极坐标展开”词条）。`--split` 选择切分模式，默认 `random`：按种子和 30 度角度分箱随机留出约 15% 作验证集（每箱至少一张验证图）；`manifest` 则由 `data/val_manifest.json` 直接指定验证集成员（val = 清单 ∩ processed，清单引用不存在的文件名则报错；train = 其余全部）。每次运行都会清空并重写 `data/processed`、`data/train`、`data/val` 和 `data/split_manifest.json`；只有 `data/raw` 与 `data/val_manifest.json` 永不被脚本改动。角度标签支持一位小数（如 `_r210.9.png`），训练目标保留浮点精度。

`train.py` 只负责训练：读取 `data/train` 和 `data/val`，从不复制、移动或划分图像。训练默认自动使用 CPU 或 CUDA，输出目录 `runs/production_001`，batch size 32，数据加载为单进程（num\_workers=0），CPU 线程默认 16（可用 `--threads` 调整），最大 400 个 epoch 并带早停（patience 40）与 ReduceLROnPlateau（patience 20）。可用 `--device cpu`、`--output-dir runs/experiment_name`、`--epochs N`、`--batch-size N` 或 `--threads N` 覆盖默认值。`--smoke` 标志走正常路径只跑一个 epoch，用于验证流程，不能替代完整训练。

从中断处恢复训练：

```bash
uv run train.py --resume
uv run train.py --resume runs/experiment_name/last.pt --output-dir runs/experiment_name
```

每个输出目录包含 `best.pt`、`last.pt`、`history.json` 和 `config.json`。在空目录中启动新实验；恢复已有实验用 `--resume`，而不是静默覆盖已有 checkpoint。模型有 995,952 个可训练参数，在回归头之前保留 4x4 粗粒度空间布局，接受缩放到 `[0, 1]` 的 `4x112x112` RGBA 输入，回归正弦/余弦分量并在解码时归一化。验证指标使用环形误差，因此 0/360 边界是连续的。

训练对训练图像施加 RGB 噪声（σ=0.02，50% 概率）增强，并以 50% 概率施加一次顺时针旋转，旋转角从 24 个非零的 15 度倍数（15°–345°）中均匀选取，目标角度按模 360 加上相同旋转量；验证图像不做任何增强。90 度倍数用 `np.rot90` 无损旋转，其他角度用 PIL BICUBIC（顺时针 = `-angle`，因为 PIL 的旋转方向为逆时针）。注意：旋转后的地形与图标在真实输入中并不存在（见 [CONTEXT.md](./CONTEXT.md)），保留旋转增强是出于经验考量——它相当于 24 倍的数据乘数/正则化，在小数据集上实测收益大于分布外噪声（历史对照：带旋转 val MAE 3.14°，无旋转 6.01°）。

`predict.py` 接受恰好一个已存在的 112x112 RGBA PNG，不缩放、不转换颜色格式。默认读取 `runs/production_001/best.pt`；需要时传 `--checkpoint` 和 `--device`。

## 极坐标展开实验

第二条输入表示管线，用于验证“极坐标展开能否优于环形裁剪”（对比基准：`runs/experiment_004`，best val_circular_mae 4.556°）：

```bash
uv run prepare_data.py --input polar
uv run train.py --input polar --output-dir runs/polar_001
```

`prepare_data.py --input polar` 从 `data/raw` 生成极坐标展开样本并按相同模式划分，文件名与划分结果和 rgba 路径逐文件一致。两种格式的处理输出共用 `data/processed`，切换格式即重跑 `prepare_data.py`（已训完的 `runs/` 不受影响）。

`train.py --input polar` 切换数据加载（3 通道 360x44 RGB）、模型（`AngleCNN(in_channels=3, padding_mode="circular")`，循环 padding 使 0/360 接缝两侧连通，参数量 995,664）与增强实现：顺时针旋转变为沿角度轴 `np.roll` 15 列（24 方向全部无损，替代 bicubic）；RGB 噪声不加掩膜（全图有效）。概率与 σ 与 RGBA 管线相同。极坐标实验的 config 为 version 11，含 `input_representation` 与 `conv_padding_mode` 字段；RGBA 路径的 config（version 10）保持不变，旧 checkpoint 可继续 `--resume`。

## 数据目录

除 `data/raw` 与 `data/val_manifest.json` 外，以下内容均为脚本输出，每次运行 `prepare_data.py` 时清空重写（见 [docs/adr/0001](./docs/adr/0001-regenerable-data-layout.md)）：

- `data/raw`：原始截图；任何脚本都不会修改它。
- `data/val_manifest.json`：清单切分的验证集成员清单，由人维护，脚本只读。
- `data/processed`：处理输出（当前 `--input` 格式的全部样本）。
- `data/train` / `data/val`：划分后的训练/验证图像副本（磁盘上不做增强）。
- `data/split_manifest.json`：最近一次划分的记录（格式、模式、策略参数、文件列表）。
- `runs/`：checkpoint 与 JSON 实验结果。
