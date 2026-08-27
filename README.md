# 小地图摄像机角度识别（RGBA Angle CNN）

本项目训练一个小型 PyTorch CNN，从《明日方舟：终末地》的小地图环形截图中预测**摄像机角度**，即训练样本文件名后缀 `_r<角度>.png` 所编码的角度。角度以度为单位，取值范围 `[0, 360)`，公开预测结果四舍五入到整度。

领域术语与核心约束（小地图世界锚定、方向指示器、视野扇形、箭头等）见 [CONTEXT.md](./CONTEXT.md)。

## 工作流

依赖由 [uv](https://docs.astral.sh/uv/) 管理。先运行一次 `uv sync` 创建 `.venv` 并安装锁定依赖，之后通过 `uv run` 执行各脚本：

```bash
uv run crop_ring.py
uv run split_dataset.py
uv run train.py --output-dir runs/experiment_001
uv run predict.py data/val/<one-val-file>.png --checkpoint runs/experiment_001/best.pt
```

`crop_ring.py` 从 `data/raw` 重建 `data/processed`。它的关键目的是**滤除箭头、保留视野扇形**：ROI 中心是角色模型朝向的箭头，与摄像机角度无关，且在真实输入中可与视野扇形方向不一致，属于误导信息；环形掩膜的内径孔洞将其剔除，只保留携带真实信号（视野扇形）与背景（地形、图标）的环形区域。输出 112x112 RGBA PNG，透明像素写为 `(0, 0, 0, 0)`。该脚本只删除 `data/processed` 中的旧 PNG，不修改原始图像。

`split_dataset.py` 是唯一操作数据集划分的脚本。它从 `data/processed` 按种子和角度分层复制：约 15% 的验证图像（按 30 度角度分箱，每箱至少一张验证图）复制到 `data/val`，其余复制到 `data/train`。划分结果记录在 `data/split_manifest.json` 并在后续运行中复用。只有刻意要重新划分时才使用 `uv run split_dataset.py --resplit`。

`train.py` 只负责训练：读取 `data/train` 和 `data/val`，从不复制、移动或划分图像。训练默认自动使用 CPU 或 CUDA，输出目录 `runs/production_001`，batch size 32，数据加载进程 0 个，CPU 线程 16 个，最大 400 个 epoch 并带早停（patience 40）与 ReduceLROnPlateau（patience 20）。可用 `--device cpu`、`--output-dir runs/experiment_name`、`--epochs N`、`--batch-size N` 或 `--workers N` 覆盖默认值。`--smoke` 标志走正常路径只跑一个 epoch，用于验证流程，不能替代完整训练。

从中断处恢复训练：

```bash
uv run train.py --resume
uv run train.py --resume runs/experiment_name/last.pt --output-dir runs/experiment_name
```

每个输出目录包含 `best.pt`、`last.pt`、`history.json` 和 `config.json`。在空目录中启动新实验；恢复已有实验用 `--resume`，而不是静默覆盖已有 checkpoint。模型有 995,952 个可训练参数，在回归头之前保留 4x4 粗粒度空间布局，接受缩放到 `[0, 1]` 的 `4x112x112` RGBA 输入，回归正弦/余弦分量并在解码时归一化。验证指标使用环形误差，因此 0/360 边界是连续的。

训练对训练图像施加 RGB 噪声（σ=0.02，50% 概率）增强；验证图像不做任何增强。**不使用旋转增强**：小地图是世界锚定的，地形永不旋转、图标始终屏幕正立，旋转后的输入在真实数据中不存在，只会引入分布外样本（见 [CONTEXT.md](./CONTEXT.md)）。

`predict.py` 接受恰好一个已存在的 112x112 RGBA PNG，不缩放、不转换颜色格式。默认读取 `runs/production_001/best.pt`；需要时传 `--checkpoint` 和 `--device`。

## 数据目录

- `data/raw`：原始截图；任何脚本都不会修改它。
- `data/processed`：重新生成的环形裁剪 RGBA 图像。
- `data/train`：复制的训练图像（磁盘上不做增强）。
- `data/val`：复制的留出验证图像。
- `data/split_manifest.json`：可复现的划分记录。
- `runs/`：checkpoint 与 JSON 实验结果。
