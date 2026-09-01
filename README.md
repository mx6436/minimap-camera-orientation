# 小地图摄像机角度识别（极坐标 Angle CNN）

本项目训练一个小型 PyTorch CNN，从《明日方舟：终末地》的小地图截图中预测**摄像机角度**，即训练样本文件名后缀 `_r<角度>.png` 所编码的角度。角度以度为单位，取值范围 `[0, 360)`。

领域术语与核心约束（小地图世界锚定、方向指示器、视野扇形、箭头等）见 [CONTEXT.md](./CONTEXT.md)。

模型输入是**极坐标展开**（选型依据见 [docs/adr/0002](./docs/adr/0002-polar-unwrap-only-input.md)）：把小地图环形区域展开为 360x44 RGB 图像——角度映射到 x 轴（1°/列，正北为第 0 列，顺时针为正），半径映射到 y 轴（内径在上）。展开输出天然全有效，位于中心圆内的箭头被排除在输入之外；0/360 接缝以循环卷积（circular padding）连通。

## 工作流

依赖由 [uv](https://docs.astral.sh/uv/) 管理。先运行一次 `uv sync` 创建 `.venv` 并安装锁定依赖，之后通过 `uv run` 执行各脚本：

```bash
uv run prepare_data.py
uv run train.py --output-dir runs/<name>
uv run predict.py data/raw/<screenshot>.png --checkpoint runs/<name>/best.pt
```

测试通过 pytest 运行：`uv run pytest`。

`prepare_data.py` 是唯一的前处理脚本，一条命令完成 raw → 极坐标展开 → 划分：

```bash
uv run prepare_data.py                    # 极坐标展开 + 清单切分
```

验证集成员由 `data/val_manifest.json` 直接指定（val = 清单 ∩ processed，清单引用不存在的文件名则报错；train = 其余全部），清单由人维护，是运行脚本的前置条件。每次运行都会清空并重写 `data/processed`、`data/train` 和 `data/val`；只有 `data/raw` 与 `data/val_manifest.json` 永不被脚本改动。角度标签支持一位小数（如 `_r210.9.png`），训练目标保留浮点精度。

`train.py` 只负责训练：读取 `data/train` 和 `data/val`，从不复制、移动或划分图像。全部训练参数集中在根目录 [`train.toml`](./train.toml)：每个键都有代码内默认值，文件明示当前基线，未知键硬报错。CLI 只保留调用管道：`--config`（默认 `train.toml`）、`--output-dir`、`--device`（auto/cpu/cuda）、`--threads`（CPU 线程，默认 16）与 `--smoke`（正常路径只跑一个 epoch，用于验证流程，不能替代完整训练）。数据加载为单进程（num\_workers=0）。

训练对训练图像以 50% 概率施加一次顺时针旋转，旋转角从 24 个非零的 15 度倍数（15°–345°）中均匀选取，目标角度按模 360 加上相同旋转量；旋转沿 1°/列的角度轴做 `np.roll`，24 个方向全部严格无损。验证图像不做任何增强。

`predict.py` 接受恰好一张原始截图 PNG（任意分辨率，按 720p 基准等比缩放 ROI 后极坐标展开），不要求预先裁剪。默认读取 `runs/production_001/best.pt`；需要时传 `--checkpoint` 和 `--device`。

## 数据目录

除 `data/raw` 与 `data/val_manifest.json` 外，以下内容均为脚本输出，每次运行 `prepare_data.py` 时清空重写：

- `data/raw`：原始截图；任何脚本都不会修改它。
- `data/val_manifest.json`：清单切分的验证集成员清单，由人维护，脚本只读。
- `data/processed`：处理输出（全部样本的极坐标展开）。
- `data/train` / `data/val`：划分后的训练/验证图像副本（磁盘上不做增强）。
- `runs/`：checkpoint 与 JSON 实验结果。
