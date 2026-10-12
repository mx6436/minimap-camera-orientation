# 小地图摄像机角度识别

从《明日方舟：终末地》的实时小地图预测摄像机角度（`[0, 360)`，顺时针为正）。

## 快速开始

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
uv run pytest
```

### 数据

单张截图是训练/验证样本，文件名以 `_r<角度>.png` 结尾标注摄像机角度，支持一位小数（如 `_r210.9.png`）。截图按训练侧与验证侧分开准备：

- `data/train_raw`：训练侧原始截图。
- `data/hard_raw`：训练侧的**困难样本**，训练损失里按 `train.toml` 的 `hard_weight`（默认 5）计权，等价于把它们复制成 N 份。
- `data/val_raw`：训练侧原始截图。

三处数据不应有重合。

### polar

```bash
uv run prepare-data                     # raw -> data/processed，并建 data/train、data/val 视图
uv run train --run-dir runs/<name>      # 训练参数见 train.toml
```

### ref

先批量定位（可断点续跑），再生成观测与参考两路：

```bash
uv run locate-dataset                   # -> data/locator
uv run prepare-data --mode ref          # -> data/processed_ref，并建 data/train_ref、data/val_ref
uv run train --run-dir runs/<name>      # train.toml 里 input_mode = "ref"
```

定位依赖本机的 MapLocator 工作台 `local/maplocator/`。

## 交付

```bash
uv run export-artifact --out runs/<name>/bundle --ref-run runs/<ref_run>
```
