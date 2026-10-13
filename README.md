# 小地图摄像机角度识别

从《明日方舟：终末地》的实时小地图预测摄像机角度（`[0, 360)`，顺时针为正），供 MaaEnd 等自动化工具对齐视角。

流程：构建 C++ 数据准备工具 → 生成数据集 → 训练 → 导出 ONNX。

## 构建数据准备工具

依赖 MaaEnd submodule 与 MaaDeps：

```bash
git submodule update --init --recursive
python3 cpp/MaaEnd/tools/maadeps-download.py

cmake -S cpp --preset linux-x64
cmake --build cpp/build
```

## 数据约定

输入数据是 1280x720 的原始截图，放在 `data/{train_raw,hard_raw,val_raw}` 目录下。

## 准备数据集

在仓库根目录执行：

```bash
cpp/build/bin/dataset-prep
```

## 训练

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)：

```bash
cd python
uv sync
uv run train --run-dir runs/<name>
```

训练参数以 `python/train.toml` 为准。

## 导出 ONNX

```bash
cd python
uv run export-onnx --run-dir runs/<name>
```
