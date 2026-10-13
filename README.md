# 小地图摄像机角度识别

从《明日方舟：终末地》的实时小地图预测摄像机角度（`[0, 360)`，顺时针为正），供 MaaEnd 等自动化工具对齐视角。术语与各环节的权威见 [GLOSSARY.md](./GLOSSARY.md)。

流程：构建 C++ 数据准备工具 → 生成数据集 → 训练 → 导出 ONNX。

## 构建数据准备工具

依赖 MaaEnd submodule 与 MaaDeps：

```bash
git submodule update --init --recursive
python3 cpp/MaaEnd/tools/maadeps-download.py

cmake -S cpp --preset linux-x64
cmake --build cpp/build
```

平台 preset 见 `cpp/CMakePresets.json`。

## 生成数据集

原始截图放在 `data/{train_raw,hard_raw,val_raw}`，由人维护。在仓库根目录执行：

```bash
cpp/build/bin/dataset-prep
```

跳过原因见 `data/processed_ref/report.json`；门槛、参数与产物格式以工具自身（`--help` 与 `cpp/src/main.cpp`）为准。

## 训练

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)：

```bash
cd python
uv sync
uv run train --run-dir runs/<name>
```

参数以 `python/train.toml` 为准。

## 导出 ONNX

```bash
uv run export-onnx --run-dir runs/<name>
```

图落在 run 目录；输入输出契约与校验以 `python/endfield/onnx_contract.py` 为准。
