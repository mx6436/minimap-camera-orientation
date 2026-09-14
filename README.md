# 小地图摄像机角度识别

从《明日方舟：终末地》的实时小地图预测摄像机角度（`[0, 360)`，顺时针为正），供 MaaEnd 等自动化工具对齐视角。模型在极坐标展开的条带上做 360 bin 分类，输出角度概率分布。

## 背景

小地图是世界锚定的：地形永远正北朝上，不随摄像机旋转，画面中唯一可靠的朝向信号是视野扇形。中心箭头指示的是角色朝向，与摄像机朝向可能不同步甚至相反，预处理时必须排除。完整术语与约束见 [CONTEXT.md](./CONTEXT.md)。

## 两种输入模式

观测小地图在极坐标下展开成条带后喂给模型：

- **`polar`**：只用观测条带，3 通道 BGR。
- **`ref`**：加上一路参考条带，7 通道 `[obs.BGR, ref.BGR, ref.A]`。参考由 MapLocator 定位到的 zone 底图按定位坐标裁出同视野、同尺度的一小块；`ref.A` 标记参考缺失（0 = 缺失）。模型对比两者地形差异判断朝向，精度更高，代价是数据管线多一步定位。

## 快速开始

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
uv run pytest
```

### 数据

单张截图是训练/验证样本，文件名以 `_r<角度>.png` 结尾标注摄像机角度，支持一位小数（如 `_r210.9.png`）。截图按训练侧与验证侧分开准备：

- `data/train_raw`：训练侧原始截图。由人维护，脚本只读。
- `data/val_raw`：验证侧，语义同上。划分由样本所在目录决定；跨侧同名样本会硬报错。

### polar

```bash
uv run prepare-data                     # raw -> data/processed，并建 data/train、data/val 视图
uv run train --run-dir runs/<name>      # 训练参数见 train.toml
uv run live --run-dir runs/<name>       # 实机预览，见下
```

### ref

先批量定位（可断点续跑），再生成观测与参考两路：

```bash
uv run locate-dataset                   # -> data/locator
uv run prepare-data --mode ref          # -> data/processed_ref，并建 data/train_ref、data/val_ref
uv run train --run-dir runs/<name>      # train.toml 里 input_mode = "ref"
```

定位与 ref 实机推理依赖本机的 MapLocator 工作台 `local/maplocator/`，搭建见 [docs/maplocator-workspace.md](docs/maplocator-workspace.md)。

### 数据目录

- `data/processed`（polar）/ `data/processed_ref`（ref）：前处理产物，挂缓存戳，定义或输入变化时重生成。
- `data/train`、`data/val`（polar）/ `data/train_ref`、`data/val_ref`（ref）：划分视图，符号链接到 processed。
- `data/locator/`：ref 的定位产物（`locate-dataset` 增量维护）。
- `runs/<name>/`：一次训练的产物目录（checkpoint、`record.json`、指标与曲线），由 `--run-dir` 指定，`live` 与导出从这里读。

## 实机预览

游戏在 gamescope 会话中运行时，叠加显示圆盘、模型输入与概率曲线：

```bash
uv run live --run-dir runs/<name>                            # 输入模式由 run 的 record.json 决定
uv run live --run-dir runs/<name> --snapshot <overlay 路径>    # 保存一张 overlay 后退出
```

## 交付

`export-artifact` 一次导出交付 bundle：`preprocess.onnx` + `polar.onnx` + `polar_with_ref.onnx` + `manifest.json`。

```bash
uv run export-artifact --out runs/<name>/bundle \
    --polar-run runs/<polar_run> --ref-run runs/<ref_run>
```

### 工件校验（conformance）

```bash
uv run verify-artifact --bundle runs/<name>/bundle   # 结构自检 + 内置场景数值比对
```

判定口径与其余用法见 `--help`，契约细节见 [docs/agents/engineering.md](docs/agents/engineering.md)。

### 拷入 MaaEnd

bundle 三图对应 MaaEnd 交付布局 `assets/resource/model/map/cameraorientation/`，`manifest.json` 留在本仓作为交付凭据。拷入与提交在 MaaEnd 的模型子模块内完成：

```bash
MAAEND=<MaaEnd 工作副本>        # 切到 feat/camera-orientation
BUNDLE=runs/<name>/bundle
mkdir -p "$MAAEND/assets/resource/model/map/cameraorientation"
cp "$BUNDLE"/{preprocess,polar,polar_with_ref}.onnx \
   "$MAAEND/assets/resource/model/map/cameraorientation/"
cd "$MAAEND/assets/resource/model"
git add map/cameraorientation && git commit -m "model: cameraorientation 三图工件" && git push
```

## 文档

- [CONTEXT.md](./CONTEXT.md)：领域术语与核心约束
- [docs/adr/](docs/adr/)：架构决策记录
- [docs/maplocator-workspace.md](docs/maplocator-workspace.md)：本地工作台布局与重建
- [docs/agents/engineering.md](docs/agents/engineering.md)：工程契约、不变量与模块归属（改代码前读）
