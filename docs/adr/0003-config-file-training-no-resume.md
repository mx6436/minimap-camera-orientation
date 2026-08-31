# 训练入口极简化：配置文件制与无恢复能力

train.py 曾把全部实验参数放在命令行（15 个 flag），并维护一套完整的断点恢复机器：RNG 状态捕获/恢复、optimizer/scheduler 状态持久化、跨二十余个键的 checkpoint 配置严格校验。参数散落在命令行里，实验配置无法以文件形式留存与审阅；恢复机器则占去训练代码近三分之一，服务的却是数百 epoch 规模、重跑代价可接受的训练。

决定：

- 模型、损失、优化与增强的全部训练参数（14 项）迁入根目录 `train.toml`（TOML，标准库 `tomllib` 读取）。所有键都有代码内默认值，文件明示当前基线；未知键硬报错——拼错键名必须失败，不许静默回落默认值浪费一次训练。CLI 只保留调用管道（`--config`、`--output-dir`、`--device`、`--threads`、`--smoke`）。
- resume 能力整体移除。checkpoint 缩为两键 `{"model": state_dict, "config": 重建架构所需 kwargs}`；`last.pt` 废除，仅在验证指标创新低时原子覆写 `best.pt`；训练中断即从头重跑，早停与 best 追踪在内存内照常工作。
- 默认架构翻转为生产基线：2x22 细粒度读出网格、读出前 1×1 卷积压缩到 64 通道、半径轴 avg 池化、GroupNorm、dropout 0，共 937,872 可训练参数。裸构造 `AngleCNN()` 即 production 模型，参数计数绊线随基线更新。
- checkpoint 加载与设备选择收进 `model.py`（`load_model` / `choose_device`），训练与推理脚本互不依赖。

权衡：中断 = 丢弃全部进度重跑，换取删除 RNG 恢复、optimizer/scheduler 状态与配置校验三大块机器；checkpoint 不再携带任何恢复状态，缺架构键的旧 checkpoint 加载时响亮报错，不做迁移。

Considered options：

- 保留可选 resume：被否，恢复机器是训练代码最大的复杂度来源，本项目训练规模下重跑成本可接受。
- 每实验一份 config 文件存入输出目录：被否，输出目录已有解析后的 `config.json` 记录可溯源，输入文件只需一份。
- batch size 等训练参数压成代码常量：被否，它们与架构参数同样参与决定训练产物，没有理由区别对待。
