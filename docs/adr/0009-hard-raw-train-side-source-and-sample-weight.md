# `data/hard_raw` 是训练侧第三个原始样本目录，困难样本身份即目录成员，权重单点收在 `hard_weight`

人手工截图选出的困难样本放进 `data/hard_raw/`，训练时按一个全局权重加权。这条决定把两件事
一起定死：困难样本的身份由**目录成员身份**表达（不是标记覆盖层），权重由 `train.toml` 的
`hard_weight` 单点表达（全局一个值，不做逐样本权重）。

`data/hard_raw/` 因此是训练侧的第三个原始样本目录，与 `data/train_raw` 并列：样本只存在于
该目录（不再同时留在 `train_raw`），训练侧样本并集 = `train_raw ∪ hard_raw`，val 侧仍只有
`val_raw`。口径单点收在 `endfield/dataset.py` 的 `TRAIN_RAW_DIRS`：`raw_samples` 覆盖三个
目录，同名跨目录仍走 `union_png_samples` 的硬报错（碰撞即说明样本没搬干净）；`locate-dataset`
的 `RAW_DIRS` 与训练侧/验证侧名单都由同一个值导出，漏掉训练目录会让困难样本在 ref 侧静默进
`skipped`。

权重的语义就是「复制成 N 份」：损失按 `Σw·KL / Σw` 归一，`w = hard_weight`（名字在
`hard_raw`）否则 `1`。这与「把该样本复制 N 份、在扩容后的集合上取普通均值」逐个数值相等，
是这条方案的验收口径。名单在 `max_ref_missing` 过滤**之后**与训练划分求交——困难样本同样要
过 `prepare-data` 与（ref 模式的）定位入选门，被剔除的样本不进训练划分，权重也就落不到它
头上（`cli/train.py` 对「有文件但全部落空」给出显式警示）。

## Considered Options

- **标记覆盖层**（样本留在 `train_raw`，另有一份「这是困难样本」的标记）：同一份样本会有两个
  来源，并集、缓存戳与 `record.json` 的 `train_files_sha256` 都要额外定义去重口径。
  「把文件搬进 `hard_raw`」把困难身份变成一次可核对的文件移动，且没有第二处真相。
- **过采样（`WeightedRandomSampler`）**：改的是采样分布而非损失口径，「等价于复制 N 份」的
  验收表述不再成立，同一 epoch 内样本出现次数随机，读数不可复述。
- **两段式微调 / 课程学习**：需要两套训练阶段与超参，超出「一个全局权重」的量级。
- **逐样本权重 / 按 loss 自动挖掘（focal、OHEM）**：要再引入一份权重表或在线挖掘逻辑，
  首批只有 4 张手工样本，不值得。
- **`data/hard_val/` 与困难子集单独指标**：val 侧无权，加目录会催生第二套指标口径。
- **把权重做成配置里的名单 → 权重映射**：困难身份会同时存在于文件系统与配置两处，两边不一致
  时无从判断。

## Consequences

- **并集不变量**：样本在 `train_raw` / `hard_raw` 之间移动不改变训练侧并集，故 `processed*`
  缓存戳仍命中（不触发重渲染），`data/train` 的名字集合与 `record.json` 的
  `train_files_sha256` 都不变——加权 run 与基线 run 的档案只差 hard 字段，可直接对照。
- 加权不改变模型、前处理定义与交付物：`definition_hash`、图结构、bundle 均不动。
- `hard_weight` 只进训练损失：`eval_loss` 忽略 loader 的第三项，val 损失与全部 val 指标
  保持无权；`AngleDataset` 只认调用方传入的名单与权重，不认识 `data/hard_raw`。
- 续训前提：`hard_weight` / `hard_count` / `hard_names_sha256` 落进 `record.json` 的
  metadata，改动权重或增删落在划分内的困难样本都会让 `--resume` 按既有口径拒绝（数据前提
  变了），不需要额外代码。
- 已知局限：加权后可见读数只有整体 val RMS。首批 4 张在 7882 张训练集里约占 0.25% 的损失
  份额，是否有效主要看整体指标能否移动。
