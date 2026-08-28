# data 目录除 raw 与验证清单外均为可再生输出

数据前处理（格式变换 + 训练/验证划分）曾分散在三个脚本中，`split_manifest.json` 被当作跨运行资产维护（复用、变更守卫、`--resplit`），且 rgba 与 polar 两条管线各自维护一套 processed/train/val 目录。这些半持久状态是维护负担：划分结果本质上是 seed 确定性的纯函数输出，持久化它只能制造"manifest 与实际数据不一致"这类问题。

决定：将三个前处理脚本合并为 `prepare_data.py`（`--input {rgba,polar}` × `--split {random,manifest}` 两个正交维度）；除 `data/raw`（只读事实来源）与 `data/val_manifest.json`（人工维护的验证集清单）外，`data/` 下所有内容都在每次运行时清空重写，`split_manifest.json` 仅记录最近一次运行的结果，不再是跨运行资产；`--resplit` 与变更守卫随之删除。处理输出共用单个 `data/processed`，两种格式不能在磁盘上共存，切换格式即重跑脚本。

权衡：放弃持久化意味着新增 raw 数据后重跑随机切分会重新洗牌（已有验证文件可能被换出）——接受这一语义，换取"脚本输出可以随时无损重建"的不变量。清单切分不受此影响（验证集成员由清单固定）。

Considered options：
- 保留跨运行 manifest 与 `--resplit` 守卫：被否，守卫逻辑复杂且只在"数据变了但人没注意"时有价值，而这正应该通过重跑来体现。
- 为清单生成留脚本：被否，清单是一次性的人工圈定，生成方式记在来源元数据里即可。
