# 本地工作台契约与资产根溯源

仓库对 MaaEnd 的依赖收在一个 gitignored 的本地工作台 `local/maplocator/`：定位 CLI 是
mx6436/MaaEnd fork 上某个提交的构建物，资源是 MaaEnd 上游的只读镜像。布局、CLI 契约、
重建步骤与资产来源的唯一文档是 `docs/maplocator-workspace.md`；各消费方只经
`endfield/maplocator.py` 取路径，`prepare_data.py --mode ref` 把所用资产根写进
`processed_ref` 缓存戳，run 档案的 `ref_reference_assets_root` 取自该戳，而不是训练配置。

## Considered Options

- **每脚本各持叶子开关 + `train.toml` 声明资产根**（原状）：同一路径决策散在
  `locate_dataset.py`、`prepare_data.py`、`live.py`、`train.toml` 四处（`prepare_data.py`
  还拆成 assets 与 ZmdMap 两个开关，而它们永远同源同变）；数据由资产根 A 生成、
  `record.json` 记资产根 B 时没有任何检查。
- **硬 pin 上游资产提交**：把上游数据变成本仓维护的版本，与「资产只作镜像」冲突；弃用。
  资产漂移靠 `SOURCE` 自证与数据戳记账，而不是给镜像加锁。

## Consequences

- 对 MaaEnd/MaaEnd 不开 PR/issue；CLI 变更只在 fork 上做，fork 分支与提交 pin 在工作台文档里。
- 本仓与 fork 都不维护任何 MaaEnd 数据；工作台的资产目录永远可以从上游工作副本重新镜像。
- 训练配置不再有 `map_assets_root`：带该键的旧 `train.toml` 会因未知键硬报错。
- 无 `provenance` 的旧 `processed_ref` 数据在训练时硬报错，需重跑 `prepare_data.py --mode ref`
  补戳；换资产根会触发数据重生成（戳随内容失效，不是仅记账）。
