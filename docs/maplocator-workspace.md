# 本地 MapLocator 工作台

仓库对 MaaEnd 的依赖收在一个 gitignored 的本地工作台 `local/maplocator/`：定位 CLI 是
[mx6436/MaaEnd](https://github.com/mx6436/MaaEnd) fork 上某个提交的构建物，资源是 MaaEnd
上游工作副本的只读镜像。仓库内脚本只引用这个目录，不引用仓库外路径。本文是该目录的布局、
CLI 契约、重建步骤与资产来源的唯一文档。

## 布局

```
local/maplocator/
├── SOURCE                       # 本机来源快照（见下）
├── bin/map-locate               # 定位 CLI
├── lib/                         # CLI 的共享库依赖（见下）
└── resource/                    # CLI --resource-dir
    ├── image/MapLocator/**      # zone 底图资产（参考底图真源）
    └── model/map/               # cls.onnx / cls.json / tile_mapping.json（小地图检测）
```

| 路径 | 消费方 | 使用条件 |
| --- | --- | --- |
| `bin/map-locate` | `locate-dataset`（批量定位）、`live`（ref 实机流式定位） | 后者要求 CLI 支持 `--stream` |
| `lib/` | 上述 CLI 的启动 | 二进制不是自包含的：RUNPATH 首项是 `$ORIGIN/../lib`，缺库起不来 |
| `resource/` | 上述两者的 CLI 调用（`--resource-dir`） | `image/MapLocator` 与 `model/map` 同时存在才能初始化 |
| `resource/image/MapLocator/` | `prepare-data --mode ref` 裁参考条带 | 路径写进 `processed_ref` 缓存戳 |

路径的具体推导（根目录 → 各叶子）在 `placement/workspace.py`；入口脚本对外只暴露一个
`--maplocator-root`（`prepare-data`、`locate-dataset`），探测失败时报错并指向本文。

## 来源与约束

**CLI**：fork `mx6436/MaaEnd` 的本地分支 `local/maplocator-cli-feat`，提交 `26b52c97f`（2026-10-06，尚未推送 fork）。
以最新主线 `origin/v2` 的 `f4f99c4b0` 为基点，将旧 pin `b5aa2cd69` 的独立 CLI 提交
变基到主线，删除主线已移除的 `MapPosition::isHeld` 输出，不从匹配分数补造标记。旧分支的摄像机模型接入提交不再重放，
使用主线现有实现。CLI 包含静态截图批量定位（stdin 路径、stdout JSONL）、
`--max-attempts`、`--stream` 流式追踪与定位记录 `scale` 字段。此 CLI 不在 MaaEnd 上游。

**资产**：`resource/image/MapLocator`、`resource/model/map` 都是 MaaEnd 上游工作副本对应
路径的镜像，保持原样——不修改、不增删、不手工修补。本仓与 fork 都不维护任何 MaaEnd 数据。

**约束**：对 MaaEnd/MaaEnd 不开 PR/issue；CLI 的变更只在 fork 上做，并把新的分支与提交 pin
写回本文。资产不 pin 上游提交：漂移靠 `SOURCE` 自证与数据戳记账（见「产物溯源」）。

`SOURCE` 是本机工作台的来源快照，重建/刷新时更新：

```
cli_repo=git@github.com:mx6436/MaaEnd.git
cli_branch=local/maplocator-cli-feat
cli_commit=26b52c97f6bd6d1beafe3291a3b1541f7a992c7c
upstream_commit=f4f99c4b0f4bb294e21f4a270c29553f3cfcf5bf
model_commit=d0a08dd86370d3b7a941481800d173fbed92fce5
mirrored_at=<YYYY-MM-DD>
```

## 重建与刷新

```bash
MAAEND=<MaaEnd 工作副本>
DEPS="$MAAEND/agent/cpp-algo/MaaUtils/MaaDeps"

# 1) CLI：fork 的 pin 提交 -> bin/map-locate + lib/
git -C "$MAAEND" fetch mx6436
git -C "$MAAEND" fetch origin
# 当前 pin 尚未推送，需使用已有的本地提交。
git -C "$MAAEND" checkout 26b52c97f
# 首次配置按 MaaEnd 上游文档（docs/zh_cn/developers/getting-started.md）。
# 已配置的本机构建树：加载 MaaEnd shell.nix，仅构建 CLI 目标。
nix-shell "$MAAEND/shell.nix" --run "cmake --build $MAAEND/agent/cpp-algo/build --config RelWithDebInfo --target map-locate --parallel 4"
install -D "$MAAEND/agent/cpp-algo/build/bin/RelWithDebInfo/map-locate" local/maplocator/bin/map-locate
install -D "$MAAEND/agent/cpp-algo/build/bin/RelWithDebInfo/libMaaUtils.so" local/maplocator/lib/libMaaUtils.so
cp -L "$DEPS/vcpkg/installed/maa-x64-linux/lib/libopencv_world4.so.412" local/maplocator/lib/
cp -L "$DEPS/vcpkg/installed/maa-x64-linux/lib/libonnxruntime.so.1" local/maplocator/lib/
cp -L "$DEPS/x-tools/x86_64-linux-gnu/x86_64-linux-gnu/sysroot/usr/lib/libc++.so.1" local/maplocator/lib/
cp -L "$DEPS/x-tools/x86_64-linux-gnu/x86_64-linux-gnu/sysroot/usr/lib/libc++abi.so.1" local/maplocator/lib/
cp -L "$DEPS/x-tools/x86_64-linux-gnu/x86_64-linux-gnu/sysroot/usr/lib/libunwind.so.1" local/maplocator/lib/
cp -L "$DEPS/x-tools/x86_64-linux-gnu/x86_64-linux-gnu/sysroot/lib/libatomic.so.1" local/maplocator/lib/
# libc++ 的间接依赖不继承可执行文件的 RUNPATH。
patchelf --set-rpath '$ORIGIN' local/maplocator/lib/libc++.so.1

# 2) 资产：SOURCE 记录的上游提交，原样镜像
git -C "$MAAEND" checkout <upstream_commit>
git -C "$MAAEND" submodule update --init assets/resource/model
mkdir -p local/maplocator/resource/image/MapLocator local/maplocator/resource/model/map
rsync -a "$MAAEND/assets/resource/image/MapLocator/" local/maplocator/resource/image/MapLocator/
rsync -a "$MAAEND/assets/resource/model/map/" local/maplocator/resource/model/map/
```

`lib/` 是构建产物（MaaDeps 的 vcpkg 库与 x-tools sysroot 库，路径按当前工作副本的
`maa-x64-linux` / `x86_64-linux-gnu` 三元组；`cp -L` 解引用符号链接）；`RelWithDebInfo`
是当前工作台使用的构建配置，其它配置的产物在 `agent/cpp-algo/build/bin/<Config>/`。库齐
后二进制不依赖 MaaEnd 构建树的绝对路径。`libatomic.so.1` 是 libc++ 的间接依赖，
打包的 libc++ 需补 `$ORIGIN` RUNPATH；仅修改本地库副本，不改 MaaDeps 原件。

## CLI 契约

```
map-locate --resource-dir <dir> [--output <jsonl>] [--max-attempts N] [image.png ...]
map-locate --resource-dir <dir> --stream     # 帧路径从 stdin 逐行读，一帧一行 JSON
```

- 无位置参数时从 stdin 读完全部路径再逐张处理；批量模式每张先重置追踪状态，再以强制全局
  搜索最多调用 `--max-attempts`（默认 3）次完成冷启动共识。
- `--stream` 不重置追踪状态、不强制全局搜索、每帧调用一次 `locate()`，追踪状态与冷启动共识
  由连续帧推进；`--stream` 与位置参数互斥。
- JSONL 走 stdout（`--output` 时写文件），日志与错误走 stderr。`resource-dir` 布局要求
  `<dir>/image/MapLocator/**.png` 与 `<dir>/model/map/cls.onnx`（含 `cls.json`、
  `tile_mapping.json`）。
- 无 `--help`；参数错误打印 usage 到 stderr 并以 2 退出。

每张图一行 JSON，字段：

| 字段 | 含义 |
| --- | --- |
| `name` | 图片文件名（样本标识） |
| `status` | MapLocator 状态：`0` Success / `1` TrackingLost / `2` ScreenBlocked / `3` Teleported / `4` YoloFailed / `5` NotInitialized；CLI 级失败为 `-1` 读图失败、`-2` 小地图 ROI 越界 |
| `message` | 状态原文；失败分类见 `summary.json` 的 `excluded_by_reason` |
| `zone` | 定位到的 MapLocator zone（如 `Wuling_Base`、`ValleyIV_L6_109`；tier zone 的 x/y 为切片坐标） |
| `x`, `y` | zone 图上的像素坐标 |
| `rot` | MapLocator 输出的箭头朝向，**不是**摄像机角度 |
| `scale` | 该 zone 的 `ZoneTemplateScale`（参考底图与观测的像素尺度比；无缩放 zone 为 1.0）。定位侧携带的尺度真源：训练/实机侧的参考裁剪消费此字段，不在消费方镜像 zone -> scale 表；定位失败时无 zone，恒为 1.0（不参与消费） |
| `locConf` | 匹配分数（原始值，未加工） |
| `latencyMs` / `elapsedMs` | 单次 locate 内部耗时 / 单图端到端耗时 |
| `attempts` | 该图实际 locate 调用次数（批量模式 1..N，流式恒 1） |
| `accepted` / `accept_reason` | 不是 CLI 字段：`locate-dataset` 解析后写入的入选门标注 |

新版 CLI 不输出 `isHeld`，入选门为 `status == 0 && locConf >= 0.55`。消费端只对历史
记录中显式携带的 `isHeld=true` 保留 `held` 排除；不从低分推导该标记。新版低分记录
归入 `below_loc_threshold`，汇总仅在输入包含历史字段时输出 `held` 计数。

早于 `b5aa2cd69` 的 CLI 不输出 `scale`，消费端（`placement/placement.py` 的 `Placement.from_record`）直接报错。

## 产物溯源

`locate.jsonl` 的 `(zone, x, y, scale)` 由工作台的 CLI 产生，是 ref 管线的输入指纹之一。
`prepare-data --mode ref` 在 `processed_ref/.preprocess.json` 的 `provenance.assets_root`
记录本批数据所用的资产根，换根即缓存失效重生成；训练读该戳写进 run 的 `record.json`
（`ref_reference_assets_root`），`live` 按它加载参考底图。资产根因此只有这一个真源：
数据本身。
