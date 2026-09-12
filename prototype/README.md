# prototype: preprocess 语义候选（#23，throwaway）

本目录是 **一次性原型**，用于回答 issue #23 的问题：`preprocess.onnx` 应采用
「干净重定义」还是「现行 cv2 顺序复刻」，以及 torch 语义与草稿图在 ORT 1.19.2
上的行为是否一致。不是生产代码；生产定义模块在 #25 落地。

## 三个语义

三个变体共用同一 strip 几何（极点、内外径、1°/列），差别只在资产侧的采样映射与
合成次序：

- `clean_ideal`（干净版默认语义）：图内从 `(asset, minimap, x, y, scale)` 直接一次
  采样；参考中心 = 精确 `(x, y)`、尺度 = 精确 `scale`；观测/参考 BGR/alpha 全部在条带
  域一次合成、每输出一次舍入。
- `clean_cv2align`（诊断变体）：同样的单段采样与条带域合成，但资产采样位置逐点复刻
  现行 cv2 链（`round(x)` 中心、`round(118*scale)` 裁剪窗、半像素 `INTER_LINEAR`），
  用于把「结构变化」与「亚像素重对齐」两种差异分离开。
- `replica`（回退路径）：按现行 cv2 顺序两段复刻（全资产黑底合成 → crop/resize →
  观测背底合成 → 展开），中间保留 uint8 舍入。

资产越界一律按参考缺失处理（BGR=0、A=0 → 合成后等于观测）：图上对应
`GridSample(padding_mode="zeros")`；观测侧仍用 `border`（对应 cv2 `BORDER_REPLICATE`，
有效几何下不会触发）。

## 运行

```bash
uv run python -m prototype.export_drafts                       # -> prototype/drafts/*.onnx
uv run python -m prototype.compare_real_samples --limit 600    # 子集快速对拍
uv run python -m prototype.compare_real_samples --limit 0      # 全部 accepted 样本
```

依赖：`data/raw`、`data/locator/locate.jsonl`、本地 MapLocator 底图资产
（`local/maplocator`，gitignored）；ORT 1.19.2（dev 依赖已固定）。

对拍口径：现行实现 = `endfield.ref.ref_strip`（黑底合成按 zone 缓存的等价实现，
启动时对前 5 个样本断言与 `ref_strip` 逐字节一致）。逐样本输出 per-output
（observed / ref BGR / ref alpha / reference 4ch）的 max/mean/p99/差异像素占比、
缺口占比差、以及 ORT vs torch eager 的引擎一致性；最坏样本渲染到
`prototype/out/visuals/`。数据派生的数字只落在 gitignored 的 `prototype/out/`
（`SUMMARY.md`、`summary.json`、`samples.jsonl`、`conformance/*.json`），不入库。

## 已知结论（方法级）

- 三个草稿图在 ORT 1.19.2 上均可加载运行，动态 asset H/W 生效；torch eager 与
  ORT 图输出逐像素差 ≤1 LSB（数值细节见本地产物）。
- `replica` 与现行实现逐像素差 ≤2 LSB、几乎无 >1 像素；`clean_*` 的差异主要来自
  （a）精确 `(x, y)` 的亚像素重对齐与精确 `scale`（仅 `clean_ideal`，集中在
  scale≠1 zone），（b）条带域合成的乘积次序（集中在资产 alpha 边缘），
  （c）scale≠1 时单段采样相对两段重采样的差异。
- 缺口占比差异在多数样本上 <0.01；超差样本集中在非 1:1 zone。分派阈值复核见 map
  #21 的 fog 项。

## 对 #25 的契约提示

- 结构断言目前要求所有 `GridSample` 都 `padding_mode="border"`；干净语义下资产侧必须
  `zeros`（越界=缺口），#25 需要按输出角色区分。
- 3 通道资产：本原型图按 4 通道（BGRA）输入搭建，`asset_rgb_3ch` fixture 无法运行；
  是图内补 alpha 还是入口归一化需在 #25 决定。
- 期望侧目前是现行 cv2 定义；#25 定义模块落地后应以新定义重算 fixture 期望。
