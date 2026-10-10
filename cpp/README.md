# cpp：前处理的 C++ 等价实现

`cpp/` 是交付图 `preprocess.onnx` 的 C++ 等价实现，供 MaaEnd cpp-algo 直接调用，不再经过
onnxruntime（或 ncnn）执行前处理图。**定义仍是 `endfield/preprocess.py`**：C++ 只是它的一份
可执行翻译，靠本目录的测试与定义逐字节对齐，并用 `definition_hash` 锁死对应的定义版本。

```
cpp/
├── include/camori/preprocess.h   接口：camori::preprocess(minimap, asset, x, y, scale, strips)
├── src/preprocess.cpp            实现（约 150 行，只依赖 OpenCV core 的 cv::Mat）
├── src/strip_grid_table.inc      条带网格 (u, v) 的 float32 位模式（生成文件）
├── tests/fixture_test.cpp        fixture 比对测试（误差统计 + 计时）
├── tests/ort_bench.cpp           用同一输入计时 preprocess.onnx（可选）
├── tools/gen_fixtures.py         由定义 + 导出图生成 fixtures
├── tools/gen_strip_grid.py       由定义生成 strip_grid_table.inc（--check 校验是否过期）
├── scripts/                      Windows / Android 构建与真机运行脚本
└── TESTING.md                    测试说明与误差对比
```

## 为什么需要

前处理图没有可学习参数，只是两次双线性 `GridSample` 加窗口裁剪与白底合成，但放在推理框架里
代价不小：

- MaaEnd 升到 MaaDeps v3（onnxruntime 1.29）后，Android 上这张图单次 25～30 ms（小米 12X，
  SM8250），C++ 等价实现约 1.1 ms；
- ncnn 无法执行这张图：数据相关的 `Slice`（窗口裁剪）、`Shape`/`Gather` 表达式与 uint8
  `Cast` 都没有对应层，只有拆图才能上 ncnn，拆完剩下的部分 C++ 一行 `remap` 就能做完。

## 逻辑

与 `endfield/preprocess.py` 一一对应：

| 步骤 | 定义（preprocess.py） | C++ |
| --- | --- | --- |
| 条带网格 | `strip_roi_uv()`：42×360，`r = 12 + (i+0.5)`，`θ = j°`，`u = 59 + r·sinθ`，`v = 60 − r·cosθ` | `StripGrid`，值直接取自生成的 `strip_grid_table.inc` |
| 观测条带 | `sample_minimap()`：ROI 上 bilinear，`padding_mode=border` | `bilinear<3, Border>` |
| 参考窗口 | `_asset_window()`：`extent = scale·(12 + 41.5)`，`[floor(c−extent−2), floor(c+extent+2)+2)` 裁到资产内、非空 | `window_axis()`，零拷贝 `cv::Mat` ROI |
| 参考采样 | `_sample_asset_crop()`：`au = x + (u−59)·scale − w0`，bilinear，`padding_mode=zeros` | `bilinear<4, Zeros>` |
| 合成 | `_compose_strips()`：`w = A/255`，`BGR·w + 255·(1−w)`，A 原样 | 同 |
| 取整 | `_to_uint8()`：`round`（半偶）→ `clamp(0,255)` → `uint8` | `std::nearbyint` + clamp |

两处为逐字节对齐而做的选择：

1. **条带网格查表**：`sin/cos` 在不同 libm（MSVC / bionic / numpy / torch）之间约有 5% 的点差
   1 ulp，最终表现为条带上偶发的 1 LSB 差。表由 `gen_strip_grid.py` 从 `strip_roi_uv()` 原样
   导出（与导出图里存的常量同值），C++ 不再自己算三角函数。
2. **双线性权重形式**：按 torch `grid_sampler_2d` 的写法先算四个角的权重
   `w = dy·dx` 再求和 `w11·p11 + w12·p12 + w21·p21 + w22·p22`，而不是 ORT 1.19 CPU 的
   `dy2·(dx2·p11 + dx1·p12) + …`。两种写法数学等价，float32 下在 `.5` 边界上取整结果可能差 1。

坐标归一化 / 反归一化（`(2p+1)/L − 1` 再 `((n+1)·L − 1)/2`）照样走一遍，保留与图内一致的 float32
舍入；编译选项关闭 fast-math 与 FMA 合并（`/fp:precise`、`-ffp-contract=off`）。

## 构建与测试

见 [TESTING.md](./TESTING.md)。

## 定义变更时

`endfield/preprocess.py` 一改 `definition_hash` 就变：

1. `uv run python cpp/tools/gen_strip_grid.py` 重新生成网格表（表头带新的哈希）；
2. 按定义改动同步 `src/preprocess.cpp`，把 `include/camori/preprocess.h` 的 `kDefinitionHash` 改成新值
   （`strip_grid_table.inc` 与 `kDefinitionHash` 不一致会编译期 `static_assert` 失败）；
3. 重新生成 fixtures 并跑 `camori_fixture_test`（fixtures 与实现哈希不一致时测试直接退出码 3）。

`endfield/preprocess.py` 必须保持 LF（仓库根 `.gitattributes`）：`definition_hash` 取文件原始字节，
Windows `core.autocrlf=true` 检出成 CRLF 会得到另一个哈希。
