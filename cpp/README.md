# cpp：预处理的 C++ 等价实现

`cpp/` 是交付图 `preprocess.onnx` 的 C++ 等价实现，MaaEnd cpp-algo 直接调用它，不再经过
onnxruntime（或 ncnn）执行预处理图。**定义仍是 `endfield/preprocess.py`**：C++ 只是它的一份
可执行翻译，靠本目录的测试与定义逐字节对齐，并用 `definition_hash` 锁死对应的定义版本。

```
cpp/
├── src/CameraOrientationPreprocess.h      接口：maplocator::BuildOrientationStrips(minimap, asset, x, y, scale, strips)
├── src/CameraOrientationPreprocess.cpp    实现（只依赖 OpenCV core 的 cv::Mat）
├── src/CameraOrientationAzimuthTable.inc  360 个方位角 sin / cos 的 float32 位模式（生成文件）
├── compat/MaaUtils/NoWarningCV.hpp        独立构建时替代 MaaEnd 同名头文件
├── tests/fixture_test.cpp                 fixture 比对测试（误差统计 + 计时）
├── tests/ort_bench.cpp                    用同一输入计时 preprocess.onnx（可选）
├── tools/gen_fixtures.py                  由定义 + 导出图生成 fixtures
├── tools/gen_azimuth_table.py             由定义生成方位角表（--check 校验是否过期）
├── tools/sync_maaend.py                   把 src/ 原样同步进 MaaEnd（--check 只比对）
├── scripts/                               Windows / Android 构建与真机运行脚本
├── .clang-format                          MaaEnd agent/cpp-algo/.clang-format 的副本
└── TESTING.md                             测试说明与误差对比
```

## 与 MaaEnd 的关系

`src/` 下三个文件就是 MaaEnd `agent/cpp-algo/source/MapLocator/` 里的同名文件，逐字节相同，
只有这里一份在改：

- 改动只在 `cpp/src/` 做，跑完测试后用 `sync_maaend.py` 拷进 MaaEnd，不在 MaaEnd 里直接改；
- `sync_maaend.py --check` 比对两边字节，并确认 MaaEnd 的 `agent/cpp-algo/source/CMakeLists.txt`
  仍对该文件关闭 FMA 收缩（`-ffp-contract=off`）；
- 唯一的 MaaEnd 专有依赖 `<MaaUtils/NoWarningCV.hpp>` 由 `compat/` 替身提供，独立构建按 MaaEnd
  cpp-algo 的告警级别编译（`/W4 /WX`、`-Wall -Wextra -Wpedantic -Werror`），这里能过的那边也能过；
- `src/` 固定 LF（仓库根 `.gitattributes`），与 MaaEnd 的统一 LF 一致；排版按 `.clang-format`
  （MaaEnd CI 用 clang-format 17 自动格式化，新版 clang-format 会把空的 `{}` 写成 `{ }`，以 17 为准）。

## 为什么需要

预处理图没有可学习参数，只是两次双线性 `GridSample` 加窗口裁剪与白底合成，但放在推理框架里
代价不小：

- MaaEnd 升到 MaaDeps v3（onnxruntime 1.29）后，Android 上这张图单次 25～30 ms（小米 12X，
  SM8250），C++ 等价实现约 1.1 ms；
- ncnn 无法执行这张图：数据相关的 `Slice`（窗口裁剪）、`Shape`/`Gather` 表达式与 uint8
  `Cast` 都没有对应层，只有拆图才能上 ncnn，拆完剩下的部分 C++ 一行 `remap` 就能做完。

## 逻辑

与 `endfield/preprocess.py` 一一对应：

| 步骤 | 定义（preprocess.py） | C++ |
| --- | --- | --- |
| 条带网格 | `strip_roi_uv()`：42×360，`r = 12 + (i+0.5)`，`θ = j°`，`u = 59 + r·sinθ`，`v = 60 − r·cosθ` | `StripGrid`：`sin/cos` 取自生成的方位角表，半径与 `u/v` 按 float32 现算 |
| 观测条带 | `sample_minimap()`：ROI 上 bilinear，`padding_mode=border` | `SampleBilinear<3, Border>` |
| 参考窗口 | `_asset_window()`：`extent = scale·(12 + 41.5)`，`[floor(c−extent−2), floor(c+extent+2)+2)` 裁到资产内、非空 | `WindowAxis()`，零拷贝 `cv::Mat` ROI |
| 参考采样 | `_sample_asset_crop()`：`au = x + (u−59)·scale − w0`，bilinear，`padding_mode=zeros` | `SampleBilinear<4, Zeros>` |
| 合成 | `_compose_strips()`：`w = A/255`，`BGR·w + 255·(1−w)`，A 原样 | 同 |
| 取整 | `_to_uint8()`：`round`（半偶）→ `clamp(0,255)` → `uint8` | `ToUint8()`：`std::nearbyint` + clamp |

两处为逐字节对齐而做的选择：

1. **方位角查表**：`sin/cos` 在不同 libm（MSVC / bionic / numpy / torch）之间有若干个点差
   1 ulp（numpy 与 torch 在 360 个点里差 62 个），最终表现为条带上偶发的 1 LSB 差。表由
   `gen_azimuth_table.py` 用定义同样的 torch 表达式算出，C++ 不再自己算三角函数；半径与
   `u/v` 的乘加没有 libm 参与，在 C++ 里按 float32 现算。生成器写表前会用 numpy 按 C++ 的
   运算顺序重放一遍，确认与 `strip_roi_uv()` 逐位相同，定义改了网格几何时生成直接失败。
2. **双线性权重形式**：按 torch `grid_sampler_2d` 的写法先算四个角的权重
   `w = dy·dx` 再求和 `w11·p11 + w12·p12 + w21·p21 + w22·p22`，而不是 ORT 1.19 CPU 的
   `dy2·(dx2·p11 + dx1·p12) + …`。两种写法数学等价，float32 下在 `.5` 边界上取整结果可能差 1。

坐标归一化 / 反归一化（`(2p+1)/L − 1` 再 `((n+1)·L − 1)/2`）照样走一遍（`GridRoundTrip()`），
保留与图内一致的 float32 舍入；编译选项关闭 fast-math 与 FMA 合并（`/fp:precise`、`-ffp-contract=off`）。

## 构建与测试

见 [TESTING.md](./TESTING.md)。

## 定义变更时

`endfield/preprocess.py` 一改 `definition_hash` 就变（只改注释也会变）：

1. `uv run python cpp/tools/gen_azimuth_table.py` 重新生成方位角表（带新的哈希；网格几何变了
   会在这一步报错，先改 `StripGrid()` 与生成器里的重放）；
2. 按定义改动同步 `src/CameraOrientationPreprocess.cpp`，把 `src/CameraOrientationPreprocess.h`
   的 `kPreprocessDefinitionHash` 改成新值（方位角表与它不一致会编译期 `static_assert` 失败）；
3. 重新生成 fixtures 并跑 `camori_fixture_test`（fixtures 与实现哈希不一致时退出码 3）；
4. `uv run python cpp/tools/sync_maaend.py --maaend <MaaEnd>` 同步进 MaaEnd，与按新定义训练的
   `polar_with_ref.onnx` 一起提交（MaaEnd 不收录 `preprocess.onnx`，两者的配套只靠这一步保证）。

`endfield/preprocess.py` 必须保持 LF（仓库根 `.gitattributes`）：`definition_hash` 取文件原始字节，
Windows `core.autocrlf=true` 检出成 CRLF 会得到另一个哈希。
