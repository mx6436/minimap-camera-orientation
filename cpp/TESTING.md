# 测试说明与误差对比

## 测什么

`camori_fixture_test` 对每个 case 跑一次 `maplocator::BuildOrientationStrips`（`src/`，与 MaaEnd 同一份源码），
与两份期望逐字节比较：

| 期望 | 来源 | 角色 |
| --- | --- | --- |
| `def_*` | `endfield/preprocess.strip_pair()`（torch 定义） | **判定基准**：所有 case 与它的最大绝对差 ≤ `--tolerance`（默认 0，即逐字节相同）才算通过 |
| `ort_*` | `preprocess.onnx` 经 onnxruntime 执行 | 只报告不判定，用来和现交付路径对照 |

另外三道守卫：fixtures 的 `definition_hash`（`index.json`）与 C++ 的 `kPreprocessDefinitionHash` 不一致时测试
直接退出（码 3）；`ReadPreprocessDefinitionHash()` 从 fixtures 里的 `preprocess.onnx` 读不出同一个哈希时退出
（码 4，MaaEnd 运行时靠它判定是否启用预测器）；`CameraOrientationAzimuthTable.inc` 与 `kPreprocessDefinitionHash`
不一致时编译期 `static_assert` 失败。

### fixture 分组

由 `tools/gen_fixtures.py` 生成（默认 seed 固定，可复现）：

| 组 | 数量 | 内容 |
| --- | ---: | --- |
| `conf_*` | 8 | `conformance.builtin_scenarios()`：极坐标展开、参考配对、越界裁剪、scale=15/16、alpha 全 0、3 通道资产、空窗口、负坐标 |
| `synth_*` | 120 | 520×600 随机 BGRA 资产（alpha 0～255 随机）上随机位置与缩放，位置在资产外 ±80 px 内，缩放取 1、15/16、0.75、1.25 或 [0.5, 1.5] 随机 |
| `real_*` | 8 | `--maaend` 时：MaaEnd 测试集 8 张真实 720p 截图的小地图 ROI（`(49, 51, 118, 120)`）× 武陵 `Base.png`（2016×2976），坐标取自一次实机采集日志 |
| `rand_*` | 200 | `--maaend` 时：同一真实资产上随机位置与缩放（含越界），小地图一半真实 ROI、一半随机噪声 |

不带 `--maaend` 只生成前两组（128 个），不依赖 MaaEnd 仓库。

## 怎么跑

前置：本仓 Python 环境（`uv sync`，含 torch、onnx、onnxruntime）、CMake ≥ 3.21、Ninja、OpenCV 4（core）。
下面用 MaaDeps 自带的 OpenCV 举例。

```bash
# 1. 生成 fixtures（默认现场导出一份 preprocess.onnx 作为 ort_* 的来源；--onnx 可指定已交付的图）
uv run python cpp/tools/gen_fixtures.py --out cpp/build/fixtures --maaend F:/Project/Golang/MaaEnd

# 2a. Windows（MSVC + Ninja）
cpp\scripts\build-windows.bat F:\Project\Cpp\MaaFramework\source\MaaUtils\MaaDeps\vcpkg\installed\maa-x64-windows\share\opencv4
cpp\build\windows\camori_fixture_test.exe cpp\build\fixtures --bench 50

# 2b. Android arm64（NDK r29，真机运行）
cpp/scripts/build-android.sh /f/Android/SDK/ndk/29.0.13599879 \
    /f/Project/Golang/MaaEnd/agent/cpp-algo/MaaUtils/MaaDeps/vcpkg/installed/maa-arm64-android/share/opencv4 \
    [-DCAMORI_BUILD_ORT_BENCH=ON]
cpp/scripts/run-android.sh cpp/build/fixtures <含 libopencv_world4.so 与 libc++_shared.so 的目录> --bench 50

# 方位角表是否过期
uv run python cpp/tools/gen_azimuth_table.py --check

# MaaEnd 里的副本是否与 src/ 逐字节相同（去掉 --check 即同步）
uv run python cpp/tools/sync_maaend.py --maaend F:/Project/Golang/MaaEnd --check
```

`camori_fixture_test` 参数：`--tolerance N`（判定阈值，默认 0；conformance 剖面是 1）、`--bench N`（每个
case 额外计时 N 次）、`--verbose`（逐 case 打印）。只有失败的 case 会单独打印。

`camori_ort_bench <preprocess.onnx> <fixtures_dir> <case> [iters] [threads]`：在同一 case 的输入上计时
onnxruntime 执行交付图（cpp-algo 的 `CameraOrientationPredictor` 默认 2 线程）。

## 误差对比

定义版本 `definition_hash 223450125d44…`（`b8ba185`），336 个 case（含 `--maaend`）。
数值为逐元素（像素 × 通道）占比；observed 每 case 42×360×3，reference 42×360×4。

### C++ vs 定义（判定基准）

Windows x64（MSVC 19.51.36256.0，`/fp:precise`）与 Android arm64（NDK r29 clang 20.0.0，`-ffp-contract=off`，小米 12X）
结果相同。下表最初由 u/v 全表版本测得；现行实现（方位角查表、u/v 现算，与 MaaEnd 同一份源码）算出的网格
与全表逐位相同，两平台复测结果不变：

| 组 | cases | observed 相同 | reference 相同 | 最大差 |
| --- | ---: | ---: | ---: | ---: |
| conf | 8 | 100% | 100% | 0 |
| synth | 120 | 100% | 100% | 0 |
| real | 8 | 100% | 100% | 0 |
| rand | 200 | 100% | 100% | 0 |
| **全部** | **336** | **100%** | **100%** | **0** |

C++ 与定义逐字节相同，不需要 conformance 的 1 LSB 容差。

### 交付图（onnxruntime）vs 定义（参照）

同一批 fixtures，`ort_*` 与 `def_*` 之差，即交付图经 onnxruntime 执行时自身相对定义的误差
（`ort_*` 由生成 fixtures 的 Python 环境里的 onnxruntime 1.26.0 CPU 产生；本仓 dev 依赖钉的是 1.19.2，
用 `uv run` 生成时以实际版本为准）：

| 组 | cases | observed 相同 | observed 差 1 | reference 相同 | 最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| conf | 8 | 99.99780% | 0.00220% | 100% | 1 |
| synth | 120 | 99.99848% | 0.00152% | 100% | 1 |
| real | 8 | 100% | 0% | 100% | 0 |
| rand | 200 | 99.99969% | 0.00031% | 100% | 1 |
| **全部** | **336** | **99.99922%** | **0.00078%** | **100%** | **1** |

113 个 case 里共 119 个元素差 1，全部在 observed（`padding_mode=border` 那一路）。抽查的差异点都是
双线性结果落在 `.5` 边界、两种求和写法舍入到两侧的情况。例：`conf_polar_basic` 第 33 行第 90 列 B 通道，采样点
`(104.5, 60.000008)`，ORT 1.19 CPU 的写法 `dy2·(dx2·p11 + dx1·p12) + dy1·(…)` 得到 `174.5` → 半偶取整
`174`；torch 先算四角权重再求和得到 `174.50002` → `175`。两者都在 conformance 的 1 LSB 容差内；
C++ 按 torch 的写法，所以与定义一致、与 ORT 在这 119 个元素上差 1。

reference（`padding_mode=zeros`）一路与定义 100% 相同。onnxruntime 新版本对 bilinear + zeros 有单独的
快路径（预先算四角权重 `w = dy·dx` 再求和，与 torch 写法相同），推测是这一路没有差异的原因。

`--onnx` 指向 MaaEnd 现交付的 `assets/resource/model/map/cameraorientation/preprocess.onnx` 结果相同：
该文件与本仓 `export-preprocess` 现场导出的图 `definition_hash` 相同、graph 逐字节相同。

### 早期版本（用于说明两处对齐选择）

Windows 上逐步对齐的过程（与定义比，最大差均为 1，直到最后一步）：

| 版本 | observed 相同 | reference 相同 |
| --- | ---: | ---: |
| 现场算 `sin/cos` + ORT 1.19 双线性写法 | 99.99873% | 99.99960% |
| u/v 全表 + ORT 1.19 双线性写法 | 99.99922% | 99.99977% |
| u/v 全表 + torch 权重写法 | 100% | 100% |
| 方位角查表、u/v 现算 + torch 权重写法（当前） | 100% | 100% |

现场算 `sin/cos` 时，Android（bionic libm）与 Windows 的差异位置也不同，查表后两平台结果一致。u/v 全表
（15120×2）换成方位角表（360×2）后网格逐位不变：误差只来自 `sin/cos`，半径与乘加在 float32 下各平台一致。

## 耗时

同一 case（`real_00`，武陵 Base 2016×2976，scale 1），单次 `BuildOrientationStrips`：

| 平台 | C++（单线程） | onnxruntime 执行交付图 |
| --- | ---: | ---: |
| 小米 12X（SM8250，Android arm64） | **1.08 ms** | ORT 1.29.0：1 线程 30.5 ms / 2 线程 25.6 ms |
| 桌面 x64 | **0.95 ms** | ORT 1.26.0（Python）：1.30 ms |

Android 上 ORT 1.29 慢主要是 MaaDeps v3 的 onnxruntime 构建问题（同机 1.19.2 的推理普遍快约 5～7 倍），
但即使与桌面 ORT 比，C++ 也不慢，并且省掉了 asset 整张转 tensor 的开销与 onnxruntime 依赖。
