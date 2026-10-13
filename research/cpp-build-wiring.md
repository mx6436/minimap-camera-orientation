# Research: C++ 数据准备工具的构建接线与源码清单

Ticket: wayfinder「研究」票 — C++ 数据准备工具的构建接线与源码清单
Branch: `research/cpp-build-wiring`
Target source (authoritative): `/home/mx/Projects/minimap-camera-orientation/cpp/MaaEnd`
(submodule `cpp/MaaEnd`, branch `refactor/cameraorientation-preprocess`,
HEAD `2c65ca2eca944d448702c2ffb2f2881fd85e3f1d` — matches the worktree gitlink
`git ls-tree HEAD cpp/` → `160000 commit 2c65ca2eca944d448702c2ffb2f2881fd85e3f1d`).

Legend: **[直接证据]** = reproduced on this machine; **[推断]** = reasoned from code/config
without a full run.

---

## 0. TL;DR / 结论

- 可以用 CMake 构建一个**不依赖 MaaFramework 运行时**的独立可执行文件：只编译
  `agent/cpp-algo/source/MapLocator/` 下 7 个 `.cpp` + 链接 `MaaUtils`（MaaUtils 自身
  只依赖 Boost / OpenCV / ONNX Runtime，不 include 任何 `MaaFramework/*`）。
- **已在本机实际 configure + build + link + run 成功** **[直接证据]**（用 MaaDeps
  的 clang 工具链 + vcpkg 依赖，见 §6）。`MapLocator::initialize` 返回成功、
  `ZoneTemplateScale("ValleyIV_Base")` 输出 `0.9375`（=15/16）。
- 唯一的运行时依赖是 ONNX Runtime + OpenCV shared libs（MaaDeps `runtime/` 下），
  MaaFramework 既不需要头也不需要库。`MapLocateAction.cpp`（含 MaaFramework）与
  `utils.h` **不在**最小源集合内。
- 引用路径：本仓 `cpp/MaaEnd` 是被研究仓库的子模块，源码/依赖都在其内部。

---

## 1. 最小源文件集合

### 1.1 必需的 `.cpp`（7 个，全部在 `agent/cpp-algo/source/MapLocator/`）**[直接证据]**

| 文件 | 作用 | 证据 |
|---|---|---|
| `MapLocator.cpp` | `MapLocator::Impl`、所有调度逻辑 | `MapLocator.h:11-29` |
| `MapAlgorithm.cpp` | `GenerateMinimapMask` / `InferYellowArrowRotation` | `MapAlgorithm.h:9-11` |
| `MatchStrategy.cpp` | `IMatchStrategy` / `MatchStrategyFactory` / `CoreMatch` | `MatchStrategy.h:54-116` |
| `MotionTracker.cpp` | `MotionTracker` | `MotionTracker.h:11-51` |
| `YoloPredictor.cpp` | YOLO 分类器（ONNX Runtime） | `YoloPredictor.h:17-63` |
| `CameraOrientationPredictor.cpp` | 参考配对分类器推理 | `CameraOrientationPredictor.h:1-14` |
| `CameraOrientationPreprocess.cpp` | `BuildOrientationStrips`（纯 OpenCV，无模型） | `CameraOrientationPreprocess.h:26`, `.cpp:79` |

这 7 个文件互相 include，且只 include 以下外部头（`grep -n '#include'` 的汇总，见
`MapLocator/*.cpp|*.h`）：

- `MaaUtils/NoWarningCV.hpp`、`MaaUtils/Logger.h`、`MaaUtils/Platform.h`、`MaaUtils/ImageIo.h`
- `<opencv2/opencv.hpp>`（`MotionTracker.h:5`）
- `<onnxruntime/onnxruntime_cxx_api.h>`（`YoloPredictor.h:5`, `CameraOrientationPredictor.h:9`）
- `<boost/regex.hpp>`（`MapLocator.cpp:22`, `YoloPredictor.cpp:7`）
- `<meojson/json.hpp>`（`MapTypes.h:3`, `MapLocator.cpp:23`, `YoloPredictor.cpp:8`）
- `../Common/JsoncFile.h`（`YoloPredictor.cpp:10`）— **header-only**，无对应 `.cpp`，不需要额外编译单元
- 同目录 `MapLocator.h` / `MapTypes.h` / `MatchStrategy.h` / `MotionTracker.h` /
  `YoloPredictor.h` / `MapAlgorithm.h` / `CameraOrientationPredictor.h` / `CameraOrientationPreprocess.h`

### 1.2 明确**排除**的文件 **[直接证据]**

- `MapLocator/MapLocateAction.cpp`：include `<MaaFramework/MaaAPI.h>`（`.cpp:9-10`）
  与 `../utils.h`（`.cpp:19`），是 MaaFramework Recognition/行动层适配器，不属于库本体。
- `MapLocator/MapLocateAction.h`：include `MaaFramework/MaaAPI.h`（`.h:3`）。
- `Common/*.cpp`（GameRegion / FramelessWindow / WebView2 / …）：不被上述 7 个文件 include。
- `source/utils.h`：只在 `MapLocateAction.cpp` 里被 include。

### 1.3 include 根 **[直接证据]**

只需下列三个 include 目录即可编译全部 7 个文件（已用 `g++ -fsyntax-only` 逐一验证，7/7 exit=0）：

```cmake
target_include_directories(<exe> PRIVATE
    ${CPP_ALGO_SOURCE}                                  # .../agent/cpp-algo/source
    ${MAAUTILS}/include                                 # .../MaaUtils/include  (MaaUtils/..., meojson/...)
    ${OPENCV_INCLUDE_DIRS}                              # 通常由 find_package(OpenCV) 带入
)
```

- `source/` 作为 include 根，因此 `#include "MapLocator.h"`（同目录）与
  `#include "../Common/JsoncFile.h"` 都能解析（相对 include 以文件所在目录为基准）。
- `<onnxruntime/...>` 与 `<opencv2/...>` 由 `find_package` 的 imported target 带入
  （`ONNXRuntime::ONNXRuntime`、`OpenCV_LIBS`/`OpenCV_INCLUDE_DIRS`）。
- MaaUtils 自己的 `MaaUtils.cmake:43` 也 `include_directories(${MAAUTILS_DIR}/include)`，
  所以只要 include 了 `MaaUtils.cmake`，`MaaUtils/include` 就是全局的。

> 注意：`cpp-algo` 官方 CMake 里出现的 `${MAA_PRIVATE_INC}` / `${MAA_PUBLIC_INC}`
> (`agent/cpp-algo/source/CMakeLists.txt:30,66`) 来自 MaaAgentServer/MaaFramework 包，
> **本工具不需要**，不要照抄。

---

## 2. MaaDeps 接入

### 2.1 `MaaUtils.cmake` 做了什么 **[直接证据]**

`agent/cpp-algo/MaaUtils/MaaUtils.cmake`（副本
`/home/mx/Projects/MaaEnd/agent/cpp-algo/MaaUtils/MaaUtils.cmake`）：

- `:18-19` `set(MAADEPS_DIR ${MAAUTILS_DIR}/MaaDeps)` → `include(${MAADEPS_DIR}/maadeps.cmake)`
- `:22-24` include `cmake/config.cmake`（C++20、`-Wall -Werror`、RPATH、编译选项）、
  `cmake/utils.cmake`、`cmake/version.cmake`
- `:26-30` 依赖查找：
  - `find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)`
  - `find_package(Boost REQUIRED CONFIG COMPONENTS system regex)`
  - `find_package(ZLIB REQUIRED)`
  - `find_package(fastdeploy_ppocr REQUIRED)`  ← **注意：这是 REQUIRED**
  - `find_package(ONNXRuntime REQUIRED)`  ← 走 `MaaUtils/cmake/modules/FindONNXRuntime.cmake`
- `:45-47` `BUILD_MAA_UTILS=ON`（默认）时 `add_subdirectory(source)` 生成 **SHARED**
  库目标 `MaaUtils`（`MaaUtils/source/CMakeLists.txt:4`）。

链接目标名：
- OpenCV：`${OpenCV_LIBS}`（OpenCVConfig 变量；本 vcpkg 是 `libopencv_world4.so.412` 单库）
- ONNX Runtime：`ONNXRuntime::ONNXRuntime`（由 `FindONNXRuntime.cmake:24` 定义）
- Boost：`Boost::regex` / `Boost::system`
- ZLIB：`ZLIB::ZLIB`
- MaaUtils：目标 `MaaUtils`

### 2.2 `maadeps.cmake` 的作用 **[直接证据]**

`agent/cpp-algo/MaaUtils/MaaDeps/maadeps.cmake`：

- `:2-32` 由 `CMAKE_SYSTEM_NAME` / `CMAKE_SYSTEM_PROCESSOR` 自动推导 triplet
  → Linux x64 为 `maa-x64-linux`
- `:40-47` 校验 `MaaDeps/vcpkg/installed/${MAADEPS_TRIPLET}` 存在，否则 FATAL_ERROR
  并提示跑 `maadeps-download.py`
- `:49-53` **非交叉编译**时 `list(PREPEND CMAKE_PREFIX_PATH .../vcpkg/installed/${triplet})`；
  **交叉编译**（设置了 `CMAKE_SYSTEM_NAME`，CMake 会把 `CMAKE_CROSSCOMPILING=TRUE`）时
  改为 prepend 到 `CMAKE_FIND_ROOT_PATH`
- `:55-57` `ZLIB_USE_STATIC_LIBS ON`（默认）
- `:59-65` `maadeps_install()`：从 `MaaDeps/runtime/${triplet}/` 拷运行时 so

> ⚠️ **NixOS 关键坑（已复现）**：只用 `MaaDeps/cmake/maa-x64-linux-toolchain.cmake`
> 时，`CMAKE_SYSTEM_NAME` 被显式设置 → `CMAKE_CROSSCOMPILING=TRUE` →
> `maadeps.cmake` 把依赖前缀放进 `CMAKE_FIND_ROOT_PATH` 而非 `CMAKE_PREFIX_PATH`，
> 导致 `find_package(OpenCV ...)` 报
> `Could not find a package configuration file provided by "OpenCV"`。
> 解决：显式传 `-DCMAKE_PREFIX_PATH=<...>/vcpkg/installed/maa-x64-linux`
> （`~/Projects/MaaEnd/shell.nix:40` 正是为此在 `shellHook` 里导出 `CMAKE_PREFIX_PATH`）。

### 2.3 是否需要联网 / 离线路径 **[直接证据]**

- `MaaDeps/` 在 MaaUtils 仓库里被 **gitignore**（`MaaUtils/.gitignore` 末行 `MaaDeps`），
  `git ls-files MaaDeps` 计数为 0。所以 clone 出 MaaUtils 后**必须**额外获取 MaaDeps。
- 官方下载脚本：
  - MaaUtils 内：`agent/cpp-algo/MaaUtils/tools/maadeps_download.py`（按 triplet 下载）
  - 仓库根包装：`tools/maadeps-download.py` → `REPO="MaaXYZ/MaaDeps"`, `VERSION="v3.0.0"`
    （MaaEnd 根 `tools/maadeps-download.py:11-12`）
  - 下载源是 GitHub release（`maadeps_download.py` 里 `https://api.github.com/repos/{repo}/releases/tags/{version}`），**需要联网**。
- **离线可用路径**：直接复用已备好的副本
  `/home/mx/Projects/MaaEnd/agent/cpp-algo/MaaUtils/MaaDeps`
  （内含 `vcpkg/installed/maa-x64-linux`、`runtime/maa-x64-linux`、`x-tools/llvm`、
  `tarball/MaaDeps-x64-linux.tar.xz`）。把它拷进 checkout 的
  `agent/cpp-algo/MaaUtils/MaaDeps` 即可离线 configure/build。
  也可把已下载的 `tarball/MaaDeps-x64-linux.tar.xz` 放进该目录后用
  `tools/maadeps-extract.py` 离线解包。

### 2.4 MaaUtils 是否依赖 MaaFramework **[直接证据]**

`grep -rn 'MaaFramework\|MaaAPI\|MaaToolkit\|MaaAgent' include source` 在 MaaUtils 内
**只命中一条注释**（`source/Runtime/Runtime_Win.cpp:25` 的 GitHub issue 链接）。
即 `Logger` / `Platform` / `ImageIo` 等**不依赖 MaaFramework 头或库**。
本机实际链接 MaaUtils（12 个源文件）+ 7 个 MapLocator 源文件 → 成功产出可执行文件，
未链接任何 MaaFramework（§6）。

---

## 3. 嵌套 submodule 获取

### 3.1 URL 与 pinned commit **[直接证据]**

`cpp/MaaEnd/.gitmodules`（子模块仓库自己的）：

```
[submodule "agent/cpp-algo/MaaUtils"]
    path = agent/cpp-algo/MaaUtils
    url = https://github.com/MaaXYZ/MaaUtils.git
[submodule "assets/resource/model"]
    path = assets/resource/model
    url = https://github.com/MaaEnd/MaaEnd-AI.git
[submodule "tests/MaaEndTestset"]  url = https://github.com/MaaEnd/MaaEndTestset
[submodule "Android/MaaFwApp"]     url = https://github.com/Aliothmoon/MaaFwApp.git
```

`git submodule status`（在 `/home/mx/Projects/minimap-camera-orientation/cpp/MaaEnd`）：

```
-f8994a38ddc6858800dfea8a04e958a0adbfb184 agent/cpp-algo/MaaUtils
-c674ceff042408687efe4e05c320d6b0865645c2 assets/resource/model
```

- MaaUtils pinned = `f8994a38ddc6858800dfea8a04e958a0adbfb184`，
  与副本 `/home/mx/Projects/MaaEnd/agent/cpp-algo/MaaUtils` 的 `git rev-parse HEAD`
  **一致** **[直接证据]**。
- model pinned = `c674ceff042408687efe4e05c320d6b0865645c2`，
  与副本 `/home/mx/Projects/MaaEnd/assets/resource/model` 的 HEAD **一致**，
  其 remote 也是 `https://github.com/MaaEnd/MaaEnd-AI.git` **[直接证据]**。
- 外层 worktree 的 `cpp/MaaEnd` URL 是 fork：
  `/tmp/minimap-cpp-build-wiring/.gitmodules` →
  `url = git@github.com:mx6436/MaaEnd.git`, `branch = refactor/cameraorientation-preprocess`。

### 3.2 初始化命令 **[推断]**

先初始化外层子模块（会 checkout 到 pin 的 `2c65ca2e…`），再递归取嵌套：

```bash
# 在本仓根目录
git submodule update --init --recursive cpp/MaaEnd
# 或只取需要的两个嵌套（在 cpp/MaaEnd 内）
git -C cpp/MaaEnd submodule update --init agent/cpp-algo/MaaUtils assets/resource/model
# 之后（需要联网，或按 §2.3 离线拷贝 MaaDeps）
python3 tools/maadeps-download.py x64-linux   # 在 cpp/MaaEnd 根执行
```

（`tools/MaaEndTestset`、`Android/MaaFwApp` 与本工具无关，可不取。）

### 3.3 模型库路径 **[直接证据]**

`assets/resource/model/`（= MaaEnd-AI 子模块）下：

```
assets/resource/model/map/cls.onnx
assets/resource/model/map/cls.json
assets/resource/model/map/tile_mapping.json
assets/resource/model/map/cameraorientation/polar_with_ref.onnx
assets/resource/model/map/cameraorientation/preprocess.onnx     # C++ 不使用
assets/resource/model/map/navmesh/…                             # MapNavigator 用，本工具不需要
```

---

## 4. 运行时数据路径（以 submodule 根为基准）

### 4.1 zone 底图 **[直接证据]**

`assets/resource/image/MapLocator/` 下的 zone 子目录（现有：`Dung`, `IndieDg005`,
`IndieDg007`, `OMVBase`, `ValleyIV`, `Wuling`），每个目录含 `Base.png` 与
`Lv<NNN>Tier<NNN>.png`。加载器
`MapLocator.cpp:loadAvailableZones`（`:875-916`）递归扫描：

- 文件名（小写化后）为 `base.png` → key = `<父目录名>_Base`（`MapLocator.cpp:895-897`）
- 匹配 `Lv(\d+)Tier(\d+)\.(png|jpg|webp)` → key = `<父目录名>_L<lv>_<tier>`
  （regex `MapLocator.cpp:890`, key `:900-903`）
- 只读 `.png/.jpg/.jpeg/.webp/.bmp`（`MapLocator.cpp:50`）

因此 `MapLocatorConfig.mapResourceDir` 应指向
`…/assets/resource/image/MapLocator`（含各 zone 子目录的那层）。

官方 action 也是这么拼的（`MapLocateAction.cpp:336`）：
`installDir/resource/image/MapLocator`，
`installDir/resource/model/map/cls.onnx`（`:337`），
`installDir/resource/model/map/cameraorientation/polar_with_ref.onnx`（`:338-339`）。

### 4.2 模型及其 json 的相对路径 **[直接证据]**

`MapLocatorConfig`（`MapTypes.h:24-31`）字段：
`mapResourceDir` / `yoloModelPath` / `cameraOrientationRefModelPath` / `yoloThreads`。

`YoloPredictor` 的 sidecar 解析（`YoloPredictor.cpp`）：
- `yoloModelPath` = `.onnx` 路径（`:28-29`）
- `.json` = 把 `yoloModelPath` 扩展名换成 `.json`（`:32-34`）→ 因此
  `assets/resource/model/map/cls.onnx` 要求同目录 `cls.json`
- `tile_mapping.json` = 该 onnx 所在**目录**下固定名（`:61-63`）

`cls.json` 提供 `input_name="images"`、`output_name="output0"`、`classes[...]`
（`cls.json` 头部 **[直接证据]**），`tile_mapping.json` 提供
`base_class/x/y/w/h/infer_margin`（`TileRegion`，`YoloPredictor.h:33-43`）。

`CameraOrientationPredictor` 只需 `polar_with_ref.onnx`（`MapLocateAction.cpp:338-339`）；
`preprocess.onnx` 是 Python 侧预处理遗留，C++ 在 `BuildOrientationStrips` 里自算
（`CameraOrientationPredictor.cpp:122` 调用，注释 `:21`）。

`CameraOrientationPreprocess.h:8-17` 契约：输入 ROI 必须 `CV_8UC3` 且 118×120，
asset `CV_8UC4`，否则 `BuildOrientationStrips` 返回 false（`.cpp:80-86`）。

---

## 5. NixOS 环境

### 5.1 参考 `~/Projects/MaaEnd/shell.nix` **[直接证据]**

关键片段（`/home/mx/Projects/MaaEnd/shell.nix`）：

- `:37-45` `shellHook` 导出
  `CMAKE_PREFIX_PATH=${toString ./.}/agent/cpp-algo/MaaUtils/MaaDeps/vcpkg/installed/maa-x64-linux`；
  若 `NIX_LD_LIBRARY_PATH` 存在则 prepend 到 `LD_LIBRARY_PATH`（注释：Nix 构建的
  agent 绕过 nix-ld，需要给动态加载器显式暴露库）
- `:54-58` `env.C_INCLUDE_PATH = "${pkgs.linuxHeaders}/include"`；
  `:13-35` packages 里带 `ccache clang-tools cmake ninja`（cpp 工具链）
- 本仓 worktree 的 `shell.nix` 只有 `python312/ruff/uv`，**没有 cmake/ninja/clang**，
  需要自行补（或直接进入 MaaEnd 的 dev shell）。**[直接证据]**

### 5.2 已复现的坑 **[直接证据]**

1. **CMake 会被判成交叉编译**：用 MaaDeps toolchain 时 `CMAKE_CROSSCOMPILING=TRUE`，
   OpenCV 找不到 → 必须 `-DCMAKE_PREFIX_PATH=<vcpkg prefix>`（§2.2）。
2. **编译器 ABI**：MaaDeps 的预编译库面向 **clang + libc++ + 自带 sysroot**
   （`MaaDeps/cmake/maa-x64-linux-toolchain.cmake:4-20`：`-stdlib=libc++ --sysroot=…`,
   `-fuse-ld=lld`）。用 Nix 的系统 g++/libstdc++ 直接链接 `libopencv_world4.so` 会跨
   std C++ 边界（OpenCV 以 `std::string` 传参），**应走 toolchain 里的 clang**。
   本机成功构建即用的该 toolchain（clang 20.1.8，见 §6）。
3. **Boost 版本**：vcpkg 是 `boost-1.91.0`；Nix 的 boost 版本可能不同 →
   走 `find_package(Boost CONFIG)` 并依赖 vcpkg 前缀，别用系统 boost。
4. **ZLIB**：`maadeps.cmake:55-57` 默认 `ZLIB_USE_STATIC_LIBS=ON`，会用
   `libz.a`（本机 configure 输出 `Found ZLIB: …/lib/libz.a … 1.3.2`）。
5. **nix-ld / `LD_LIBRARY_PATH`**：运行时需要 MaaDeps 的 shared libs
   （`libopencv_world4.so.412`、`libonnxruntime.so.1`，在
   `MaaDeps/runtime/maa-x64-linux/`）与自建的 `libMaaUtils.so`。shell.nix 的做法是
   把 `NIX_LD_LIBRARY_PATH` prepend 进 `LD_LIBRARY_PATH`；本机实测直接设
   `LD_LIBRARY_PATH=<build/bin>:<MaaDeps/runtime/maa-x64-linux>` 即可 **[直接证据]**。
6. `C_INCLUDE_PATH=${pkgs.linuxHeaders}/include`：本机构建未显式设置也成功 **[直接证据]**，
   但保留以对齐上游环境（某些内核头/`linux/*.h` 相关编译在别的发行里需要）。
7. `install.yml` CI 用 `python3 tools/maadeps-download.py x64-linux` 引导 MaaDeps
   （`.github/workflows/install.yml:262-267`），可作环境复现参考。

### 5.3 可执行文件产物与搬运

- `MaaUtils.cmake` 的 `config.cmake` 把产物统一放到 `${CMAKE_BINARY_DIR}/bin`
  （`cmake/config.cmake:88-90`），RPATH 为 `$ORIGIN`（`:8`）。
- 要独立分发，需把 `MaaDeps/runtime/maa-x64-linux/*`（opencv/onnxruntime/dawn 等）
  放到可执行文件同目录或设 `LD_LIBRARY_PATH`。`maadeps_install()` 提供了安装规则
  （`maadeps.cmake:59-65`）。

---

## 6. 可直接粘贴的最小 CMake 片段（已实测通过）

### 6.1 实测记录 **[直接证据]**

在 `/tmp/cppbuild` 用下列等价内容，配合
`-DCMAKE_TOOLCHAIN_FILE=…/MaaDeps/cmake/maa-x64-linux-toolchain.cmake`
`-DCMAKE_PREFIX_PATH=…/MaaDeps/vcpkg/installed/maa-x64-linux`，
`cmake -G Ninja` + `cmake --build` 成功：25/25 步，产出 `bin/libMaaUtils.so` 与
`bin/maplocator-tool`。随后运行：

```
LD_LIBRARY_PATH=bin:…/MaaDeps/runtime/maa-x64-linux bin/maplocator-tool \
  assets/resource/image/MapLocator \
  assets/resource/model/map/cls.onnx \
  assets/resource/model/map/cameraorientation/polar_with_ref.onnx
# 输出: 0.9375   (ZoneTemplateScale("ValleyIV_Base") == 15/16)
# exit=0，initialize() 成功
```

`main.cpp` 只调用了 `MapLocator::initialize` / `ZoneTemplateScale` /
`BuildOrientationStrips` 三个符号，验证了目标 API 面。

### 6.2 `cpp/CMakeLists.txt` 建议（目标名可换）

```cmake
cmake_minimum_required(VERSION 3.28)
project(maplocator-data-prep CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- MaaEnd submodule 位置 ---
set(MAAEND_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/MaaEnd")
set(CPP_ALGO_SOURCE "${MAAEND_ROOT}/agent/cpp-algo/source")
set(MAAUTILS "${MAAEND_ROOT}/agent/cpp-algo/MaaUtils")

# 关键：include MaaUtils.cmake 会 include MaaDeps、find_package 依赖、
# 并把 MaaUtils 作为 SHARED 目标以 add_subdirectory 建出来。
include("${MAAUTILS}/MaaUtils.cmake")

set(maplocator_src
    "${CPP_ALGO_SOURCE}/MapLocator/MapLocator.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/MapAlgorithm.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/MatchStrategy.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/MotionTracker.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/YoloPredictor.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/CameraOrientationPredictor.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/CameraOrientationPreprocess.cpp"
)

add_executable(maplocator-data-prep
    src/main.cpp                 # 下一张票实现
    ${maplocator_src}
)

target_include_directories(maplocator-data-prep PRIVATE "${CPP_ALGO_SOURCE}")

# OpenCV_INCLUDE_DIRS / OpenCV_LIBS / ONNXRuntime::ONNXRuntime / Boost::* / ZLIB::ZLIB
# 都已由 MaaUtils.cmake 里的 find_package 提供。
target_link_libraries(maplocator-data-prep PRIVATE
    MaaUtils
    Boost::regex Boost::system
    ONNXRuntime::ONNXRuntime
    ZLIB::ZLIB
    ${OpenCV_LIBS}
)
if(UNIX)
    target_link_libraries(maplocator-data-prep PRIVATE pthread dl)
endif()
add_dependencies(maplocator-data-prep MaaUtils)
```

配置命令（NixOS）：

```bash
nix develop ~/Projects/MaaEnd            # 拿到 cmake/ninja/clang/ccache（shell.nix 已导出 CMAKE_PREFIX_PATH）
cd cpp
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/MaaEnd/agent/cpp-algo/MaaUtils/MaaDeps/cmake/maa-x64-linux-toolchain.cmake" \
  -DCMAKE_PREFIX_PATH="$PWD/MaaEnd/agent/cpp-algo/MaaUtils/MaaDeps/vcpkg/installed/maa-x64-linux"
cmake --build build -j
```

依赖查找顺序说明：`MaaUtils.cmake` 里 `include(maadeps.cmake)` 先把 vcpkg 前缀塞进
`CMAKE_PREFIX_PATH`（或 root path），随后才 `find_package(OpenCV/Boost/ZLIB/fastdeploy_ppocr/ONNXRuntime)`，
所以**只要 include `MaaUtils.cmake`，就不要自己重复 find_package**（顺序和 REQUIRED 语义
都由它保证）。`fastdeploy_ppocr` 虽被 `find_package(... REQUIRED)` 要求，但 MaaUtils 与
本工具都不链接它，只是配置期必须能找到。

---

## 7. 批处理相关 API 缺口（供下一张票参考）

现状 `MapLocator` 公共 API（`MapLocator.h:11-29`）：

- `bool initialize(const MapLocatorConfig&)`
- `bool isInitialized() const`
- `LocateResult locate(const cv::Mat& minimap, const LocateOptions& = {})`
- `YoloCoarseResult predictCoarse(const cv::Mat& minimap) const`
- `void resetTrackingState()`
- `std::optional<MapPosition> getLastKnownPos() const`

观察 **[直接证据]**：

1. **冷启动 burn-in**：首帧没有 `lastPos` 时走全局搜索，需要连续
   `kColdStartConsensusFrames = 3` 帧位置聚集（`MapTypes.h:188`, 逻辑
   `MapLocator.cpp:2100-2112`）才输出成功，否则返回
   `LocateStatus::TrackingLost` + `kColdStartCollectingMessage`
   （`MapTypes.h:193`）。批处理逐图跑时前 2 帧必然“失败”，调用方需要自己处理。
2. **状态重置**：`resetTrackingState()`（实现 `MapLocator.cpp:2129-2141`）会
   `forceLost` + 清速度、清 `currentZoneId`、清 `coldStartBuffer` / `stablePosition` /
   arbiter 状态，但**不重载 zone、不重置 RNG/缓存之外的持久状态**（zone 底图仍在
   `zones`，`zoneGeneration` 不变）。批处理在不同序列间迁移时需要显式调用它，否则
   跨序列的追踪/速度估计会污染。
3. **`force_global_search`**：`LocateOptions::force_global_search`（`MapTypes.h:49`）
   强制放弃追踪走全局全图搜（使用点 `MapLocator.cpp:1430, 1949`）。批处理里对每张
   独立图强制全局搜可绕开冷启动追踪，但仍受第 1 点的冷启动共识门限影响。
4. `locate` 内部会起 `std::async` 做箭头朝向识别（`MapLocator.cpp:2170-2172`），
   并对 YOLO 用异步共享 future（`MapLocator.cpp:1415`）；批量高并发调用同一
   `MapLocator` 实例需注意其内部 `taskMutex` / `asyncYoloState` 的单帧假设
   （**§本轮未验证多线程并发 locate 的行为 [推断]**）。
5. `initialize` 幂等（`MapLocator.cpp:847-850`：已初始化直接返回 true），
   **无法在不破坏对象的前提下更换 model 路径 / 重新加载 zone**——批处理若需换模型
   必须新建实例。

---

## 8. Missing evidence / 未验证项

- 未在**纯 NixOS dev shell（系统 clang + nix boost/opencv）**下验证；本机成功路径用的是
  MaaDeps 自带 clang/libc++ toolchain 与 vcpkg 依赖（§5.2 第 2 条）。用系统
  g++/clang 是否可链接 vcpkg 的 libc++ 库 **未验证**。
- 未实际跑 `locate()` 端到端（缺 720p 测试帧）；只验证了 `initialize` 成功、
  zone 加载与 `ZoneTemplateScale` / `BuildOrientationStrips` 编译链接。
- 未验证多线程并发调 `locate()` 的正确性（§7 第 4 条）。
- 嵌套 submodule 的 `git submodule update --init --recursive` **未实际执行**（本机用的是
  `/home/mx/Projects/...` 现成副本），URL/commit 已核对一致。
- `maadeps-download.py` 联网下载 **未实际执行**（离线副本已具备）。
- 未验证把 MaaDeps `runtime/` 拷到 exe 同目录后无 `LD_LIBRARY_PATH` 能否启动
  （理论可行，RPATH=`$ORIGIN`）。

---

## 9. Start Here（下一张票先看哪里）

1. `cpp/MaaEnd/agent/cpp-algo/source/MapLocator/MapLocator.h` + `MapTypes.h`
   —— 公共 API 与 `MapLocatorConfig` / `LocateOptions` / `LocateResult`。
2. `cpp/MaaEnd/agent/cpp-algo/MaaUtils/MaaUtils.cmake` + `MaaDeps/maadeps.cmake`
   —— 依赖接线与 CMake 片段来源。
3. 按 §6.2 落 `cpp/CMakeLists.txt`，`main.cpp` 按 §6.1 的调用面实现数据准备逻辑。
