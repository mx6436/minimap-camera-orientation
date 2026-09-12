# ORT 1.19.2 运行 `preprocess.onnx`：算子支持、动态尺寸与外部内存取证（issue #22）

## 取证口径与限制（先读）

- 本次运行的工具集中 **没有注册 `web_search` / `source_check`**；网络取址由父会话用 `curl` 预取到 `/tmp/ort-research/`。本文件引用的官方 URL 前缀统一为
  `https://raw.githubusercontent.com/microsoft/onnxruntime/v1.19.2/`，内容即预取到本地的同名文件（`OperatorKernels.md`、`grid_sample.cc`、`grid_sample.h`）。
- 行号为**本次读取窗口内的近似位置**（我用 `read` 的 offset/limit 逐窗读取，无法 grep）。每条引用都附了可复核的唯一字符串，评审时按字符串检索即可定位。
- 结论标注三类：**【直接证据】**（源文件原文/表格单元）、**【源码解读】**（对已读源码语义的推断，附具体代码）、**【研究者推断】**（无直接出处，仅工程判断）。凡未在本次运行中核实的，都写进「风险与未知项」。
- 本地一手来源全部为可读文本：MaaDeps 内 ORT 1.19.2 头文件、训练 venv 内 `onnx` / `onnxscript` / `torch` 源码、`export_onnx.py`、`endfield/polar.py`、`endfield/ref.py`、MaaEnd 的 `CameraOrientationPredictor.cpp`。

## 结论速览

| # | 问题 | 结论 | 关键条件 |
|---|---|---|---|
| 1 | 15 个算子在 ORT 1.19.2 | **除 `GridSample` 外全部支持**；`GridSample` 支持但**仅 float32**（opset 16–19） | 见下表逐条 |
| 2 | 动态尺寸输入 | **支持**：kernel 运行期读 shape，任意 spatial size；但导出必须产出动态维，且 `GridSample` 只吃 4D/5D、grid 秩=数据秩+2' | 无 free-dim override API（本地头文件窗口内未见） |
| 3 | uint8 输入张量 | 模型**图输入可以是 uint8**（现网模型已如此）；但 `GridSample` **不能直接吃 uint8**，必须图内 `Cast` 到 float32；且 `Equal/Less/Add/Mul/Sub/Div/ReduceMean/Round` 都不支持 uint8 | 算术/比较一律在 float32 域做，末尾 `Round`（半偶）+ `Cast` 回 uint8 |
| 4 | C++ `CreateTensor` 外部内存 | **零拷贝、调用方持有内存**；无 stride 参数 ⇒ 非连续行跨距**不可表达**；同一 buffer 顺序复用安全，并发 Run 不安全 | 保持现有 `predictMutex` 串行化；shape 必须与模型声明一致 |
| 5 | torch 导出与 ORT 实现状态 | `GridSample` 是**标准 ai.onnx 算子**（非 contrib）；ORT 1.19.2 有 CPU 实现。opset 18 必须用 **16 形态**（`mode="bilinear"`），写 `"linear"` 会被 ORT 直接抛错 | torch dynamo 导出路径的具体 lowering **未核实** |

### 逐算子支持表（ORT 1.19.2 CPU EP，模型 opset = 18）

来源：`[ORT-KERNELS]` = `docs/OperatorKernels.md`（CPUExecutionProvider 表，按算子名排序；引号内为表格单元原文）。

| 算子 | opset 18 下适用版本 | ORT 1.19.2 CPU 类型支持 | 结论 |
|---|---|---|---|
| `GridSample` | `[16, 19]` | `T1` = tensor(float)<br>`T2` = tensor(float) | **支持，仅 float32**（`20+` 才加 double） |
| `Cast` | `[13, 18]` | 全类型，含 uint8/int8/string | 支持（uint8→float32 的通道） |
| `Concat` | `13+` | 全类型，含 uint8 | 支持 |
| `Slice` | `13+` | 全类型含 uint8；`Tind` = int32/int64 | 支持 |
| `Gather` | `13+` | 全类型含 uint8；`Tind` = int32/int64 | 支持 |
| `Where` | `16+` | double, float, int32, int64, **string, uint8** | 支持；**注意没有 int8/uint16** |
| `Resize` | `[13, 17]` / `18` / `19+` | `T1` = float, int32, int8, uint8 | 支持（整型插值语义需另验，见风险 5） |
| `Round` | `11+` | double, float, float16 | 支持；**整型不支持**，必须 float 域舍入后再 Cast |
| `Reciprocal` | `13+` | double, float | 支持 |
| `Add` / `Sub` / `Div` / `Mul` | `13`(Add 为 `14+`) | double, float, int32, int64 | 支持；**uint8 不支持** |
| `ReduceMean` | `18+` | double, float, int32 | 支持 |
| `Equal` | `[13, 18]` | bool, double, float, int32, int64 | 支持；**uint8 不支持** |
| `Less` | `[13, 18]` | double, float, int32, int64 | 支持；**uint8 不支持** |
| `Pad`（现有导出用 `Pad(wrap)`） | `[13, 17]` / `18` | bool, double, float, int32, int64, int8, uint32, uint64, uint8 | 支持 |

**一句话口径**：这 15 个算子全部在 ORT 1.19.2 中可用；真正的限制不在「有没有 kernel」，而在 **`GridSample` 的 dtype（只有 float32）+ 属性字符串（opset 18 下 `mode` 必须是 `bilinear`）+ 非 float 算子的 dtype 白名单**。

---

## Q1 逐算子支持（出处）

### 1.1 `GridSample`：标准算子，opset 16–19 仅 float

- **【直接证据】** CPU 表原文：`|GridSample|*in* X:**T1**<br> *in* grid:**T2**<br> *out* Y:**T1**|[16, 19]|**T1** = tensor(float)<br/> **T2** = tensor(float)|`，另一行 `|20+|**T1** = tensor(double), tensor(float)<br/> **T2** = tensor(double), tensor(float)|`。
  → 模型 opset=18 ⇒ 落到 `GridSample-16`，**X 与 grid 都必须是 float32**。`[ORT-KERNELS]`
- **【直接证据】** kernel 注册（`[ORT-GS-CC]` 文件头部宏）：
  - `ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(GridSample, kOnnxDomain, 16, 19, T, kCpuExecutionProvider, KernelDefBuilder().TypeConstraint("T1", ...T).TypeConstraint("T2", ...T), GridSample<T>)`，随后 `REGISTER_KERNEL_TYPED(float)`；
  - `ONNX_OPERATOR_TYPED_KERNEL_EX(GridSample, kOnnxDomain, 20, T, ...)`，随后 `REGISTER_KERNEL_TYPED_20(float)`、`REGISTER_KERNEL_TYPED_20(double)`。
  → 官方源码与 Kernel 表一致：**opset 16–19 只有 float；opset 20 才加 double**；且 `T1` 与 `T2` 被约束为**同一类型**（所以 float X 必须配 float grid）。
- **【直接证据】** 属性字符串按 `info.node().SinceVersion()` 分支（`[ORT-GS-H]` 构造函数）：
  - `start_version >= 20`：接受 `linear` / `nearest` / `cubic`，否则抛 `mode "..." not supported, expect linear, nearest or cubic`；
  - 否则（含 opset 18）：接受 `bilinear` / `nearest` / `bicubic`，否则抛 `mode "..." not supported, expect bilinear, nearest or bicubic`。
  - `padding_mode` 三个值 `zeros` / `border` / `reflection` 都实现了；不认识的抛错；`align_corners` 读 int（默认 0）。
  → **opset 18 图里写成 `mode="linear"` 会让 1.19.2 直接拒绝加载**。这是本次取证中最容易踩的坑。
- **【直接证据】** 三种插值 + 三种 padding 的实际实现都在 `Compute`/`PixelAtGrid` 中：`mode_==Linear` 走 `p11..p22` 四邻点加权；`Zeros` 越界返回 0，`Border` 用 `std::clamp` 夹索引，`Reflection` 用 `GsReflect`。
- **【源码解读】** `align_corners` 归一化语义（`[ORT-GS-CC]` 注释 + `GsDenormalize`）：
  - `true`：`x = (n + 1) / 2 * (length - 1)`（(-1,-1)→左上像素中心）；
  - `false`（默认）：`x = ((n + 1) * length - 1) / 2`（(-1,-1)→图像空间 (-0.5,-0.5)，即像素中心外半像素）。
  与 ONNX 参考实现完全一致（见 Q5）。

### 1.2 非 float 算子的 dtype 白名单（易被忽略）

- **【直接证据】**（均出自 `[ORT-KERNELS]` CPU 表对应行）
  - `Add` / `Sub` / `Div` / `Mul`：`T` = double, float, int32, **int64** —— **没有 uint8/int8/uint16**。
  - `Equal`（`[13, 18]`）：bool, double, float, int32, int64 —— 没有 uint8；`Less`（`[13, 18]`）：double, float, int32, int64 —— 没有 uint8。
  - `ReduceMean`（`18+`）：double, float, int32。
  - `Round`（`11+`）：double, float, float16 —— **整型不能 Round**。
  - `Where`（`16+`）：double, float, int32, int64, string, **uint8**（uint8 可以，但 int8/uint16 不行，且 X/Y 必须同型）。
  - `Resize`：`T1` = float, int32, int8, uint8（uint8 在列，但线性插值在整型上的具体语义见风险 5）。
- **【研究者推断】** 因此图内预处理必须是：`uint8 → Cast → 全部 float32 计算 → Round（半偶）→ Cast → uint8`；任何「顺手对 uint8 做 Add/Less/Equal」都会在 ORT 建会话/校验期失败，而不是静默降级。

### 1.3 `Round` 的舍入语义与 C++ 侧对齐

- **【直接证据】** ONNX 参考实现 `onnx/reference/ops/op_round.py`：`return (np.round(x).astype(x.dtype),)` —— `np.round` 是**半偶（banker's rounding）**。
- **【源码解读】** MaaEnd 现有 C++ 用 `std::nearbyint`（默认 to-nearest-even）+ `cv::saturate_cast`（注释明确写了「Python round() 与 OpenCV saturate_cast 均为半偶舍入」），与 ONNX `Round` 语义一致 ⇒ 图内 `Round` + `Cast` 回 uint8 在**舍入规则**层面与 C++ 对齐（但浮点求值路径不同，见 Q5 数值风险）。

---

## Q2 动态尺寸输入的限制

- **【直接证据】** `GridSample::Compute` 运行期从张量取维度：`H_in = input_dims[2]`、`W_in = input_dims[3]`、`H_out = grid_dims[1]`、`W_out = grid_dims[2]`，输出 shape 现算 `{N, C, H_out, W_out}`；唯一的形状约束是
  - `grid` 秩 = 数据秩 + 2，`grid` 最后一维 == 数据秩（2 表示 2D 采样），
  - `grid_dims[0] == N`，
  - 输入秩只能是 4 或 5（`Only 4-D or 5-D tensor is supported`），5-D 不允许 `Cubic`。
  → **spatial size 任意（含 1440x1350、2016x2976），无静态尺寸要求**；只要 grid 是 float32 且 shape 自洽。`[ORT-GS-CC]`
- **【直接证据】** ONNX 规范侧 opset 16 的 `GridSample` 文档限制在 4-D：「Currently, only spatial (4-D) inputs are supported … output `Y` will have shape (N, C, H_out, W_out)」（`onnxscript/onnx_opset/_impl/opset16.py` 中 `GridSample` docstring；`mode` 默认值是 `"bilinear"`，`padding_mode` 默认 `"zeros"`, `align_corners` 默认 `0`）。
  → 我们的 2D 极坐标采样正是 4-D，grid 形状 `(1, 42, 360, 2)`。
- **【源码解读】** 会话侧：`Ort::Value::CreateTensor(info, p_data, count, shape, shape_len)` 要求**每次 Run 传具体 shape**；模型若把 H/W 导出成固定常量，run 时尺寸不同会因维度不符而报错。`[ORT-CXXINLINE]` + `[ORT-CAPI]` 的 `Run` 文档（"Will not return until the model run has completed"）。
- **【直接证据/否定证据】** 本地 1.19.2 头文件中，`OrtApi` 的 SessionOptions 段（我读过的约 700–1150 行窗口：`CreateSessionOptions` … `SetIntraOpNumThreads` / `SetInterOpNumThreads` / `AddCustomOpDomain` / `RegisterCustomOpsLibrary`）与 `Ort::detail::SessionOptionsImpl` 方法列表（`onnxruntime_cxx_api.h` 约 840–905 行窗口）中**都没有 `AddFreeDimensionOverride*`**。我未逐行核对整份 4753 行文件，但据这两处窗口，**不要依赖「按名覆盖自由维度」**；应直接导出动态维（`dim_param`）。
- **【研究者推断/导出侧风险】** `export_onnx.py` 用 `dummy = torch.zeros(1, POLAR_H, POLAR_W, channels, dtype=torch.uint8)` 走 `torch.onnx.export(..., opset_version=18)`；若不给 `dynamic_shapes`，dynamo 导出器会把这一批维度固化为静态常量。要支持「底图 H/W 随 zone 变化」，必须显式声明动态维（`dynamic_shapes` / `Dim`），或改成把尺寸作为输入张量传入并全程用 `Shape/Gather/Slice` 派生 —— 否则 1440x1350 与 2016x2976 两次运行至少有一次直接失败。**该结论基于 torch 导出语义，未在本次运行中实测导出物。**

---

## Q3 uint8 输入张量的接受路径

- **【直接证据】** 现网 MaaEnd 模型就是 uint8 输入：`CameraOrientationPredictor.cpp` 用
  `Ort::Value::CreateTensor<std::uint8_t>(memoryInfo, inputMat->data, 42*360*C, inputShape, 4)`（HWC uint8），并用 `Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)`，会话正常加载/推理。
  → ORT 1.19.2 接受 uint8 输入张量，且 uint8 外部缓冲区零拷贝路径在项目里已验证可用。`[CPP]`
- **【直接证据/限制】** 但 `GridSample` 不能直接吃 uint8：CPU kernel 只注册 float（opset 16–19），Kernel 表也写 `T1 = tensor(float)`。`[ORT-GS-CC]` `[ORT-KERNELS]`
- **【直接证据】** ONNX **schema** 层面 `GridSample` 的 `T1` 名义上包含 uint8（`onnxscript/.../opset16.py` 的 `T1_GridSample = TypeVar(..., UINT8, UINT16, ..., STRING, COMPLEX64 ...)`，opset 20 同）。
  → **规范允许 ≠ 运行时支持**：类型检查通过不了 ORT 的 kernel 类型约束，图内必须显式 `Cast`。这是一条值得写进 issue 的「规范与实现差异」。
- **【研究者推断】** 推荐图结构：`grid` 常量/计算（float32）+ `Cast(X: uint8 → float32)` → `GridSample(mode="bilinear", padding_mode="border", align_corners=0)` → `Round`（半偶）→ `Cast(→ uint8)`。采样后的 `Round/Cast` 是否真需要，取决于下游是否仍在 uint8 域（现有模型输入契约是 uint8 NHWC）。

---

## Q4 C++ `Ort::Value::CreateTensor` 外部内存：注意事项与限制

1. **零拷贝、内存归调用方**——**【直接证据】** `onnxruntime_c_api.h`（约 1340 行窗口）`CreateTensorWithDataAsOrtValue` 文档：
   > "Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
   > p_data is owned by caller. ReleaseValue won't release p_data."
   → 不复制、不发生所有权转移；`Ort::Value` 只是「张量视图」。现有代码每次 `predict` 都新建一个 `Ort::Value` 包同一块 scratch（构造开销极小、无数据拷贝）。
2. **元素数 → 字节数**——**【直接证据】** `onnxruntime_cxx_inline.h`（约 1630 行窗口）：
   `Value::CreateTensor(info, p_data, p_data_element_count, shape, shape_len)` → `CreateTensor(info, p_data, p_data_element_count * sizeof(T), ..., TypeToTensorType<T>::type)` → `CreateTensorWithDataAsOrtValue`。
   → 传错元素数/类型会得到「张量元素数与 shape 不匹配」或静默错读。
3. **不支持非连续行跨距（stride）**——**【直接证据 + 源码解读】** 该 API 只接受 `shape`（没有 strides 参数），ORT 的张量模型即「dense row-major」。因此把 `cv::Mat` ROI/带 padding 的子矩阵（`step[0] != cols*elemSize`）直接喂进去会**按紧凑布局解释内存**，得到错位数据且不报错。
   现有实现的应对方式即为此：注释写明「展开输出总是连续内存」，且所有 scratch 都由 `cv::Mat::create` 分配成连续块。**建议在新代码里显式 `CV_Assert(mat.isContinuous())`（或 `clone()`）后再建张量。**
4. **生命周期**——**【源码解读】** 因为零拷贝，缓冲区必须在 `Ort::Value` 存活期间、且至少到本次 `Run` 返回之前保持有效；同步 `Run` 返回后即可改写缓冲区内容用于下一次推理。用 `RunAsync` / `IOBinding` 时必须保证 run 完成前不被释放或改写（`[ORT-CAPI]` Run 文档：调用会阻塞到本次 run 结束；`RunAsync` 则回调前都可能在读）。
5. **每次 Run 复用同一 buffer 是否安全**——**【源码解读】**
   - 顺序复用（本项目的 `predict` 形态）：**安全**。`Run` 是同步的，返回后再覆写 scratch 不存在竞争。
   - 并发复用同一 buffer/`Ort::Value`：**不安全**。ORT 的同会话并发 `Run` 要求各自独立的输入张量；两块线程共享一块输入内存必然数据竞争。现有代码用 `std::lock_guard<std::mutex> predictMutex` 串行化 `predict()`，保持这个不变即可；若将来要并行，则「每线程一块 scratch + 每线程一个 `Ort::Value`」或改用 `IOBinding`。
6. **`MemoryInfo` 的选择**——**【源码解读 + 直接证据】** `MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)` 只描述「这块内存在 CPU 上、按默认存储」；外部缓冲区不会被 arena 接管也不会被释放。`MemoryInfoImpl` 只暴露 allocator name / device id / device type / mem type（`onnxruntime_cxx_api.h` 该段），没有「必须与分配方式一致」的校验文档。现有 uint8 路径用的就是 arena 名 + `cv::Mat` 内存，工作正常。
7. **对齐要求**——**未见头文件声明**（未核实 ORT 内部是否有 SIMD 对齐假设）。列入风险清单，不作为结论。
8. **shape 一致性**——**【源码解读】** 传入 shape 与模型声明不符时在 `Run` 处失败；动态维模型每次传实际 H/W 即可（Q2）。

---

## Q5 torch 导出语义与 ORT 侧实现状态

### 5.1 标准算子、非 contrib

- **【直接证据】** ORT 1.19.2 在 CPU EP 注册的是 `kOnnxDomain`（`ai.onnx`）的 `GridSample`，opset `[16,19]` / `20`（`[ORT-GS-CC]` 的两个 `..._KERNEL_EX(GridSample, kOnnxDomain, ...)` 宏；`[ORT-KERNELS]` 的 `**Operator Domain:** *ai.onnx*` 段内 GridSample 行）。
  → **不需要 `com.microsoft` contrib 版本**，也不涉及自定义 op 注册。（legacy contrib `com.microsoft.GridSample` 在 1.19.2 是否仍注册，本次未核对——我们用不到。）

### 5.2 opset 18 的形态与属性拼写

- **【直接证据】** `GridSample-16` 的属性取值是 `bilinear`/`nearest`/`bicubic`；`GridSample-20` 改名为 `linear`/`cubic`（`onnxscript/.../opset16.py` 与 `.../opset20.py` 的 `def GridSample(..., mode: str = "bilinear")` vs `mode: str = "linear"`；ORT 亦按模型 opset 分支校验，见 1.1）。
  → 导出 `opset_version=18` 时**必须**得到 `mode="bilinear"`；任何 `mode="linear"` 的产物在 ORT 1.19.2 上会以 `mode "linear" not supported, expect bilinear, nearest or bicubic` 失败。**这是一条可直接做成回归断言的检查。**

### 5.3 坐标约定：PyTorch / ONNX 参考 / ORT 三方一致

- **【直接证据】** `onnx/reference/ops/op_grid_sample.py` 的 `_gs_denormalize` 顶部直接引 PyTorch 源码链接
  `# https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/GridSampler.h#L26`，并给出：
  - `align_corners=True`：`x = (n + 1) / 2.0 * (length - 1)`
  - `align_corners=False`：`x = ((n + 1) * length - 1) / 2.0`
  与 ORT 的 `GsDenormalize`（同一公式，`(n + 1) / 2.f * (length - 1)` 与 `((n + 1) * length - 1) / 2.f`）逐字一致。
  → 三方语义一致；若要在图内复现 C++ 的像素坐标 `x_px`，用 `align_corners=0` 时反解为 **`n = (2 * x_px + 1) / W - 1`**；用 `align_corners=1` 则是 `n = 2 * x_px / (W - 1) - 1`。两种都能表达同一像素坐标，但浮点路径不同（见 5.4）。

### 5.4 数值差异：数学等价、不保证逐位一致

- **【直接证据】** ORT 线性插值表达式（`[ORT-GS-CC]`，`mode_ == Linear` 分支）：
  `dx2 = x2 - x`、`dx1 = x - x1`、`dy2 = y2 - y`、`dy1 = y - y1`，
  `*Y = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (dx2 * p21 + dx1 * p22)`。
  而 MaaEnd 的 `RemapStrip`（`[CPP]`）是显式权重式：`w00=(1-fx)*(1-fy)`（等四个权重）后四项乘加。
- **【源码解读】** 两者数学等价，但**乘加分组、乘法次数、常数（`x2 - x` vs `1 - fx`）都不同**，编译器还可能做 FMA 合并 ⇒ uint8 结果可能出现 **±1 LSB** 级差异。
  → 若 #22 要求「与 C++ 逐点对齐（bit-exact）」，必须用真实数据做逐点比对；除非 ORT 的实现恰好逐位一致，否则契约要放宽为「±1 LSB 或 K% 像素」，或者把极坐标采样保留在 C++ 侧、图内只做卷积。**这是本次最重要的未决风险。**
- **【源码解读】padding 对齐**：`RemapStrip` 对越界索引做边界复制（`x0 = clamp(floor(x), 0, maxX)`、`x1 = min(x0+1, maxX)`）。ORT 的 `border` 模式夹的是四个邻居索引（`PixelAtGrid` 里 `std::clamp`），在 `x ∈ [0, maxX]` 与 `x ∈ (maxX, maxX+1)` 两个区间与 `RemapStrip` 结果一致；但在 `x ∈ [-1, 0)` 处 ORT 会退化为 `p[0]`，而 `RemapStrip` 会做 `p[0]/p[1]` 混合——**两者的负坐标行为不等价**。
  不过按现有几何校验（`endfield/polar.py`：`r_out + 1 <= cx <= width - r_out - 1`；`CameraOrientationPredictor::prepareRingGeometry` 同条件），有效几何下所有采样坐标都 ≥ 0，所以差异只在 `x` 恰好落在 `[maxX, maxX+1)`（外圈端点）时出现：此时 `padding_mode="zeros"` 会把越界邻居当 0（与 C++ 复制不同），**`border` 才与 C++ 一致**。→ 建议固定 `padding_mode="border"`，不要用 `zeros`。

### 5.5 torch 侧 lowering：未核实

- **【直接证据/否定证据】** 本环境 torch（`torch/onnx/_internal/exporter/_torchlib/ops/__init__.py` 的 `__all__ = ["core", "hop", "nn", "symbolic", "symops"]`）中，我完整读过 `ops/nn.py`（372 行）与 `ops/core.py`，**均没有 `grid_sample` / `GridSample` 的 `onnx_impl` 注册**；`torch/onnx/symbolic_opset16.py` 已是「Backward compatibility module」shim。
  → 该 torch 版本的 dynamo 导出器把 `aten.grid_sampler` 走到哪条路径（`_torchlib` 之外的分解 / onnxscript 侧实现）**本次未能定位**，属未核实项。
- **【研究者推断】** 无论走哪条路径，产物最终要么是 `GridSample-16` 节点，要么是等价图（例如 5-D 情形被拆成基本算子）。**建议在 CI 里对导出的 `.onnx` 做结构断言**：存在 `GridSample` 节点、`mode="bilinear"`、`padding_mode="border"`、`align_corners=0`；不存在 `com.microsoft` 域节点。

### 5.6 版本落差（必须显式对待）

- **【直接证据】** 训练 venv 里装的是 **onnxruntime 1.30.0**（`site-packages/onnxruntime/__init__.py` 顶部 `__version__ = "1.30.0"`），而交付目标是 C++ 侧 **1.19.2**（`ORT_API_VERSION 19` 见 `onnxruntime_c_api.h` 文件头部；MaaDeps 安装路径见「来源」）。
  → 任何「用 Python ORT 跑通就算过」的验收都不足以证明 1.19.2 可用（类型白名单、`mode` 名称校验、kernel 版本范围都可能在两个版本间不同）。**建议验收用 `pip install onnxruntime==1.19.2` 或直接跑 MaaEnd 的 C++ 会话。**

---

## 风险与未知项清单

1. **数值逐位一致性未知（最高优先级）**：ORT `GridSample` 的浮点求值分组与 `RemapStrip` 不同，±1 LSB 差异需实测（Q5.4）。
2. **torch dynamo 导出路径未核实**：该 torch 版本 `_torchlib` 中找不到 grid_sample lowering；导出物是否直接就是 `GridSample-16` 需实测（Q5.5）。
3. **静态 shape 陷阱**：`export_onnx.py` 若不声明动态维，导出物会把 H/W 固化为常量，多分辨率 zone 直接失败（Q2）。
4. **dtype 白名单**：`GridSample` 仅 float32；`Add/Sub/Div/Mul/Equal/Less/ReduceMean/Round` 不支持 uint8；`Where` 不支持 int8/uint16。图内必须统一 Cast 到 float32（Q1.2）。
5. **`Resize` 对整型输入的线性插值语义未核实**：参考裁剪/合成若要在图内复现 `cv::resize(INTER_LINEAR)`（半像素中心、`scale = out/in`），建议 `Cast→float32→Resize(mode="linear", coordinate_transformation_mode="half_pixel")→Round(半偶)→Cast`，不要直接对 uint8 做 linear Resize。**ONNX `half_pixel` 与 cv::resize 的等价性为研究者推断，未实测。**
6. **内存高水位**：大 zone 底图（如 2016×2976）一次性 `Cast` 到 float32 会多出约 4 倍于 uint8 张量的中间缓冲（多通道时可达几十至上百 MB），加上 `Resize`/`Concat` 的中间量，重复交替两种 zone 尺寸时可能反复分配；**未测量**。建议压测峰值 RSS，或考虑「C++ 侧裁剪 + 图内只做极坐标采样」的分工。
7. **无 free-dimension override API**（本地 1.19.2 头文件窗口内未见 `AddFreeDimensionOverride*`）：不能靠会话配置补救静态 shape，必须导出动态维。未逐行核对整份头文件。
8. **外部内存对齐要求未声明**：未核实 ORT 对未对齐 host buffer 是否有 SIMD 假设。
9. **legacy `com.microsoft.GridSample` 注册状态未核对**（我们用不到，仅记录）。
10. **CUDA/DML 的 GridSample 行未读取**（本项目走 CPU EP，不影响结论）。

## 建议的最小验证清单（把未知项收敛）

1. `pip install onnxruntime==1.19.2` 到临时 venv → 用候选图跑 `InferenceSession`，对同一输入比对「ORT 输出 / numpy float32 参考 / C++ `RemapStrip` 输出」，记录 uint8 最大绝对差与差异像素占比。
2. 导出候选图（opset 18，含**动态 H/W**）→ `onnx.load` 后断言：`GridSample` 属性 `mode="bilinear"`、`padding_mode="border"`、`align_corners=0`；输入 dim 为符号维；无 `com.microsoft` 节点。
3. C++ 侧对同一 session 连续 Run 两种底图尺寸（1440x1350、2016x2976），观察：会话是否可复用、每次 Run 的耗时/RSS 峰值、`Ort::Session::GetInputTypeInfo` 打印出的 dim_param。
4. 保留现有 `predictMutex`；对 scratch 加 `CV_Assert(isContinuous())`；确认无并发共享输入 buffer 的新路径。

## 来源

**Kept（官方一手，父会话预取）**

- `docs/OperatorKernels.md`（ORT v1.19.2）— `https://raw.githubusercontent.com/microsoft/onnxruntime/v1.19.2/docs/OperatorKernels.md`：CPU EP 逐算子 opset/类型支持表，本文件 Q1 全部类型结论的出处。本地副本 `/tmp/ort-research/OperatorKernels.md`。
- `onnxruntime/core/providers/cpu/tensor/grid_sample.cc`（v1.19.2）— 同名 raw URL：kernel 注册（opset/类型约束）、`GsDenormalize` 注释与公式、`Compute` 的秩/形状校验与线性插值表达式。本地 `/tmp/ort-research/grid_sample.cc`。
- `onnxruntime/core/providers/cpu/tensor/grid_sample.h`（v1.19.2）— 同名 raw URL：`mode`/`padding_mode`/`align_corners` 属性解析与枚举。本地 `/tmp/ort-research/grid_sample.h`。

**Kept（本地一手）**

- `onnxruntime_c_api.h` / `onnxruntime_cxx_api.h` / `onnxruntime_cxx_inline.h`（MaaDeps 内 ORT 1.19.2 头文件，路径见下）— `ORT_API_VERSION 19`、`CreateTensorWithDataAsOrtValue` 的零拷贝/所有权说明、`Run` 语义、`Value::CreateTensor` 实现、SessionOptions 可用方法集合。
  `.../MaaDeps/vcpkg/installed/maa-x64-linux/include/onnxruntime/`
- `CameraOrientationPredictor.cpp`（MaaEnd，`agent/cpp-algo/source/MapLocator/`）— 现有 uint8 外部内存张量用法、`predictMutex`、`RemapStrip` 精确采样公式与半偶舍入契约。
- `export_onnx.py`（本仓库）— opset 18 + dynamo 导出与固定 dummy 输入（动态 shape 风险来源）。
- `endfield/polar.py`、`endfield/ref.py`（本仓库）— 采样几何、参考裁剪/合成与舍入契约（`np.round`、`clip`、`cv2.resize(INTER_LINEAR)`）。
- `onnxscript/onnx_opset/_impl/opset16.py`、`opset20.py`（venv）— ONNX `GridSample-16/20` 规范文本、属性默认值与 `T1/T2` 类型集合（uint8 在列）。
- `onnx/reference/ops/op_grid_sample.py`、`op_round.py`（venv）— 坐标反归一化公式（引 PyTorch `GridSampler.h`）与 `Round` 半偶语义。
- `torch/onnx/_internal/exporter/_torchlib/ops/*`（venv）— 用于**否定性**证据：该 torch 版本的 torchlib 未见 grid_sample lowering。

**Rejected/deprioritized**

- `onnx/docs/Operators.md`、`onnxruntime/capi/version_info.py` 等猜想路径：不存在（读取返回 ENOENT），未作为来源。
- 未采用任何二手博客/问答内容（本 run 无网络检索能力，也不依赖二手材料）。

## 矛盾与缺失记录

- **矛盾 1（规范 vs 实现）**：ONNX `GridSample-16/20` 的 `T1` 名义包含 uint8 等全部类型（`onnxscript` 生成文件），而 ORT 1.19.2 的 CPU kernel 把 `T1/T2` 都约束为 float（opset 16–19）。→ 以**实现**为准：图内 Cast。
- **矛盾 2（规范 vs 实现）**：ONNX opset-16 文档写「only spatial (4-D) inputs」，ORT 的 `Compute` 同时实现了 5-D（`data_dims == 3`）分支。→ ORT 是超集，不影响我们的 4-D 用例。
- **缺失证据**：torch dynamo 对 `grid_sample` 的 lowering；`Resize` 整型线性插值语义；ORT 内部对齐假设；内存峰值实测；`com.microsoft.GridSample` 在 1.19.2 的注册状态；整份 `onnxruntime_c_api.h`（4753 行）未逐行读完，`AddFreeDimensionOverride*` 的缺席结论仅覆盖已读窗口。
