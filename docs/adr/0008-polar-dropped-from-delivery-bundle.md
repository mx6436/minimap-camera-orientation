# polar 分类器退出交付：交付集合收敛为 preprocess + polar_with_ref

`ref` 分类器在几乎所有场景的精度都优于 `polar`（`ref_004` val rms 3.45° vs `polar_003` 4.34°），并列交付只会让 MaaEnd 多持一份权重、多一次推理，并在两者之间做选路。故本仓当前交付的图集合收敛为 `preprocess` + `polar_with_ref`，单点收在 `endfield/bundle.py` 的 `DELIVERED_ROLES`：`export-artifact` 只接 `--ref-run`，交付集合外的 run 在 `build_manifest` 即被拒，`check_structure` 按当前集合要求 manifest 声明齐全。交付角色词汇与图文件名（跨仓契约）保留——`polar` 仍是可训练的输入模式，`polar.onnx` 仍可由 `export-onnx` 单独导出，`--require polar` 仍能指名旧 bundle 里的那张图。

## Considered Options

- **继续并列交付 `polar` 作轻量后备**：它没有任何场景更优，后备价值不成立，代价是 MaaEnd 侧的权重体积、一次多余推理与选路逻辑。
- **从 `DeliveryRole` 删除 `POLAR`**：`export-onnx` 按输入模式推默认交付文件名，删掉角色会让 `polar` 模式失去文件名；旧 bundle 的结构自检与 conformance 的 `polar` 数值比对路径也会一并失效。
- **把交付集合做成 `export-artifact` 的一次性参数**（给哪几个 run 就交哪几图）：交付契约变成每次调用时的选择，`check_structure` 无从要求 manifest 声明齐全，交付范围也就不再是一份可复述的决定。

## Consequences

- 跨仓契约变化：MaaEnd 交付布局不再拷 `polar.onnx`，README「拷入 MaaEnd」同步为两图。
- 旧 bundle（如 `runs/bundle_003`）按新集合自检会报 `graphs_roles`：它不再是当前交付契约的形状。
- `polar` 模式的训练、`export-onnx` 导出与 conformance 的 `--require polar` 数值比对路径都保留，用于后续比较。
