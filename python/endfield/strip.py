"""训练与交付共用的条带几何：参考配对极坐标展开的尺寸与通道布局。

条带由 MaaEnd C++ 的 `CameraOrientationPreprocess`（`BuildOrientationStrips`）产出：
行 = 半径（内径 12 px 到外径 54 px 均分 42 行），列 = 方位角（北为 0、顺时针 1 度/列）。
本模块只持有形状契约，不持有采样语义。
"""

STRIP_H = 42
STRIP_W = 360

OBSERVED_CHANNELS = 3
REFERENCE_CHANNELS = 4
# 参考配对 [obs.BGR, ref.BGR, ref.A] 的通道数，也是 AzimuthNet 首层通道数
INPUT_CHANNELS = OBSERVED_CHANNELS + REFERENCE_CHANNELS
