#pragma once

// 独立构建用的替身：cpp/src/ 与 MaaEnd 逐字节相同，引用的是 MaaEnd 的 <MaaUtils/NoWarningCV.hpp>
// （屏蔽 OpenCV 头文件告警后引入 OpenCV）。本实现只用到 cv::Mat。

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <opencv2/core.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
