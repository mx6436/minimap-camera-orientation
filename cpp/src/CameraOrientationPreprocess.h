#pragma once

#include <string_view>

#include <MaaUtils/NoWarningCV.hpp>

namespace maplocator
{

// map/cameraorientation/preprocess.onnx 的 C++ 等价实现。
//
// 定义是 mx6436/minimap-camera-orientation 的 endfield/preprocess.py；图里没有可学习参数（两次双线性
// 采样、窗口裁剪、白底合成）。本实现按定义的 float32 运算顺序逐步计算，输出与定义逐字节相同，
// 由该仓 cpp/tests 的 fixtures 判定。
//
// 规范副本在该仓 cpp/src/，MaaEnd 的 MapLocator/CameraOrientation{Preprocess.h,Preprocess.cpp,
// AzimuthTable.inc} 由 cpp/tools/sync_maaend.py 原样同步；改动在那边做，不在 MaaEnd 里直接改。
//
// kPreprocessDefinitionHash 锁定本实现对应的定义版本。交付的 preprocess.onnx 在 metadata 里带着定义的
// 哈希，MaaEnd 在构建期（agent/cpp-algo/cmake/CameraOrientation.cmake）核对二者，不一致则构建失败。
inline constexpr std::string_view kPreprocessDefinitionHash = "223450125d4464985270056d6e88b8394515c31494588dcc5f6bfc70022ef600";

inline constexpr int kOrientationRoiWidth = 118;
inline constexpr int kOrientationRoiHeight = 120;
inline constexpr int kOrientationStripHeight = 42;
inline constexpr int kOrientationStripWidth = 360;

struct OrientationStrips
{
    cv::Mat observed;  // CV_8UC3 strip, obs.BGR
    cv::Mat reference; // CV_8UC4 strip, ref.BGR (white-composited) + ref.A
};

// minimap: CV_8UC3 BGR observed ROI (118x120 at the 720p baseline).
// asset:   CV_8UC4 BGRA zone asset, may be a non-continuous ROI.
// (x, y):  located position in asset pixels; scale: ZoneTemplateScale(zone).
// Returns false when the inputs do not match the contract.
bool BuildOrientationStrips(const cv::Mat& minimap, const cv::Mat& asset, float x, float y, float scale, OrientationStrips& out);

} // namespace maplocator
