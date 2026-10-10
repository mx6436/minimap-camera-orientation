#pragma once

// C++ equivalent of the delivered `preprocess.onnx` (camera-orientation preprocess graph).
//
// Source of truth: `endfield/preprocess.py` in this repo
// (definition_hash 223450125d4464985270056d6e88b8394515c31494588dcc5f6bfc70022ef600).
// The graph has no learned weights: two bilinear grid samples (align_corners=0) plus window cropping and
// white-background alpha compositing. Arithmetic is float32 in the same operation order as the torch
// definition, and the strip grid is baked bit-exact from strip_roi_uv(), so the output is byte-identical to
// endfield/preprocess.strip_pair() (conformance tolerance strips_uint8 = 1 LSB is not needed).

#include <cstdint>
#include <string_view>

#include <opencv2/core.hpp>

namespace camori
{

// endfield/preprocess.py geometry constants
inline constexpr int kRoiW = 118;
inline constexpr int kRoiH = 120;
inline constexpr float kRoiPoleU = 59.0f;
inline constexpr float kRoiPoleV = 60.0f;
inline constexpr float kInnerR = 12.0f;
inline constexpr float kOuterR = 54.0f;
inline constexpr int kStripH = 42;
inline constexpr int kStripW = 360;
inline constexpr float kWindowPad = 2.0f;

// definition this implementation follows; compare with preprocess.onnx metadata `definition_hash`
inline constexpr std::string_view kDefinitionHash = "223450125d4464985270056d6e88b8394515c31494588dcc5f6bfc70022ef600";

struct Strips
{
    cv::Mat observed;  // CV_8UC3, kStripH x kStripW, obs.BGR
    cv::Mat reference; // CV_8UC4, kStripH x kStripW, ref.BGR (white-composited) + ref.A
};

// Precomputed strip grid in ROI pixel-center coordinates (u right, v down); one per process.
class StripGrid
{
public:
    StripGrid();

    const cv::Mat& u() const { return u_; } // CV_32FC1 kStripH x kStripW
    const cv::Mat& v() const { return v_; }

private:
    cv::Mat u_;
    cv::Mat v_;
};

// minimap: CV_8UC3 BGR, kRoiH x kRoiW (MapLocator minimap ROI at 720p).
// asset:   CV_8UC4 BGRA zone map (3-channel assets must be expanded with alpha 255 first).
// x, y:    MapLocator position in asset pixels; scale: ZoneTemplateScale(zone).
// Returns false on invalid input.
bool preprocess(const cv::Mat& minimap, const cv::Mat& asset, float x, float y, float scale, Strips& out);

const StripGrid& strip_grid();

} // namespace camori
