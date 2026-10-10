#include "camori/preprocess.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>

namespace camori
{
namespace
{

#include "strip_grid_table.inc"

static_assert(std::string_view(CAMORI_STRIP_GRID_DEFINITION_HASH) == kDefinitionHash, "strip grid table is from another definition");

// ORT GsDenormalize(align_corners = false): [-1, 1] -> [-0.5, length - 0.5]
inline float denormalize(float n, int length)
{
    return ((n + 1.0f) * static_cast<float>(length) - 1.0f) / 2.0f;
}

// _normalized(): pixel-center coordinate -> GridSample grid value
inline float normalize(float p, float length)
{
    return (2.0f * p + 1.0f) / length - 1.0f;
}

enum class Padding
{
    Zeros,
    Border,
};

// Bilinear grid sample (align_corners = 0) of all channels of an HWC uint8 image, matching torch
// F.grid_sample (the definition): border clamps the coordinate before floor, zeros reads 0 outside.
// `img` may be a zero-copy ROI of the zone asset (the window crop is never materialized).
template <int C, Padding P>
inline void bilinear(const cv::Mat& img, float x, float y, float out[C])
{
    const int W = img.cols;
    const int H = img.rows;
    if constexpr (P == Padding::Border) {
        // ORT clamps the denormalized coordinate for border padding before taking the floor
        x = std::clamp(x, 0.0f, static_cast<float>(W - 1));
        y = std::clamp(y, 0.0f, static_cast<float>(H - 1));
    }
    const int64_t x1 = static_cast<int64_t>(std::floor(x));
    const int64_t y1 = static_cast<int64_t>(std::floor(y));
    const int64_t x2 = x1 + 1;
    const int64_t y2 = y1 + 1;

    auto pixel = [&](int64_t r, int64_t c, int ch) -> float {
        if constexpr (P == Padding::Zeros) {
            if (c < 0 || c >= W || r < 0 || r >= H) {
                return 0.0f;
            }
        }
        else {
            c = std::clamp<int64_t>(c, 0, W - 1);
            r = std::clamp<int64_t>(r, 0, H - 1);
        }
        return static_cast<float>(img.ptr<uint8_t>(static_cast<int>(r))[c * C + ch]);
    };

    const float dx2 = static_cast<float>(x2) - x;
    const float dx1 = x - static_cast<float>(x1);
    const float dy2 = static_cast<float>(y2) - y;
    const float dy1 = y - static_cast<float>(y1);
    // torch grid_sampler_2d weights (nw, ne, sw, se), summed in that order
    const float w11 = dy2 * dx2;
    const float w12 = dy2 * dx1;
    const float w21 = dy1 * dx2;
    const float w22 = dy1 * dx1;
    for (int ch = 0; ch < C; ++ch) {
        out[ch] = w11 * pixel(y1, x1, ch) + w12 * pixel(y1, x2, ch) + w21 * pixel(y2, x1, ch) + w22 * pixel(y2, x2, ch);
    }
}

// _to_uint8(): torch.round (half to even) + clamp + cast
inline uint8_t to_u8(float v)
{
    const float r = std::nearbyint(v); // default rounding mode is FE_TONEAREST (half to even)
    return static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
}

// _window_axis(): [start, end) covering center±extent plus bilinear support, clipped, non-empty
inline void window_axis(float center, float extent, int size, int& start, int& end)
{
    const float s = std::clamp(std::floor(center - extent - kWindowPad), 0.0f, static_cast<float>(size - 1));
    const float e = std::clamp(std::floor(center + extent + kWindowPad) + 2.0f, s + 1.0f, static_cast<float>(size));
    start = static_cast<int>(s);
    end = static_cast<int>(e);
}

} // namespace

StripGrid::StripGrid()
    : u_(kStripH, kStripW, CV_32FC1)
    , v_(kStripH, kStripW, CV_32FC1)
{
    // strip_roi_uv() values baked bit-exact from the definition (see strip_grid_table.inc)
    static_assert(std::size(kStripGridU) == size_t(kStripH) * kStripW);
    static_assert(std::size(kStripGridV) == size_t(kStripH) * kStripW);
    for (int i = 0; i < kStripH; ++i) {
        float* pu = u_.ptr<float>(i);
        float* pv = v_.ptr<float>(i);
        for (int j = 0; j < kStripW; ++j) {
            pu[j] = std::bit_cast<float>(kStripGridU[i * kStripW + j]);
            pv[j] = std::bit_cast<float>(kStripGridV[i * kStripW + j]);
        }
    }
}

const StripGrid& strip_grid()
{
    static const StripGrid grid;
    return grid;
}

bool preprocess(const cv::Mat& minimap, const cv::Mat& asset, float x, float y, float scale, Strips& out)
{
    if (minimap.type() != CV_8UC3 || minimap.cols != kRoiW || minimap.rows != kRoiH) {
        return false;
    }
    if (asset.type() != CV_8UC4 || asset.empty()) {
        return false;
    }

    const StripGrid& grid = strip_grid();
    out.observed.create(kStripH, kStripW, CV_8UC3);
    out.reference.create(kStripH, kStripW, CV_8UC4);

    // _sampling_extent(): scale * max radius
    const float step = (kOuterR - kInnerR) / static_cast<float>(kStripH);
    const float extent = scale * (kInnerR + (static_cast<float>(kStripH) - 0.5f) * step);
    int w0 = 0, w1 = 0, h0 = 0, h1 = 0;
    window_axis(x, extent, asset.cols, w0, w1);
    window_axis(y, extent, asset.rows, h0, h1);
    // asset[h0:h1, w0:w1] without copy
    const cv::Mat crop = asset(cv::Rect(w0, h0, w1 - w0, h1 - h0));
    const float crop_w = static_cast<float>(w1 - w0);
    const float crop_h = static_cast<float>(h1 - h0);

    for (int i = 0; i < kStripH; ++i) {
        const float* pu = grid.u().ptr<float>(i);
        const float* pv = grid.v().ptr<float>(i);
        uint8_t* po = out.observed.ptr<uint8_t>(i);
        uint8_t* pr = out.reference.ptr<uint8_t>(i);
        for (int j = 0; j < kStripW; ++j) {
            // sample_minimap(): grid computed on the fly, then the graph normalizes / ORT denormalizes
            float obs[3];
            bilinear<3, Padding::Border>(
                minimap,
                denormalize(normalize(pu[j], static_cast<float>(kRoiW)), kRoiW),
                denormalize(normalize(pv[j], static_cast<float>(kRoiH)), kRoiH),
                obs);

            // _sample_asset_crop(): world coordinate relative to the window origin
            const float au = x + (pu[j] - kRoiPoleU) * scale - static_cast<float>(w0);
            const float av = y + (pv[j] - kRoiPoleV) * scale - static_cast<float>(h0);
            float smp[4];
            bilinear<4, Padding::Zeros>(
                crop,
                denormalize(normalize(au, crop_w), w1 - w0),
                denormalize(normalize(av, crop_h), h1 - h0),
                smp);

            // _compose_strips(): white-background composite
            const float weight = smp[3] / 255.0f;
            for (int c = 0; c < 3; ++c) {
                po[j * 3 + c] = to_u8(obs[c]);
                pr[j * 4 + c] = to_u8(smp[c] * weight + 255.0f * (1.0f - weight));
            }
            pr[j * 4 + 3] = to_u8(smp[3]);
        }
    }
    return true;
}

} // namespace camori
