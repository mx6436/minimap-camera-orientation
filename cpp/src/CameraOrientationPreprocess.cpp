#include "CameraOrientationPreprocess.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

// 逐字节对齐依赖 float32 的逐步舍入：本文件禁止把 a * b + c 收缩成 FMA。
// MaaEnd 见 agent/cpp-algo/source/CMakeLists.txt，独立构建见 minimap-camera-orientation 的 cpp/CMakeLists.txt。

namespace maplocator
{

namespace
{

// endfield/preprocess.py 的几何常量。
constexpr float kRoiPoleU = 59.0f;
constexpr float kRoiPoleV = 60.0f;
constexpr float kInnerRadius = 12.0f;
constexpr float kOuterRadius = 54.0f;
// 参考窗口在采样范围外再留的边（像素），覆盖双线性的 floor / floor + 1 两个支撑像素。
constexpr float kWindowPad = 2.0f;

// torch.sin / torch.cos(torch.deg2rad(arange(360, float32))) 的 float32 位模式。
// 三角函数是前处理里唯一依赖 libm 的部分，不同平台在最后一位上会有出入，直接取定义的值；
// 半径与 u / v 的乘加仍在这里按定义的 float32 顺序计算。
#include "CameraOrientationAzimuthTable.inc"

static_assert(kAzimuthTableDefinitionHash == kPreprocessDefinitionHash, "CameraOrientationAzimuthTable.inc 来自另一版定义，需重新生成");

enum class Padding
{
    Zeros,
    Border,
};

// align_corners = false 下像素中心坐标 -> grid 值 -> 实际采样坐标；与图内一样走一遍以保留 float32 舍入。
float GridRoundTrip(float coord, int length)
{
    const float len = static_cast<float>(length);
    const float normalized = (2.0f * coord + 1.0f) / len - 1.0f;
    return ((normalized + 1.0f) * len - 1.0f) / 2.0f;
}

// 与 torch F.grid_sample(bilinear) 一致：border 先把坐标钳到图内再取 floor，zeros 在图外读 0；
// 先算四角权重再按 nw、ne、sw、se 顺序求和。
template <int Channels, Padding Mode>
void SampleBilinear(const cv::Mat& image, float x, float y, float* out)
{
    const int width = image.cols;
    const int height = image.rows;
    if constexpr (Mode == Padding::Border) {
        x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
        y = std::clamp(y, 0.0f, static_cast<float>(height - 1));
    }
    const int64_t x1 = static_cast<int64_t>(std::floor(x));
    const int64_t y1 = static_cast<int64_t>(std::floor(y));
    const int64_t x2 = x1 + 1;
    const int64_t y2 = y1 + 1;

    const float dx2 = static_cast<float>(x2) - x;
    const float dx1 = x - static_cast<float>(x1);
    const float dy2 = static_cast<float>(y2) - y;
    const float dy1 = y - static_cast<float>(y1);
    const float w11 = dy2 * dx2;
    const float w12 = dy2 * dx1;
    const float w21 = dy1 * dx2;
    const float w22 = dy1 * dx1;

    const auto pixel = [&](int64_t row, int64_t col, int channel) -> float {
        if constexpr (Mode == Padding::Zeros) {
            if (col < 0 || col >= width || row < 0 || row >= height) {
                return 0.0f;
            }
        }
        else {
            col = std::clamp<int64_t>(col, 0, width - 1);
            row = std::clamp<int64_t>(row, 0, height - 1);
        }
        return static_cast<float>(image.ptr<uint8_t>(static_cast<int>(row))[col * Channels + channel]);
    };

    for (int channel = 0; channel < Channels; ++channel) {
        out[channel] =
            w11 * pixel(y1, x1, channel) + w12 * pixel(y1, x2, channel) + w21 * pixel(y2, x1, channel) + w22 * pixel(y2, x2, channel);
    }
}

// torch.round（半偶）+ clamp + uint8；默认舍入模式下 nearbyint 即半偶。
uint8_t ToUint8(float value)
{
    return static_cast<uint8_t>(std::clamp(std::nearbyint(value), 0.0f, 255.0f));
}

// 单轴采样窗 [start, end)：覆盖 center ± extent 与双线性支撑，裁到资产内且非空。
void WindowAxis(float center, float extent, int size, int& start, int& end)
{
    const float lo = std::clamp(std::floor(center - extent - kWindowPad), 0.0f, static_cast<float>(size - 1));
    const float hi = std::clamp(std::floor(center + extent + kWindowPad) + 2.0f, lo + 1.0f, static_cast<float>(size));
    start = static_cast<int>(lo);
    end = static_cast<int>(hi);
}

struct StripGrid
{
    // ROI 像素中心坐标系（u 右、v 下），行 = 半径、列 = 方位角（北 = 0，顺时针 1 度/列）。
    std::array<float, kOrientationStripHeight * kOrientationStripWidth> u {};
    std::array<float, kOrientationStripHeight * kOrientationStripWidth> v {};

    StripGrid()
    {
        const float step = (kOuterRadius - kInnerRadius) / static_cast<float>(kOrientationStripHeight);
        for (int row = 0; row < kOrientationStripHeight; ++row) {
            const float radius = kInnerRadius + step * (static_cast<float>(row) + 0.5f);
            for (int col = 0; col < kOrientationStripWidth; ++col) {
                const float sin_theta = std::bit_cast<float>(kAzimuthSinBits[col]);
                const float cos_theta = std::bit_cast<float>(kAzimuthCosBits[col]);
                u[row * kOrientationStripWidth + col] = kRoiPoleU + radius * sin_theta;
                v[row * kOrientationStripWidth + col] = kRoiPoleV - radius * cos_theta;
            }
        }
    }
};

const StripGrid& GetStripGrid()
{
    static const StripGrid grid;
    return grid;
}

// protobuf varint；越界返回 false。
bool ReadVarint(const std::vector<char>& data, size_t& pos, uint64_t& value)
{
    value = 0;
    for (int shift = 0; shift < 64 && pos < data.size(); shift += 7) {
        const auto byte = static_cast<uint8_t>(data[pos++]);
        value |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

bool BuildOrientationStrips(const cv::Mat& minimap, const cv::Mat& asset, float x, float y, float scale, OrientationStrips& out)
{
    if (minimap.type() != CV_8UC3 || minimap.cols != kOrientationRoiWidth || minimap.rows != kOrientationRoiHeight) {
        return false;
    }
    if (asset.type() != CV_8UC4 || asset.empty() || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(scale)) {
        return false;
    }

    const StripGrid& grid = GetStripGrid();
    out.observed.create(kOrientationStripHeight, kOrientationStripWidth, CV_8UC3);
    out.reference.create(kOrientationStripHeight, kOrientationStripWidth, CV_8UC4);

    // 采样点相对 (x, y) 的最大偏移 = 最大半径（外径 - 半步长）* scale。
    const float step = (kOuterRadius - kInnerRadius) / static_cast<float>(kOrientationStripHeight);
    const float extent = scale * (kInnerRadius + (static_cast<float>(kOrientationStripHeight) - 0.5f) * step);
    int w0 = 0;
    int w1 = 0;
    int h0 = 0;
    int h1 = 0;
    WindowAxis(x, extent, asset.cols, w0, w1);
    WindowAxis(y, extent, asset.rows, h0, h1);
    const cv::Mat window = asset(cv::Rect(w0, h0, w1 - w0, h1 - h0));

    for (int row = 0; row < kOrientationStripHeight; ++row) {
        uint8_t* observed = out.observed.ptr<uint8_t>(row);
        uint8_t* reference = out.reference.ptr<uint8_t>(row);
        for (int col = 0; col < kOrientationStripWidth; ++col) {
            const float u = grid.u[row * kOrientationStripWidth + col];
            const float v = grid.v[row * kOrientationStripWidth + col];

            float obs[3];
            SampleBilinear<3, Padding::Border>(
                minimap,
                GridRoundTrip(u, kOrientationRoiWidth),
                GridRoundTrip(v, kOrientationRoiHeight),
                obs);

            // 参考底图坐标 = (x, y) + (q_roi - pole) * scale，再减去窗口原点。
            const float au = x + (u - kRoiPoleU) * scale - static_cast<float>(w0);
            const float av = y + (v - kRoiPoleV) * scale - static_cast<float>(h0);
            float ref[4];
            SampleBilinear<4, Padding::Zeros>(window, GridRoundTrip(au, w1 - w0), GridRoundTrip(av, h1 - h0), ref);

            // 白底合成：透明（含越界读 0）向白色过渡，alpha 保留原值。
            const float weight = ref[3] / 255.0f;
            for (int channel = 0; channel < 3; ++channel) {
                observed[col * 3 + channel] = ToUint8(obs[channel]);
                reference[col * 4 + channel] = ToUint8(ref[channel] * weight + 255.0f * (1.0f - weight));
            }
            reference[col * 4 + 3] = ToUint8(ref[3]);
        }
    }
    return true;
}

std::optional<std::string> ReadPreprocessDefinitionHash(const std::filesystem::path& model_path)
{
    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    const std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // metadata_props 是 StringStringEntryProto { key = 1, value = 2 }：
    // 0x0A len "definition_hash" 0x12 len <value>。只认这一种编码，找不到就当作没有。
    constexpr std::string_view kKey = "definition_hash";
    const auto begin = data.begin();
    for (auto it = std::search(begin, data.end(), kKey.begin(), kKey.end()); it != data.end();
         it = std::search(it + 1, data.end(), kKey.begin(), kKey.end())) {
        const auto key_pos = static_cast<size_t>(it - begin);
        if (key_pos < 2 || data[key_pos - 2] != 0x0A || static_cast<uint8_t>(data[key_pos - 1]) != kKey.size()) {
            continue;
        }
        size_t pos = key_pos + kKey.size();
        if (pos >= data.size() || data[pos] != 0x12) {
            continue;
        }
        ++pos;
        uint64_t length = 0;
        if (!ReadVarint(data, pos, length) || length > data.size() - pos) {
            continue;
        }
        return std::string(data.data() + pos, static_cast<size_t>(length));
    }
    return std::nullopt;
}

} // namespace maplocator
