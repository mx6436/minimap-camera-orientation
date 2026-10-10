#pragma once

// Readers for the fixture layout written by cpp/tools/gen_fixtures.py, shared by fixture_test.cpp and ort_bench.cpp.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace fixture_io
{

// Whole file in one read; empty when the file is missing.
inline std::vector<uint8_t> read_bin(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return {};
    }
    std::vector<uint8_t> data(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return data;
}

inline cv::FileStorage open_json(const std::filesystem::path& path)
{
    return cv::FileStorage(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
}

// <case>/meta.json
struct CaseMeta
{
    int asset_h = 0;
    int asset_w = 0;
    float x = 0.0f;
    float y = 0.0f;
    float scale = 0.0f;
    std::string asset_ref; // non-empty: the asset is shared/<asset_ref>.bin instead of <case>/asset.bin
};

inline CaseMeta read_meta(const std::filesystem::path& case_dir)
{
    const cv::FileStorage fs = open_json(case_dir / "meta.json");
    CaseMeta meta;
    meta.asset_h = static_cast<int>(fs["asset_h"]);
    meta.asset_w = static_cast<int>(fs["asset_w"]);
    // double first, then float32: the same conversion as the definition's float(x) -> torch float32
    meta.x = static_cast<float>(static_cast<double>(fs["x"]));
    meta.y = static_cast<float>(static_cast<double>(fs["y"]));
    meta.scale = static_cast<float>(static_cast<double>(fs["scale"]));
    meta.asset_ref = static_cast<std::string>(fs["asset_ref"]);
    return meta;
}

inline std::filesystem::path asset_path(const std::filesystem::path& root, const std::filesystem::path& case_dir, const CaseMeta& meta)
{
    return meta.asset_ref.empty() ? case_dir / "asset.bin" : root / "shared" / (meta.asset_ref + ".bin");
}

} // namespace fixture_io
