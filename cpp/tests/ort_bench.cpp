// Time the shipped preprocess.onnx with onnxruntime on one fixture case (same inputs as preprocess_test --bench).
// usage: camori_ort_bench <preprocess.onnx> <fixtures_dir> <case> [iters] [threads]
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

static std::vector<uint8_t> read_bin(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::string meta_value(const std::string& s, const std::string& key)
{
    std::smatch m;
    std::regex re("\"" + key + R"re(":\s*"?([^",}]*))re");
    return std::regex_search(s, m, re) ? m[1].str() : std::string();
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::printf("usage: camori_ort_bench <preprocess.onnx> <fixtures_dir> <case> [iters] [threads]\n");
        return 1;
    }
    const fs::path root = argv[2];
    const fs::path dir = root / argv[3];
    const int iters = argc > 4 ? std::atoi(argv[4]) : 200;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 2; // CameraOrientationPredictor default

    std::ifstream mf(dir / "meta.json");
    const std::string meta((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    const int64_t ah = std::stoll(meta_value(meta, "asset_h"));
    const int64_t aw = std::stoll(meta_value(meta, "asset_w"));
    float x = std::stof(meta_value(meta, "x"));
    float y = std::stof(meta_value(meta, "y"));
    float scale = std::stof(meta_value(meta, "scale"));
    const std::string asset_ref = meta_value(meta, "asset_ref");
    auto minimap = read_bin(dir / "minimap.bin");
    auto asset = read_bin(asset_ref.empty() ? dir / "asset.bin" : root / "shared" / (asset_ref + ".bin"));

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    Ort::Session session(env, argv[1], so);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const std::array<int64_t, 4> mm_shape { 1, 120, 118, 3 };
    const std::array<int64_t, 4> asset_shape { 1, ah, aw, 4 };
    const char* in_names[] = { "minimap", "asset", "x", "y", "scale" };
    const char* out_names[] = { "observed", "reference" };

    auto run = [&]() {
        Ort::Value inputs[] = {
            Ort::Value::CreateTensor<uint8_t>(mem, minimap.data(), minimap.size(), mm_shape.data(), 4),
            Ort::Value::CreateTensor<uint8_t>(mem, asset.data(), asset.size(), asset_shape.data(), 4),
            Ort::Value::CreateTensor<float>(mem, &x, 1, nullptr, 0),
            Ort::Value::CreateTensor<float>(mem, &y, 1, nullptr, 0),
            Ort::Value::CreateTensor<float>(mem, &scale, 1, nullptr, 0),
        };
        return session.Run(Ort::RunOptions { nullptr }, in_names, inputs, 5, out_names, 2);
    };

    for (int i = 0; i < 5; ++i) {
        run();
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        run();
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("ORT %s preprocess.onnx case=%s threads=%d avg=%.3f ms\n", Ort::GetVersionString().c_str(), argv[3], threads, ms);
    return 0;
}
