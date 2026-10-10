// Time a preprocess.onnx with onnxruntime on one fixture case (same inputs as camori_fixture_test --bench).
// usage: camori_ort_bench <preprocess.onnx> <fixtures_dir> <case> [iters] [threads]
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "fixture_io.h"

namespace fs = std::filesystem;

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

    const fixture_io::CaseMeta meta = fixture_io::read_meta(dir);
    float x = meta.x;
    float y = meta.y;
    float scale = meta.scale;
    auto minimap = fixture_io::read_bin(dir / "minimap.bin");
    auto asset = fixture_io::read_bin(fixture_io::asset_path(root, dir, meta));

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    Ort::Session session(env, argv[1], so);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const std::array<int64_t, 4> mm_shape { 1, 120, 118, 3 };
    const std::array<int64_t, 4> asset_shape { 1, meta.asset_h, meta.asset_w, 4 };
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
