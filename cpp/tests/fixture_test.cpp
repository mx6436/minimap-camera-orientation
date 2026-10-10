// Compare maplocator::BuildOrientationStrips (cpp/src/, shipped verbatim in MaaEnd) against fixtures produced by
// cpp/tools/gen_fixtures.py.
//
// usage: camori_fixture_test <fixtures_dir> [--tolerance N] [--bench N] [--verbose]
//
// Every case carries two expectations:
//   def_*  endfield/preprocess.py (torch definition, source of truth)
//   ort_*  the preprocess.onnx exported from the definition, run by onnxruntime (reference only)
// Pass condition: max |diff| <= tolerance vs def_* for every case. Default 0 (byte-identical to the definition);
// --tolerance 1 is the conformance profile (DEFAULT_TOLERANCES["strips_uint8"]).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "CameraOrientationPreprocess.h"
#include "fixture_io.h"

namespace fs = std::filesystem;
using fixture_io::read_bin;

namespace
{

// |diff| histogram: [0] = equal, [1] = 1 LSB, [2] = more
struct Stats
{
    std::array<size_t, 3> hist {};
    int max_abs = 0;
    size_t cases = 0;
    size_t cases_over = 0;

    void add(const Stats& o)
    {
        for (int i = 0; i < 3; ++i) {
            hist[i] += o.hist[i];
        }
        max_abs = std::max(max_abs, o.max_abs);
        cases += o.cases;
        cases_over += o.cases_over;
    }

    size_t total() const { return hist[0] + hist[1] + hist[2]; }
};

Stats compare(const cv::Mat& actual, const std::vector<uint8_t>& expected, int tolerance)
{
    Stats s;
    s.cases = 1;
    const size_t n = actual.total() * actual.channels();
    if (expected.size() != n) {
        s.max_abs = 999;
        s.hist[2] = n;
        s.cases_over = 1;
        return s;
    }
    const uint8_t* a = actual.ptr<uint8_t>();
    for (size_t i = 0; i < n; ++i) {
        const int d = std::abs(int(a[i]) - int(expected[i]));
        ++s.hist[std::min(d, 2)];
        s.max_abs = std::max(s.max_abs, d);
    }
    s.cases_over = s.max_abs > tolerance;
    return s;
}

std::string group_of(const std::string& name)
{
    const auto pos = name.find('_');
    return pos == std::string::npos ? name : name.substr(0, pos);
}

void print_row(const char* group, const char* ref, const char* out, const Stats& s)
{
    const double total = double(s.total());
    std::printf(
        "%-6s %-4s %-9s %6zu %9.5f%% %9.5f%% %9.5f%% %7d %9zu\n",
        group,
        ref,
        out,
        s.cases,
        100.0 * double(s.hist[0]) / total,
        100.0 * double(s.hist[1]) / total,
        100.0 * double(s.hist[2]) / total,
        s.max_abs,
        s.cases_over);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::printf("usage: camori_fixture_test <fixtures_dir> [--tolerance N] [--bench N] [--verbose]\n");
        return 1;
    }
    const fs::path root = argv[1];
    int bench = 0;
    int tolerance = 0;
    bool verbose = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench" && i + 1 < argc) {
            bench = std::atoi(argv[++i]);
        }
        else if (arg == "--tolerance" && i + 1 < argc) {
            tolerance = std::atoi(argv[++i]);
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
    }

    const std::string fixture_hash = static_cast<std::string>(fixture_io::open_json(root / "index.json")["definition_hash"]);

    std::vector<fs::path> dirs;
    for (const auto& e : fs::directory_iterator(root)) {
        if (e.is_directory() && fs::exists(e.path() / "meta.json")) {
            dirs.push_back(e.path());
        }
    }
    std::sort(dirs.begin(), dirs.end());

    const char* refs[2] = { "def", "ort" };
    const char* outs[2] = { "observed", "reference" };
    // stats[group][ref][out]
    std::map<std::string, std::array<std::array<Stats, 2>, 2>> stats;
    std::map<std::string, std::vector<uint8_t>> shared_assets;
    int failed = 0;
    double bench_ms = 0;
    size_t bench_calls = 0;

    for (const auto& dir : dirs) {
        const std::string name = dir.filename().string();
        const fixture_io::CaseMeta meta = fixture_io::read_meta(dir);
        const float x = meta.x;
        const float y = meta.y;
        const float scale = meta.scale;

        auto minimap_buf = read_bin(dir / "minimap.bin");
        std::vector<uint8_t> own_asset;
        std::vector<uint8_t>* asset_buf = &own_asset;
        if (meta.asset_ref.empty()) {
            own_asset = read_bin(fixture_io::asset_path(root, dir, meta));
        }
        else {
            auto it = shared_assets.find(meta.asset_ref);
            if (it == shared_assets.end()) {
                it = shared_assets.emplace(meta.asset_ref, read_bin(fixture_io::asset_path(root, dir, meta))).first;
            }
            asset_buf = &it->second;
        }
        const cv::Mat minimap(maplocator::kOrientationRoiHeight, maplocator::kOrientationRoiWidth, CV_8UC3, minimap_buf.data());
        const cv::Mat asset(meta.asset_h, meta.asset_w, CV_8UC4, asset_buf->data());

        maplocator::OrientationStrips strips;
        if (!maplocator::BuildOrientationStrips(minimap, asset, x, y, scale, strips)) {
            std::printf("%-26s preprocess() returned false  <-- FAIL\n", name.c_str());
            ++failed;
            continue;
        }

        auto& group = stats[group_of(name)];
        bool case_ok = true;
        std::string line = name;
        line.resize(26, ' ');
        for (int r = 0; r < 2; ++r) {
            for (int o = 0; o < 2; ++o) {
                const auto expected = read_bin(dir / (std::string(refs[r]) + "_" + outs[o] + ".bin"));
                const Stats s = compare(o == 0 ? strips.observed : strips.reference, expected, tolerance);
                group[r][o].add(s);
                if (r == 0 && s.max_abs > tolerance) {
                    case_ok = false;
                }
                char buf[64];
                std::snprintf(buf, sizeof(buf), "  %s.%s max=%d", refs[r], o == 0 ? "obs" : "ref", s.max_abs);
                line += buf;
            }
        }
        if (!case_ok) {
            ++failed;
        }
        if (verbose || !case_ok) {
            std::printf("%s%s\n", line.c_str(), case_ok ? "" : "  <-- FAIL");
        }

        if (bench > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < bench; ++i) {
                maplocator::BuildOrientationStrips(minimap, asset, x, y, scale, strips);
            }
            bench_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            bench_calls += bench;
        }
    }

    std::printf("%zu cases, definition_hash %.12s, tolerance %d LSB vs def\n\n", dirs.size(), fixture_hash.c_str(), tolerance);
    std::printf("%-6s %-4s %-9s %6s %10s %10s %10s %7s %9s\n", "group", "vs", "output", "cases", "equal", "diff=1", "diff>1", "max", "cases>tol");
    std::array<std::array<Stats, 2>, 2> all {};
    for (const auto& [g, rs] : stats) {
        for (int r = 0; r < 2; ++r) {
            for (int o = 0; o < 2; ++o) {
                all[r][o].add(rs[r][o]);
                print_row(g.c_str(), refs[r], outs[o], rs[r][o]);
            }
        }
    }
    for (int r = 0; r < 2; ++r) {
        for (int o = 0; o < 2; ++o) {
            print_row("all", refs[r], outs[o], all[r][o]);
        }
    }
    if (bench_calls) {
        std::printf("\navg preprocess() = %.3f ms over %zu calls\n", bench_ms / double(bench_calls), bench_calls);
    }
    std::printf("\n%s\n", failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}
