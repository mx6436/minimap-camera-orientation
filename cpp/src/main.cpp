// 数据准备工具：raw 截图 -> 训练条带（obs + ref）。
//
// 每条样本独立处理：720p 门 → MapLocator 冷启动全局定位（最多 kColdStartConsensusFrames
// 次）→ 入选门（Success 且 locConf >= 0.55 且底图资产存在）→ BuildOrientationStrips →
// 落两份 PNG。定位单实例串行、解码分块并行；两份 PNG 都在即跳过（--force 覆盖），
// report.json 跨 run 按样本名并集合并。

#include "MapLocator/CameraOrientationPreprocess.h"
#include "MapLocator/MapLocator.h"
#include "MapLocator/MapTypes.h"

#include <MaaUtils/ImageIo.h>
#include <MaaUtils/Logger.h>
#include <MaaUtils/NoWarningCV.hpp>

#include <meojson/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{

using maplocator::BuildOrientationStrips;
using maplocator::LocateOptions;
using maplocator::LocateResult;
using maplocator::LocateStatus;
using maplocator::MapLocator;
using maplocator::MapLocatorConfig;
using maplocator::MapPosition;
using maplocator::OrientationStrips;
using maplocator::TryExtractMinimap;
using maplocator::ZoneTemplateScale;
using maplocator::kColdStartConsensusFrames;

constexpr int kBaseWidth = 1280;
constexpr int kBaseHeight = 720;
constexpr double kLocThreshold = 0.55;
constexpr int kYoloThreads = 4;
constexpr size_t kProgressEvery = 200;

constexpr const char* kRefSubdir = "ref";
constexpr const char* kMapResourceSubdir = "image/MapLocator";
constexpr const char* kYoloModelRel = "model/map/cls.onnx";

// 报告状态：定位状态沿用 LocateStatus（0..5）；定位之前的门用负数。
constexpr int kStatusNot720p = -3;

constexpr const char* kSkipNot720p = "not_720p";
constexpr const char* kSkipLocateFailed = "locate_failed";
constexpr const char* kSkipLowConf = "low_conf";
constexpr const char* kSkipMissingAsset = "missing_asset";
constexpr const char* kSkipWriteFailed = "write_failed";

struct Options
{
    std::vector<fs::path> raw_dirs;
    fs::path out_dir = "data/processed_ref";
    fs::path resource_dir = "cpp/MaaEnd/assets/resource";
    bool force = false;
    int jobs = 1;
};

struct SampleRecord
{
    std::string name;
    bool accepted = false;
    std::string skip_reason;
    std::string zone;
    double x = 0.0;
    double y = 0.0;
    double scale = 1.0;
    double locConf = 0.0;
    int status = 0;
    std::string message;
    int attempts = 0;
};

struct Report
{
    std::map<std::string, SampleRecord> records;
};

int DefaultJobs()
{
    const unsigned hardware = std::thread::hardware_concurrency();
    return static_cast<int>(std::min(16u, std::max(1u, hardware)));
}

void PrintUsage(const char* argv0)
{
    std::cout << "usage: " << argv0 << " [options]\n"
              << "\n"
              << "把 raw 截图目录（默认 data/train_raw, data/hard_raw, data/val_raw）处理成训练条带：\n"
              << "  <out>/<name>.png      观测条带（3 通道）\n"
              << "  <out>/ref/<name>.png  参考条带（4 通道）\n"
              << "  <out>/report.json     逐样本记录与汇总（跨 run 并集）\n"
              << "\n"
              << "options:\n"
              << "  --raw-dir DIR        原始样本目录，可重复；给定时覆盖默认三个\n"
              << "  --out DIR            输出目录（默认 data/processed_ref）\n"
              << "  --resource-dir DIR   MaaEnd 资源根（默认 cpp/MaaEnd/assets/resource），\n"
              << "                       读取 image/MapLocator 与 model/map/cls.onnx\n"
              << "  --jobs N             解码并行线程数（默认 min(16, CPU 数)）\n"
              << "  --force              忽略已存在的产物，重算并覆盖 report 记录\n"
              << "  -h, --help           显示本帮助\n";
}

enum class ParseOutcome
{
    Ok,
    Help,
    Error,
};

ParseOutcome ParseArgs(int argc, char** argv, Options* options)
{
    bool raw_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--raw-dir") {
            const char* text = value("--raw-dir");
            if (text == nullptr) {
                return ParseOutcome::Error;
            }
            if (!raw_given) {
                options->raw_dirs.clear();
                raw_given = true;
            }
            options->raw_dirs.emplace_back(text);
        }
        else if (arg == "--out") {
            const char* text = value("--out");
            if (text == nullptr) {
                return ParseOutcome::Error;
            }
            options->out_dir = text;
        }
        else if (arg == "--resource-dir") {
            const char* text = value("--resource-dir");
            if (text == nullptr) {
                return ParseOutcome::Error;
            }
            options->resource_dir = text;
        }
        else if (arg == "--force") {
            options->force = true;
        }
        else if (arg == "--jobs") {
            const char* text = value("--jobs");
            if (text == nullptr) {
                return ParseOutcome::Error;
            }
            char* end = nullptr;
            const long jobs = std::strtol(text, &end, 10);
            if (end == text || *end != '\0' || jobs < 1) {
                std::cerr << "invalid --jobs: " << text << "\n";
                return ParseOutcome::Error;
            }
            options->jobs = static_cast<int>(std::min<long>(jobs, 256));
        }
        else if (arg == "-h" || arg == "--help") {
            return ParseOutcome::Help;
        }
        else {
            std::cerr << "unknown argument: " << arg << "\n";
            return ParseOutcome::Error;
        }
    }
    return ParseOutcome::Ok;
}

json::value RecordToJson(const SampleRecord& record)
{
    return json::object {
        { "name", record.name },
        { "accepted", record.accepted },
        { "skip_reason", record.skip_reason.empty() ? json::value(nullptr) : json::value(record.skip_reason) },
        { "zone", record.zone },
        { "x", record.x },
        { "y", record.y },
        { "scale", record.scale },
        { "locConf", record.locConf },
        { "status", record.status },
        { "message", record.message },
        { "attempts", record.attempts },
    };
}

SampleRecord RecordFromJson(const json::value& item)
{
    if (!item.is_object() || !item.contains("name")) {
        throw std::runtime_error("report record must be an object with name");
    }
    SampleRecord record;
    record.name = item.at("name").as_string();
    if (item.contains("skip_reason") && !item.at("skip_reason").is_null()) {
        record.skip_reason = item.at("skip_reason").as_string();
    }
    record.accepted = record.skip_reason.empty();
    if (item.contains("zone") && item.at("zone").is_string()) {
        record.zone = item.at("zone").as_string();
    }
    if (item.contains("x")) {
        record.x = item.at("x").as_double();
    }
    if (item.contains("y")) {
        record.y = item.at("y").as_double();
    }
    if (item.contains("scale")) {
        record.scale = item.at("scale").as_double();
    }
    if (item.contains("locConf")) {
        record.locConf = item.at("locConf").as_double();
    }
    if (item.contains("status")) {
        record.status = item.at("status").as_integer();
    }
    if (item.contains("message") && item.at("message").is_string()) {
        record.message = item.at("message").as_string();
    }
    if (item.contains("attempts")) {
        record.attempts = item.at("attempts").as_integer();
    }
    return record;
}

Report LoadReport(const fs::path& path)
{
    Report report;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) {
        return report;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open report: " + path.string());
    }
    const std::string text { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
    const auto parsed = json::parse(text);
    if (!parsed || !parsed->is_object() || !parsed->contains("records")) {
        throw std::runtime_error("invalid report: " + path.string());
    }
    const auto& records = parsed->at("records");
    if (!records.is_array()) {
        throw std::runtime_error("report records must be an array: " + path.string());
    }
    for (const auto& item : records.as_array()) {
        SampleRecord record = RecordFromJson(item);
        report.records[record.name] = std::move(record);
    }
    return report;
}

struct Tally
{
    int accepted = 0;
    std::map<std::string, int> skipped_by_reason;
};

Tally TallyRecords(const Report& report)
{
    Tally tally;
    for (const auto& [name, record] : report.records) {
        if (record.accepted) {
            ++tally.accepted;
        }
        else {
            ++tally.skipped_by_reason[record.skip_reason];
        }
    }
    return tally;
}

json::value SummaryToJson(const Report& report)
{
    const Tally tally = TallyRecords(report);
    json::object skipped;
    for (const auto& [reason, count] : tally.skipped_by_reason) {
        skipped[reason] = json::value(count);
    }
    return json::object {
        { "total", static_cast<int>(report.records.size()) },
        { "accepted", tally.accepted },
        { "skipped_by_reason", skipped },
    };
}

void WriteReport(const fs::path& out_dir, const Report& report)
{
    fs::create_directories(out_dir);
    json::array records;
    for (const auto& [name, record] : report.records) {
        records.emplace_back(RecordToJson(record));
    }
    const json::value root = json::object {
        { "records", std::move(records) },
        { "summary", SummaryToJson(report) },
    };

    const fs::path final_path = out_dir / "report.json";
    const fs::path temp_path = out_dir / "report.json.tmp";
    {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        output << root.dumps(2) << '\n';
        if (!output) {
            throw std::runtime_error("cannot write report: " + temp_path.string());
        }
    }
    fs::rename(temp_path, final_path);
}

cv::Mat DecodeFrame(const fs::path& path)
{
    cv::Mat frame = MAA_NS::imread(path, cv::IMREAD_UNCHANGED);
    if (frame.empty()) {
        throw std::runtime_error("failed to decode: " + path.string());
    }
    if (frame.channels() != 3 && frame.channels() != 4) {
        throw std::runtime_error("expected a 3/4-channel PNG: " + path.string());
    }
    if (frame.depth() != CV_8U) {
        throw std::runtime_error("expected an 8-bit PNG: " + path.string());
    }
    return frame;
}

// 4 通道帧按观测定义丢 alpha：全透明像素的 BGR 先清零，再取前 3 通道。
cv::Mat ToBgr(const cv::Mat& frame)
{
    if (frame.channels() == 3) {
        return frame;
    }
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);
    const cv::Mat transparent = channels[3] == 0;
    for (int channel = 0; channel < 3; ++channel) {
        channels[static_cast<size_t>(channel)].setTo(0, transparent);
    }
    cv::Mat bgr;
    cv::merge(std::vector<cv::Mat> { channels[0], channels[1], channels[2] }, bgr);
    return bgr;
}

bool WriteStrips(const fs::path& obs_path, const fs::path& ref_path, const OrientationStrips& strips)
{
    // 先写 .part.png 再改名：中断不会留下半张被断点续跑当成完成品的 PNG。
    const fs::path obs_temp = obs_path.string() + ".part.png";
    const fs::path ref_temp = ref_path.string() + ".part.png";
    std::error_code ec;
    fs::create_directories(obs_path.parent_path(), ec);
    fs::create_directories(ref_path.parent_path(), ec);
    if (!MAA_NS::imwrite(obs_temp, strips.observed) || !MAA_NS::imwrite(ref_temp, strips.reference)) {
        std::error_code remove_ec;
        fs::remove(obs_temp, remove_ec);
        fs::remove(ref_temp, remove_ec);
        return false;
    }
    fs::rename(obs_temp, obs_path, ec);
    if (ec) {
        std::error_code remove_ec;
        fs::remove(obs_temp, remove_ec);
        return false;
    }
    fs::rename(ref_temp, ref_path, ec);
    if (ec) {
        std::error_code remove_ec;
        fs::remove(ref_temp, remove_ec);
        return false;
    }
    return true;
}

size_t RemovePartialOutputs(const fs::path& dir)
{
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return 0;
    }
    size_t removed = 0;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (!entry.is_regular_file() || !name.ends_with(".part.png")) {
            continue;
        }
        std::error_code remove_ec;
        if (fs::remove(entry.path(), remove_ec)) {
            ++removed;
        }
    }
    return removed;
}

class AssetCache
{
public:
    explicit AssetCache(fs::path root)
        : root_(std::move(root))
    {
    }

    std::optional<fs::path> ResolveAsset(const std::string& zone_id) const
    {
        static const std::regex base_pattern(R"(^(.+)_Base$)");
        static const std::regex tier_pattern(R"(^(.+)_L(\d+)_(\d+)$)");
        const auto existing = [](const fs::path& candidate) -> std::optional<fs::path> {
            if (!fs::is_regular_file(candidate)) {
                return std::nullopt;
            }
            return candidate;
        };

        std::smatch match;
        if (std::regex_match(zone_id, match, base_pattern)) {
            return existing(root_ / match[1].str() / "Base.png");
        }
        if (std::regex_match(zone_id, match, tier_pattern)) {
            const int level = std::stoi(match[2].str());
            std::ostringstream filename;
            filename << "Lv" << std::setw(3) << std::setfill('0') << level << "Tier" << match[3].str() << ".png";
            return existing(root_ / match[1].str() / filename.str());
        }

        std::vector<fs::path> candidates;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(root_, ec)) {
            if (!entry.is_directory()) {
                continue;
            }
            const fs::path candidate = entry.path() / (zone_id + ".png");
            if (fs::is_regular_file(candidate)) {
                candidates.push_back(candidate);
            }
        }
        if (candidates.empty()) {
            return std::nullopt;
        }
        std::sort(candidates.begin(), candidates.end());
        return candidates.front();
    }

    const cv::Mat& Load(const fs::path& path)
    {
        const auto found = images_.find(path);
        if (found != images_.end()) {
            return found->second;
        }
        cv::Mat image = MAA_NS::imread(path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            throw std::runtime_error("failed to decode asset: " + path.string());
        }
        if (image.channels() == 3) {
            cv::Mat bgra;
            cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
            image = std::move(bgra);
        }
        if (image.type() != CV_8UC4) {
            throw std::runtime_error("asset must be 8-bit 3/4-channel: " + path.string());
        }
        return images_.emplace(path, std::move(image)).first->second;
    }

private:
    fs::path root_;
    std::map<fs::path, cv::Mat> images_;
};

SampleRecord ProcessSample(
    const std::string& name,
    const cv::Mat& frame,
    MapLocator& locator,
    AssetCache& assets,
    const fs::path& out_dir)
{
    SampleRecord record;
    record.name = name;
    const fs::path obs_path = out_dir / name;
    const fs::path ref_path = out_dir / kRefSubdir / name;
    // 处理过而未入选的样本不能留下旧产物：文件存在即代表上次入选。
    const auto remove_outputs = [&]() {
        std::error_code ec;
        fs::remove(obs_path, ec);
        fs::remove(ref_path, ec);
    };
    const auto skip = [&](const char* reason) -> SampleRecord& {
        record.skip_reason = reason;
        remove_outputs();
        return record;
    };

    if (frame.cols != kBaseWidth || frame.rows != kBaseHeight) {
        std::ostringstream message;
        message << "frame is not 1280x720 (got " << frame.cols << "x" << frame.rows << ")";
        record.skip_reason = kSkipNot720p;
        record.status = kStatusNot720p;
        record.message = message.str();
        return skip(kSkipNot720p);
    }

    const cv::Mat bgr = ToBgr(frame);
    cv::Mat minimap;
    if (!TryExtractMinimap(bgr, /*use_adb_minimap_roi=*/false, &minimap)) {
        throw std::runtime_error(name + ": minimap ROI extraction failed");
    }

    locator.resetTrackingState();
    LocateOptions options;
    options.force_global_search = true;
    LocateResult result;
    for (int attempt = 1; attempt <= kColdStartConsensusFrames; ++attempt) {
        record.attempts = attempt;
        result = locator.locate(minimap, options);
        if (result.status == LocateStatus::Success) {
            break;
        }
    }
    record.status = static_cast<int>(result.status);
    record.message = result.debugMessage;

    if (result.status != LocateStatus::Success || !result.position.has_value()) {
        return skip(kSkipLocateFailed);
    }

    const MapPosition& position = *result.position;
    record.zone = position.zoneId;
    record.x = position.x;
    record.y = position.y;
    record.scale = ZoneTemplateScale(position.zoneId);
    record.locConf = position.score;

    if (position.score < kLocThreshold) {
        return skip(kSkipLowConf);
    }

    const std::optional<fs::path> asset_path = assets.ResolveAsset(position.zoneId);
    if (!asset_path.has_value()) {
        return skip(kSkipMissingAsset);
    }

    OrientationStrips strips;
    if (!BuildOrientationStrips(
            minimap,
            assets.Load(*asset_path),
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(record.scale),
            strips)) {
        throw std::runtime_error(name + ": BuildOrientationStrips rejected the inputs");
    }

    if (!WriteStrips(obs_path, ref_path, strips)) {
        record.message = "failed to write output PNG";
        return skip(kSkipWriteFailed);
    }

    record.accepted = true;
    return record;
}

std::map<std::string, fs::path> CollectSamples(const std::vector<fs::path>& raw_dirs)
{
    std::map<std::string, fs::path> samples;
    for (const fs::path& raw_dir : raw_dirs) {
        std::error_code ec;
        if (!fs::is_directory(raw_dir, ec)) {
            throw std::runtime_error("raw dir does not exist: " + raw_dir.string());
        }
        for (const auto& entry : fs::directory_iterator(raw_dir, ec)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".png") {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (!samples.emplace(name, entry.path()).second) {
                throw std::runtime_error("sample name in multiple raw dirs: " + name);
            }
        }
        if (ec) {
            throw std::runtime_error("cannot scan raw dir: " + raw_dir.string());
        }
    }
    if (samples.empty()) {
        throw std::runtime_error("no raw png samples");
    }
    return samples;
}

void PrintSummary(const Report& report, size_t processed, size_t existing, const fs::path& path)
{
    const Tally tally = TallyRecords(report);
    std::cout << "processed=" << processed << " existing=" << existing << " total=" << report.records.size() << "\n";
    std::cout << "accepted=" << tally.accepted << " skipped=" << (report.records.size() - static_cast<size_t>(tally.accepted));
    for (const auto& [reason, count] : tally.skipped_by_reason) {
        std::cout << " " << reason << "=" << count;
    }
    std::cout << "\n-> " << path.string() << std::endl;
}

int Run(const Options& options)
{
    const std::map<std::string, fs::path> samples = CollectSamples(options.raw_dirs);
    const fs::path obs_dir = options.out_dir;
    const fs::path ref_dir = obs_dir / kRefSubdir;
    fs::create_directories(ref_dir);

    const size_t partials = RemovePartialOutputs(obs_dir) + RemovePartialOutputs(ref_dir);
    if (partials > 0) {
        std::cout << "cleaned=" << partials << " partial outputs" << std::endl;
    }
    Report report = LoadReport(obs_dir / "report.json");

    MapLocatorConfig config;
    config.mapResourceDir = (options.resource_dir / kMapResourceSubdir).string();
    config.yoloModelPath = (options.resource_dir / kYoloModelRel).string();
    config.yoloThreads = kYoloThreads;
    MapLocator locator;
    if (!locator.initialize(config)) {
        throw std::runtime_error("MapLocator initialize failed: resource-dir=" + options.resource_dir.string());
    }
    AssetCache assets(options.resource_dir / kMapResourceSubdir);

    std::vector<std::pair<std::string, fs::path>> pending;
    size_t existing = 0;
    for (const auto& [name, source] : samples) {
        const bool both_outputs = fs::is_regular_file(obs_dir / name) && fs::is_regular_file(ref_dir / name);
        if (both_outputs && !options.force) {
            ++existing;
            continue;
        }
        pending.emplace_back(name, source);
    }
    std::cout << "raw=" << samples.size() << " existing=" << existing << " pending=" << pending.size()
              << " jobs=" << options.jobs << std::endl;

    const size_t chunk_size = std::max<size_t>(8, static_cast<size_t>(options.jobs) * 2);
    size_t processed = 0;
    for (size_t start = 0; start < pending.size(); start += chunk_size) {
        const size_t end = std::min(start + chunk_size, pending.size());
        std::vector<std::future<cv::Mat>> decodes;
        decodes.reserve(end - start);
        for (size_t index = start; index < end; ++index) {
            decodes.push_back(std::async(std::launch::async, DecodeFrame, pending[index].second));
        }
        for (size_t index = start; index < end; ++index) {
            const cv::Mat frame = decodes[index - start].get();
            SampleRecord record = ProcessSample(pending[index].first, frame, locator, assets, obs_dir);
            report.records[pending[index].first] = std::move(record);
            ++processed;
            if (processed % kProgressEvery == 0) {
                std::cout << "[" << processed << "/" << pending.size() << "] " << pending[index].first << std::endl;
                WriteReport(obs_dir, report);
            }
        }
    }
    WriteReport(obs_dir, report);
    PrintSummary(report, processed, existing, obs_dir / "report.json");
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    MAA_NS::LogNS::Logger::get_instance().set_stdout_level(MAA_NS::LogNS::level::off);

    Options options;
    options.raw_dirs = { "data/train_raw", "data/hard_raw", "data/val_raw" };
    options.jobs = DefaultJobs();

    const ParseOutcome outcome = ParseArgs(argc, argv, &options);
    if (outcome == ParseOutcome::Help) {
        PrintUsage(argv[0]);
        return 0;
    }
    if (outcome == ParseOutcome::Error) {
        PrintUsage(argv[0]);
        return 2;
    }

    try {
        return Run(options);
    }
    catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
