/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <argparse/argparse.hpp>
#include <vgf/logging.hpp>

#include "context.hpp"
#include "logging.hpp"
#include "scenario.hpp"
#include "version.hpp"

#include <filesystem>
#include <iostream>

using namespace mlsdk::scenariorunner;
using namespace mlsdk::logging;
namespace {
constexpr std::array<std::string_view, 5> extensionList = {
    VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_5_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};

std::string printExtensionList() {
    size_t numExtensions = extensionList.size() - 1;
    std::stringstream outputString;
    for (size_t idx = 0; idx < numExtensions; idx++) {
        outputString << extensionList[idx] << ", ";
    }
    outputString << extensionList[numExtensions];
    return outputString.str();
}

void loggingHandler(const std::string &logger, LogLevel logLevel, const std::string &message) {
    std::ostream &stream = logLevel == LogLevel::Error ? std::cerr : std::cout;
    stream << "[" << logger << "] " << logLevel << ": " << message << std::endl;
}

LogLevel mapVGFLogLevel(mlsdk::vgflib::logging::LogLevel logLevel) {
    switch (logLevel) {
    case mlsdk::vgflib::logging::LogLevel::INFO:
        return LogLevel::Info;
    case mlsdk::vgflib::logging::LogLevel::WARNING:
        return LogLevel::Warning;
    case mlsdk::vgflib::logging::LogLevel::DEBUG:
        return LogLevel::Debug;
    case mlsdk::vgflib::logging::LogLevel::ERROR:
    default:
        return LogLevel::Error;
    }
}

void vgfLoggingHandler(mlsdk::vgflib::logging::LogLevel vgfLogLevel, const std::string &message) {
    mlsdk::logging::log("VGF", mapVGFLogLevel(vgfLogLevel), message);
}

LogLevel parseLogLevel(const std::string &logLevel) {
    if (logLevel == "debug")
        return LogLevel::Debug;

    if (logLevel == "info")
        return LogLevel::Info;

    if (logLevel == "warning")
        return LogLevel::Warning;

    if (logLevel == "error")
        return LogLevel::Error;

    throw std::runtime_error("Unknown log level " + logLevel);
}
} // namespace

void configureLogging() {
    setDefaultLoggerName("Scenario-Runner");
    setDefaultLogLevel(LogLevel::Info);
    setDefaultHandler(loggingHandler);
    mlsdk::vgflib::logging::EnableLogging(vgfLoggingHandler);
}

int main(int argc, const char **argv) {
    configureLogging();

    try {
        argparse::ArgumentParser parser(argv[0], details::version);

        ScenarioOptions scenarioOptions;

        parser.add_argument("--scenario")
            .help("file to load the scenario from. File should be in JSON format")
            .default_value<std::string>("")
            .required();
        parser.add_argument("--output").help("output folder").default_value<std::string>("");
        parser.add_argument("--profiling-dump-path")
            .help("path to save runtime profiling")
            .default_value<std::string>("");
        parser.add_argument("--pipeline-caching")
            .help("enable the pipeline caching")
            .default_value(false)
            .implicit_value(true);
        parser.add_argument("--clear-pipeline-cache")
            .help("clear pipeline cache")
            .default_value(false)
            .implicit_value(true);
        parser.add_argument("--cache-path")
            .help("set pipeline cache location")
            .default_value<std::string>(std::filesystem::temp_directory_path().string());
        parser.add_argument("--fail-on-pipeline-cache-miss")
            .help("ensure an error is generated on a pipeline cache miss")
            .default_value(false)
            .implicit_value(true);
        parser.add_argument("--perf-counters-dump-path")
            .help("path to save performance counter stats")
            .default_value<std::string>("");
        parser.add_argument("--log-level")
            .help("set logging level")
            .default_value("debug")
            .choices("debug", "info", "warning", "error");
        parser.add_argument("--wait-for-key-stroke-before-run")
            .help("Wait for a key stroke before run")
            .default_value(false)
            .implicit_value(true);
        parser.add_argument("--dry-run")
            .help("Setup pipelines but skip the actual execution")
            .default_value(false)
            .implicit_value(true);
        parser.add_argument("--disable-extension")
            .append()
            .store_into(scenarioOptions.disabledExtensions)
            .nargs(argparse::nargs_pattern::any)
            .help("specify extensions to disable out of the following: " + printExtensionList());
        parser.add_argument("--enable-gpu-debug-markers")
            .help("enable GPU debug markers")
            .default_value(false)
            .implicit_value(true);
        parser.add_argument("--session-memory-dump-dir")
            .help("path to dump the contents of the sessions ram after inference completes");
        parser.add_argument("--repeat").help("Repeat count for scenario execution").default_value(1).scan<'i', int>();
        parser.add_argument("--capture-frame")
            .help("Enable RenderDoc integration for frame capturing")
            .default_value(false)
            .implicit_value(true);

        // Main Scenario-Runner CLI execution body
        std::string scenarioFile;
        parser.parse_args(argc, argv);

        if (parser.is_used("--log-level")) {
            LogLevel logLevel = parseLogLevel(parser.get("--log-level"));
            setDefaultLogLevel(logLevel);
        }

        scenarioFile = parser.get("--scenario");
        std::filesystem::path workDir = std::filesystem::path(scenarioFile).remove_filename();
        std::ifstream fstream(scenarioFile);
        if (!fstream) {
            throw std::runtime_error("Error while opening scenario file " + scenarioFile);
        }

        std::filesystem::path outputDir = workDir;
        if (!parser.get("output").empty()) {
            outputDir = std::filesystem::path(parser.get("output"));
        }

        if (!outputDir.empty() && !std::filesystem::exists(outputDir)) {
            std::filesystem::create_directory(outputDir);
        }

        scenarioOptions.enablePipelineCaching = parser.get<bool>("--pipeline-caching");
        if (scenarioOptions.enablePipelineCaching) {
            scenarioOptions.clearPipelineCache = parser.get<bool>("--clear-pipeline-cache");
            scenarioOptions.failOnPipelineCacheMiss = parser.get<bool>("--fail-on-pipeline-cache-miss");
            auto cacheDir = std::filesystem::path(parser.get<std::string>("--cache-path"));
            if (!std::filesystem::is_directory(cacheDir)) {
                throw std::runtime_error("Invalid cache directory: " + cacheDir.string());
            }
            scenarioOptions.pipelineCachePath =
                cacheDir / std::filesystem::path(scenarioFile).filename().replace_extension("cache");
        }

        scenarioOptions.enableGPUDebugMarkers = parser.get<bool>("--enable-gpu-debug-markers");

        if (!scenarioOptions.disabledExtensions.empty()) {
            const auto &selectableExtensions = extensionList;
            for (auto &extension : scenarioOptions.disabledExtensions) {
                if (std::find(selectableExtensions.begin(), selectableExtensions.end(), extension) ==
                    selectableExtensions.end()) {
                    throw std::runtime_error("Unrecognized extension, cannot disable: " + extension);
                }
            }
        }

        if (parser.present("--session-memory-dump-dir")) {
            auto sessionRAMsDumpDir = parser.get("--session-memory-dump-dir");
            scenarioOptions.sessionRAMsDumpDir = std::filesystem::path(sessionRAMsDumpDir);
            if (!std::filesystem::is_directory(scenarioOptions.sessionRAMsDumpDir)) {
                throw std::runtime_error("Invalid Session Memory dump directory: " + sessionRAMsDumpDir);
            }
        }

        auto perfCountersPath = parser.get("--perf-counters-dump-path");
        if (!perfCountersPath.empty()) {
            scenarioOptions.perfCountersPath = std::filesystem::path(perfCountersPath);
            if (!std::ofstream(scenarioOptions.perfCountersPath)) {
                throw std::runtime_error("Unable to open perf counters file for writing " + perfCountersPath);
            }
        }

        auto profilingPath = parser.get("--profiling-dump-path");
        if (!profilingPath.empty()) {
            scenarioOptions.profilingPath = std::filesystem::path(profilingPath);
            if (!std::ofstream(scenarioOptions.profilingPath)) {
                throw std::runtime_error("Unable to open profiling data file for writing " + profilingPath);
            }
        }

        int repeatCount = parser.get<int>("--repeat");
        if (repeatCount <= 0) {
            throw std::runtime_error("Expected positive number for repeat");
        }

        bool dryRun = parser.get<bool>("--dry-run");
        if (dryRun && repeatCount > 1) {
            mlsdk::logging::warning("Count overruled by dry-run");
            repeatCount = 1;
        }

        bool captureFrame = parser.get<bool>("--capture-frame");
        if (dryRun && captureFrame) {
            mlsdk::logging::warning("Frame capture overruled by dry-run");
            captureFrame = false;
        }

        ScenarioSpec scenarioSpec(&fstream, workDir, outputDir);
        mlsdk::logging::info("Scenario file parsed");
        Scenario scenario(scenarioOptions, scenarioSpec);
        if (parser.get<bool>("--wait-for-key-stroke-before-run")) {
            mlsdk::logging::error("Press enter to continue...");
            std::ignore = getchar();
        }
        scenario.run(repeatCount, dryRun, captureFrame);

    } catch (const std::exception &err) {
        mlsdk::logging::error(err.what());
        return -1;
    }
    return 0;
}
