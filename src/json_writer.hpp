/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "perf_counter.hpp"

namespace mlsdk::scenariorunner {

using json = nlohmann::json;

struct CommandTimestamps {
    CommandTimestamps() = default;
    CommandTimestamps(const std::string &commandType, const std::vector<uint64_t> &commandTimestamps,
                      const float timestampPeriod, const int iteration = 1)
        : type(commandType), timestamps(commandTimestamps), period(timestampPeriod), iteration(iteration) {}

    std::string type = {};
    std::vector<uint64_t> timestamps = {};
    float period = {};
    int iteration;
};

void writePerfCounters(std::vector<PerformanceCounter> &perfCounters, std::filesystem::path &path);

void writeProfilingData(const std::vector<uint64_t> &timestamps, const float timestampPeriod,
                        const std::vector<std::string> &profiledCommands, const std::filesystem::path &path,
                        const int &iteration, const int &repeatCount);

// Serialize PerformanceCounter to JSON
void to_json(json &j, const PerformanceCounter &perfCounter);

// Serialize CommandTimestamps to JSON
void to_json(json &j, const CommandTimestamps &commandTimestamps);
} // namespace mlsdk::scenariorunner
