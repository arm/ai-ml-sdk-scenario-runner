/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "perf_counter.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {

struct ProfiledCommand {
    std::string type;
    std::string name;
};

struct ProfiledMemoryUsage {
    std::string commandName;
    /// Sum of graph-session allocation sizes, including neural statistics.
    /// Excludes bound resources and workload intermediates; independent of repeats.
    uint64_t sessionMemoryBytes{};
};

struct RuntimeProfilingData {
    std::vector<uint64_t> timestamps;
    float timestampPeriod{};
    std::vector<ProfiledCommand> commands;
};

struct MemoryProfilingData {
    std::vector<ProfiledMemoryUsage> usages;
};

void writePerfCounters(std::vector<PerformanceCounter> &perfCounters, std::filesystem::path &path);

void writeProfilingData(const std::optional<RuntimeProfilingData> &runtimeProfilingData,
                        const MemoryProfilingData &memoryProfilingData, const std::filesystem::path &path,
                        int iteration, int repeatCount);
} // namespace mlsdk::scenariorunner
