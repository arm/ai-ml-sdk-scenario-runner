/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "perf_counter.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {

void writePerfCounters(std::vector<PerformanceCounter> &perfCounters, std::filesystem::path &path);

void writeProfilingData(const std::vector<uint64_t> &timestamps, const float timestampPeriod,
                        const std::vector<std::string> &profiledCommands, const std::filesystem::path &path,
                        int iteration, int repeatCount);
} // namespace mlsdk::scenariorunner
