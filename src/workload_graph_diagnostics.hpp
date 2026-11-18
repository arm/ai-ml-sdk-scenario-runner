/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/session.hpp"

#include <cstdint>
#include <filesystem>

namespace mlsdk::scenariorunner {

// Diagnostic dumps for one graph executable in a workload. Memory reads
// require the caller to wait for GPU completion before invoking these helpers.
// The following two helpers return whether dump data was available, so failed writes
// still consume a report index. Executable indices identify workload segments;
// graphPipelineIndex is a separate output-file counter.
bool dumpWorkloadGraphDebugDatabase(const workloadlib::Session &session, uint32_t executableIndex,
                                    const std::filesystem::path &path);
bool dumpWorkloadGraphStatistics(const workloadlib::Session &session, uint32_t executableIndex,
                                 const vk::raii::Device &device, const std::filesystem::path &path);
void dumpWorkloadGraphProfiling(const workloadlib::Session &session, uint32_t executableIndex,
                                const std::filesystem::path &directory, uint32_t graphPipelineIndex);
void dumpWorkloadGraphSessionRAMs(const workloadlib::Session &session, uint32_t executableIndex,
                                  const vk::raii::Device &device, const std::filesystem::path &directory,
                                  uint32_t &graphPipelineIndex);

} // namespace mlsdk::scenariorunner
