/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/session.hpp"

#include <cstdint>
#include <filesystem>

namespace mlsdk::scenariorunner {

// Diagnostic dumps for a standalone workload's sole graph executable. Memory reads
// require the caller to wait for GPU completion before invoking these helpers.
// The following two helpers return whether dump data was available, so failed writes
// still consume a report index.
bool dumpWorkloadGraphDebugDatabase(const workloadlib::Session &session, const std::filesystem::path &path);
bool dumpWorkloadGraphStatistics(const workloadlib::Session &session, const vk::raii::Device &device,
                                 const std::filesystem::path &path);
void dumpWorkloadGraphProfiling(const workloadlib::Session &session, const std::filesystem::path &directory,
                                uint32_t graphPipelineIndex);
void dumpWorkloadGraphSessionRAMs(const workloadlib::Session &session, const vk::raii::Device &device,
                                  const std::filesystem::path &directory, uint32_t &graphPipelineIndex);

} // namespace mlsdk::scenariorunner
