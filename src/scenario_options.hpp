/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {

/// @brief Options that are passed for configuring a scenario.
struct ScenarioOptions {
    bool enablePipelineCaching{false};
    bool clearPipelineCache{false};
    bool failOnPipelineCacheMiss{false};
    bool enableGPUDebugMarkers{false};
    bool captureFrame{false};
    bool enableRobustnessFeatures{false};
    std::filesystem::path pipelineCachePath;
    std::filesystem::path neuralDebugDatabaseDumpDir;
    std::filesystem::path neuralStatisticsDumpDir;
    std::filesystem::path graphProfilingDumpDir;
    std::filesystem::path sessionRAMsDumpDir;
    std::filesystem::path perfCountersPath;
    std::filesystem::path profilingPath;
    std::vector<std::string> disabledExtensions;
    vk::NeuralAcceleratorStatisticsModeARM neuralStatisticsMode{};

    bool shouldDumpNeuralDebugDatabase() const { return !neuralDebugDatabaseDumpDir.empty(); }
    bool shouldDumpNeuralStatistics() const { return !neuralStatisticsDumpDir.empty(); }
    bool shouldDumpGraphProfiling() const { return !graphProfilingDumpDir.empty(); }
};

} // namespace mlsdk::scenariorunner
