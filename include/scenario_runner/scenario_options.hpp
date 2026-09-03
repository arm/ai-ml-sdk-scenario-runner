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
    /// @brief Enable Vulkan® pipeline caching.
    bool enablePipelineCaching{false};
    /// @brief Remove existing pipeline cache contents before setup.
    bool clearPipelineCache{false};
    /// @brief Fail setup when a required pipeline is absent from the cache.
    bool failOnPipelineCacheMiss{false};
    /// @brief Emit GPU debug markers when supported.
    bool enableGPUDebugMarkers{false};
    /// @brief Capture a frame using the configured capture integration.
    bool captureFrame{false};
    /// @brief Enable supported Vulkan® robustness features.
    bool enableRobustnessFeatures{false};
    /// @brief Pipeline cache input and output path.
    std::filesystem::path pipelineCachePath;
    /// @brief Directory for neural debug database output.
    std::filesystem::path neuralDebugDatabaseDumpDir;
    /// @brief Directory for neural accelerator statistics output.
    std::filesystem::path neuralStatisticsDumpDir;
    /// @brief Directory for data graph profiling output.
    std::filesystem::path graphProfilingDumpDir;
    /// @brief Directory for session RAM output.
    std::filesystem::path sessionRAMsDumpDir;
    /// @brief Performance counter configuration path.
    std::filesystem::path perfCountersPath;
    /// @brief Scenario profiling output path.
    std::filesystem::path profilingPath;
    /// @brief Vulkan® device extensions to leave disabled.
    std::vector<std::string> disabledExtensions;
    /// @brief Neural accelerator statistics collection mode.
    vk::NeuralAcceleratorStatisticsModeARM neuralStatisticsMode{};

    /// @brief Return whether neural debug database output is enabled.
    bool shouldDumpNeuralDebugDatabase() const { return !neuralDebugDatabaseDumpDir.empty(); }
    /// @brief Return whether neural accelerator statistics output is enabled.
    bool shouldDumpNeuralStatistics() const { return !neuralStatisticsDumpDir.empty(); }
    /// @brief Return whether data graph profiling output is enabled.
    bool shouldDumpGraphProfiling() const { return !graphProfilingDumpDir.empty(); }
};

} // namespace mlsdk::scenariorunner
