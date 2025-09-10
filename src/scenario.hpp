/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute.hpp"
#include "context.hpp"
#include "data_manager.hpp"
#include "guid.hpp"
#include "pipeline.hpp"
#include "scenario_desc.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlsdk::scenariorunner {
/// \brief Options that are passed for configuring a scenario
struct ScenarioOptions {
    bool enablePipelineCaching{false};
    bool clearPipelineCache{false};
    bool failOnPipelineCacheMiss{false};
    bool enableGPUDebugMarkers{false};
    std::filesystem::path pipelineCachePath{};
    std::filesystem::path sessionRAMsDumpDir{};
    std::filesystem::path perfCountersPath{};
    std::filesystem::path profilingPath{};
    std::vector<std::string> disabledExtensions{};
};

class Scenario {
  public:
    /// \brief Constructor
    Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec);

    /// \brief Executes the test case
    void run(int count = 1, bool dryRun = false, bool captureFrame = false);

  private:
    void createComputePipeline(const DispatchComputeData &dispatchCompute, int iteration, uint32_t &nQueries);
    void createDataGraphPipeline(const DispatchDataGraphDesc &dispatchDataGraph, int iteration, uint32_t &nQueries);

    void createPipeline(uint32_t segmentIndex, const std::vector<TypedBinding> &sequenceBindings,
                        const VgfView &vgfView, const DispatchDataGraphDesc &dispatchDataGraph, uint32_t &nQueries);

    /// \brief Sets up runtime options
    void setupResources();
    void setupCommands(int iteration = 0);

    /// \brief Save profiling data to file
    void saveProfilingData(int iteration, int repeatCount);

    /// \brief Save results of output resources to files
    void saveResults(bool dryRun);

    bool hasAliasedOptimalTensors() const;
    void handleAliasedLayoutTransitions();
    ScenarioOptions _opts;
    Context _ctx;
    DataManager _dataManager;
    ScenarioSpec &_scenarioSpec;
    std::list<Pipeline> _pipelines{};
    std::optional<PipelineCache> _pipelineCache{};
    Compute _compute;
    std::vector<PerformanceCounter> _perfCounters{};
};

} // namespace mlsdk::scenariorunner
