/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute.hpp"
#include "context.hpp"
#include "data_manager.hpp"
#include "group_manager.hpp"
#include "scenario_desc.hpp"
#include "types.hpp"

#include <string>
#include <vector>

namespace mlsdk::scenariorunner {
/// \brief Options that are passed for configuring a scenario
struct ScenarioOptions {
    bool enablePipelineCaching{false};
    bool clearPipelineCache{false};
    bool failOnPipelineCacheMiss{false};
    bool enableGPUDebugMarkers{false};
    bool captureFrame{false};
    std::filesystem::path pipelineCachePath{};
    std::filesystem::path sessionRAMsDumpDir{};
    std::filesystem::path perfCountersPath{};
    std::filesystem::path profilingPath{};
    std::vector<std::string> disabledExtensions{};
};

struct DispatchComputeData;
struct DispatchDataGraphData;

class Scenario {
  public:
    /// \brief Constructor
    Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec);

    /// \brief Executes the test case
    void run(int repeatCount = 1, bool dryRun = false);

  private:
    void createComputePipeline(const DispatchComputeData &dispatchCompute, uint32_t &nQueries);
    void createDataGraphPipeline(const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries);

    void createPipeline(uint32_t segmentIndex, const std::vector<TypedBinding> &sequenceBindings,
                        const VgfView &vgfView, const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries);

    /// \brief Sets up runtime options
    void setupResources();
    void setupCommands();

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
    std::shared_ptr<PipelineCache> _pipelineCache{};
    Compute _compute;
    std::vector<PerformanceCounter> _perfCounters{};
    GroupManager _groupManager;
};

} // namespace mlsdk::scenariorunner
