/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute.hpp"
#include "context.hpp"
#include "data_manager.hpp"
#include "group_manager.hpp"
#include "resource_manager.hpp"
#include "scenario_desc.hpp"
#include "types.hpp"

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

struct DispatchComputeData;
struct DispatchDataGraphData;
struct DispatchSpirvGraphData;
struct DispatchFragmentData;
struct DispatchOpticalFlowData;
struct ResolvedShaderSubstitution;

class Scenario {
  public:
    /// \brief Constructor
    Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec);

    /// \brief Executes the test case
    void run(int repeatCount = 1, bool dryRun = false);

  private:
    void createComputePipeline(const DispatchComputeData &dispatchCompute, uint32_t &nQueries);
    void createDataGraphPipeline(const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries);
    void createSpirvGraphPipeline(const DispatchSpirvGraphData &dispatchSpirvGraph, uint32_t &nQueries);
    void createFragmentPipeline(const DispatchFragmentData &dispatchFragment, uint32_t &nQueries);
    void createOpticalFlowPipeline(const DispatchOpticalFlowData &dispatchOpticalFlow, uint32_t &nQueries);

    void createPipeline(uint32_t segmentIndex, const std::vector<TypedBinding> &sequenceBindings,
                        const VgfView &vgfView, const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries);

    /// \brief Sets up runtime options
    void setupResources();
    void setupCommands();

    /// \brief Save profiling data to file
    void saveProfilingData(int iteration, int repeatCount, bool dryRun);

    /// \brief Save results of output resources to files
    void saveResults(bool dryRun);

    bool hasAliasedOptimalTensors() const;
    void handleAliasedLayoutTransitions();
    const ShaderInfo &getShader(ShaderId id) const;
    const ShaderInfo &getSubstitutionShader(const std::vector<ResolvedShaderSubstitution> &shaderSubstitutions,
                                            const std::string &moduleName) const;

    ScenarioOptions _opts;
    Context _ctx;
    ResourceManager _resources;
    std::unordered_map<Guid, ShaderId> _shaderIds;
    DataManager _dataManager;
    ScenarioSpec &_scenarioSpec;
    std::shared_ptr<PipelineCache> _pipelineCache;
    Compute _compute;
    std::vector<PerformanceCounter> _perfCounters;
    GroupManager _groupManager;
};

} // namespace mlsdk::scenariorunner
