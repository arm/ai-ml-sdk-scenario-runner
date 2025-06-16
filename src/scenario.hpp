/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "commands.hpp"
#include "compute.hpp"
#include "context.hpp"
#include "data_manager.hpp"
#include "guid.hpp"
#include "json_writer.hpp"
#include "pipeline.hpp"
#include "resource_desc.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <fstream>
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

struct ScenarioSpec {
    ScenarioSpec(std::istream *is, const std::filesystem::path &workDir, const std::filesystem::path &outputDir = {});

    void addResource(std::unique_ptr<ResourceDesc> resource);

    void addCommand(std::unique_ptr<CommandDesc> command);

    std::vector<std::unique_ptr<ResourceDesc>> resources{};
    std::vector<std::unique_ptr<CommandDesc>> commands{};
    std::unordered_map<Guid, uint32_t> resourceRefs{};
    std::filesystem::path workDir{};
    std::filesystem::path outputDir{};
};

class Scenario {
  public:
    /// \brief Constructor
    Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec);

    /// \brief Executes the test case
    void run(int count = 1, bool dryRun = false, bool captureFrame = false);

  private:
    void createPipeline(const uint32_t segmentIndex, std::vector<BindingDesc> &sequenceBindings, const VgfView &vgfView,
                        DispatchDataGraphDesc &dispatchDataGraph, std::optional<PipelineCache> &_pipelineCache,
                        uint32_t &nQueries);

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
