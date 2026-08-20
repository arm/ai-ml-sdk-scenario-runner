/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "command_types.hpp"
#include "compute.hpp"
#include "context.hpp"
#include "data_manager.hpp"
#include "frame_capturer.hpp"
#include "group_manager.hpp"
#include "iscenario.hpp"
#include "resource_data.hpp"
#include "resource_manager.hpp"
#include "scenario_options.hpp"
#include "scenario_resource_io.hpp"
#include "types.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mlsdk::scenariorunner {
class ScenarioBuilder;
struct ScenarioSpec;
namespace detail {
struct ScenarioBuildData;
} // namespace detail

class Scenario : public IScenario {
  public:
    /// \brief Destructor
    ~Scenario() override = default;

    /// \brief Single execution of the scenario.
    void run() override;

    /// \brief Execute the scenario one or more times.
    void run(int repeatCount, bool dryRun) override;

    /// \brief Get a buffer resource ID from its UID
    BufferId getBufferId(std::string_view uid) const override;

    /// \brief Get an image resource ID from its UID
    ImageId getImageId(std::string_view uid) const override;

    /// \brief Get a tensor resource ID from its UID
    TensorId getTensorId(std::string_view uid) const override;

    /// \brief Upload data to an existing buffer resource
    void upload(BufferId id, const BufferDataView &data) override;

    /// \brief Upload data to an existing image resource
    void upload(ImageId id, const ImageDataView &data) override;

    /// \brief Upload data to an existing tensor resource
    void upload(TensorId id, const TensorDataView &data) override;

    /// \brief Download data from an existing buffer resource
    BufferData download(BufferId id) const override;

    /// \brief Download data from an existing image resource
    ImageData download(ImageId id) override;

    /// \brief Download data from an existing tensor resource
    TensorData download(TensorId id) const override;

  private:
    friend class ScenarioBuilder;

    Scenario(const ScenarioOptions &opts, detail::ScenarioBuildData buildData);

    void runIteration(int iteration, int repeatCount, bool dryRun);

    void createComputePipeline(const DispatchComputeData &dispatchCompute, uint32_t &nQueries);
    void createDataGraphPipeline(const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries);
    void createSpirvGraphPipeline(const DispatchSpirvGraphData &dispatchSpirvGraph, uint32_t &nQueries);
    void createFragmentPipeline(const DispatchFragmentData &dispatchFragment, uint32_t &nQueries);
    void createOpticalFlowPipeline(const DispatchOpticalFlowData &dispatchOpticalFlow, uint32_t &nQueries);

    void createPipeline(uint32_t segmentIndex, const std::vector<TypedBinding> &sequenceBindings,
                        const VgfView &vgfView, const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries);

    void initializeResourceData();
    void setupResources();
    void createRuntimeResources();
    void createRuntimeBarriers();
    void setupRuntimeCommands();

    /// \brief Save profiling data to file
    void saveProfilingData(int iteration, int repeatCount, bool dryRun);

    /// \brief Save results of output resources to files
    void saveResults(bool dryRun);

    /// \brief Reset transient execution state before another run
    void resetForNextRun();

    bool hasAliasedOptimalTensors() const;
    void handleAliasedLayoutTransitions();
    MemoryResourceId getMemoryResourceId(const Guid &guid) const;
    const ShaderInfo &getShader(ShaderId id) const;
    const ShaderInfo &getSubstitutionShader(const std::vector<ResolvedShaderSubstitution> &shaderSubstitutions,
                                            const std::string &moduleName) const;

    ScenarioOptions _opts;
    Context _ctx;
    ResourceManager _resources;
    std::unordered_map<Guid, TypedResourceId> _resourceIds;
    std::unordered_map<DataGraphId, VgfResourceCreationResult> _vgfResourceCreationResults;
    DataManager _dataManager;
    std::vector<detail::ResourceInitialization> _initializations;
    std::vector<detail::ResourceOutput> _outputs;
    std::vector<detail::ScenarioCommand> _commands;
    std::shared_ptr<PipelineCache> _pipelineCache;
    Compute _compute;
    std::vector<PerformanceCounter> _perfCounters;
    GroupManager _groupManager;
    std::unique_ptr<FrameCapturer> _frameCapturer;
    bool _hasRun{false};
};

} // namespace mlsdk::scenariorunner
