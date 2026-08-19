/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "command_types.hpp"
#include "group_manager.hpp"
#include "iscenario_builder.hpp"
#include "resource_manager.hpp"

namespace mlsdk::scenariorunner {

namespace detail {
struct ScenarioBuildData {
    ResourceManager resources;
    GroupManager groupManager;
    std::vector<ScenarioCommand> commands;
    bool useComputeFamilyQueue{};
    bool requiresGraphicsFamilyQueue{};
};
} // namespace detail

class ScenarioBuilder : public IScenarioBuilder {
  public:
    ~ScenarioBuilder() override = default;

    BufferId addBuffer(const BufferInfo &info) override;
    BufferId addBuffer(BufferInfo &&info);
    ImageId addImage(const ImageInfo &info) override;
    ImageId addImage(ImageInfo &&info);
    TensorId addTensor(const TensorInfo &info) override;
    TensorId addTensor(TensorInfo &&info);
    ShaderId addShader(const ShaderInfo &info) override;
    ShaderId addShader(ShaderInfo &&info);
    RawDataId addRawData(const RawDataInfo &info) override;
    RawDataId addRawData(RawDataInfo &&info);
    DataGraphId addDataGraph(const DataGraphInfo &info) override;
    DataGraphId addDataGraph(DataGraphInfo &&info);
    GraphConstantResourceId addGraphConstant(const GraphConstantInfo &info) override;
    GraphConstantResourceId addGraphConstant(GraphConstantInfo &&info);

    ImageBarrierId addImageBarrier(const ImageBarrierInfo &info) override;
    BufferBarrierId addBufferBarrier(const BufferBarrierInfo &info) override;
    TensorBarrierId addTensorBarrier(const TensorBarrierInfo &info) override;
    MemoryBarrierId addMemoryBarrier(const MemoryBarrierInfo &info) override;

    MemoryGroupId createMemoryGroup() override;
    void addResourceToMemoryGroup(MemoryGroupId group, MemoryResourceId resource) override;

    void addDispatchCompute(DispatchComputeData command) override;
    void addDispatchFragment(DispatchFragmentData command) override;
    void addDispatchDataGraph(DispatchDataGraphData command) override;
    void addDispatchSpirvGraph(DispatchSpirvGraphData command) override;
    void addDispatchOpticalFlow(DispatchOpticalFlowData command) override;
    void addDispatchBarrier(DispatchBarrierData command) override;
    void addMarkBoundary(MarkBoundaryData command) override;

    std::unique_ptr<IScenario> build(const ScenarioOptions &options) override;
    std::unique_ptr<IScenario> build(const ScenarioOptions &options, ScenarioSpec &spec) override;

  private:
    friend class Scenario;

    detail::ScenarioBuildData takeBuildData();
    void ensureMutable() const;
    void validateMemoryResource(MemoryResourceId resource) const;
    void validateBinding(const TypedBinding &binding) const;
    void validateGroup(MemoryGroupId group) const;

    detail::ScenarioBuildData _data;
    bool _built{};
};

} // namespace mlsdk::scenariorunner
