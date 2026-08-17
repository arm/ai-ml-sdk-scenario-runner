/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "command_types.hpp"
#include "group_manager.hpp"
#include "resource_manager.hpp"

namespace mlsdk::scenariorunner {

namespace detail {
struct ScenarioBuildData {
    ResourceManager resources;
    GroupManager groupManager;
    std::vector<ScenarioCommand> commands;
};
} // namespace detail

class ScenarioBuilder {
  public:
    BufferId addBuffer(const BufferInfo &info);
    BufferId addBuffer(BufferInfo &&info);
    ImageId addImage(const ImageInfo &info);
    ImageId addImage(ImageInfo &&info);
    TensorId addTensor(const TensorInfo &info);
    TensorId addTensor(TensorInfo &&info);
    ShaderId addShader(const ShaderInfo &info);
    ShaderId addShader(ShaderInfo &&info);
    RawDataId addRawData(const RawDataInfo &info);
    RawDataId addRawData(RawDataInfo &&info);
    DataGraphId addDataGraph(const DataGraphInfo &info);
    DataGraphId addDataGraph(DataGraphInfo &&info);
    GraphConstantResourceId addGraphConstant(const GraphConstantInfo &info);
    GraphConstantResourceId addGraphConstant(GraphConstantInfo &&info);

    ImageBarrierId addImageBarrier(const ImageBarrierInfo &info);
    BufferBarrierId addBufferBarrier(const BufferBarrierInfo &info);
    TensorBarrierId addTensorBarrier(const TensorBarrierInfo &info);
    MemoryBarrierId addMemoryBarrier(const MemoryBarrierInfo &info);

    MemoryGroupId createMemoryGroup();
    void addResourceToMemoryGroup(MemoryGroupId group, MemoryResourceId resource);

    void addDispatchCompute(DispatchComputeData command);
    void addDispatchFragment(DispatchFragmentData command);
    void addDispatchDataGraph(DispatchDataGraphData command);
    void addDispatchSpirvGraph(DispatchSpirvGraphData command);
    void addDispatchOpticalFlow(DispatchOpticalFlowData command);
    void addDispatchBarrier(DispatchBarrierData command);
    void addMarkBoundary(MarkBoundaryData command);

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
