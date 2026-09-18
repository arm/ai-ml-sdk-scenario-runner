/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_runner/scenario_builder.hpp"

#include "scenario_build_data.hpp"

#include <memory>

namespace mlsdk::scenariorunner {

namespace detail {
class ScenarioBuilderAccess;
}

class ScenarioBuilderImpl final : public ScenarioBuilder {
  public:
    ScenarioBuilderImpl() = default;
    ~ScenarioBuilderImpl() override = default;

    ScenarioBuilderImpl(const ScenarioBuilderImpl &) = delete;
    ScenarioBuilderImpl &operator=(const ScenarioBuilderImpl &) = delete;
    ScenarioBuilderImpl(ScenarioBuilderImpl &&) = delete;
    ScenarioBuilderImpl &operator=(ScenarioBuilderImpl &&) = delete;

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
    VgfId addVgf(const VgfInfo &info) override;
    VgfId addVgf(VgfInfo &&info);
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
    void addDispatchVgf(DispatchVgfData command) override;
    void addDispatchDataGraph(DispatchDataGraphData command) override;
    void addDispatchOpticalFlow(DispatchOpticalFlowData command) override;
    void addPipelineBarrier(PipelineBarrierData command) override;
    void addFrameBoundary(FrameBoundaryData command) override;

    std::unique_ptr<Scenario> build(const ScenarioOptions &options) override;

  private:
    friend class detail::ScenarioBuilderAccess;

    void ensureMutable() const;
    void validateMemoryResource(MemoryResourceId resource) const;
    void validateBinding(const TypedBinding &binding) const;
    void validateGroup(MemoryGroupId group) const;

    detail::ScenarioBuildData _data;
    bool _built{};
};

namespace detail {
class ScenarioBuilderAccess {
  public:
    static ScenarioBuildData &buildData(ScenarioBuilderImpl &builder) { return builder._data; }
    static ScenarioBuildData takeBuildData(ScenarioBuilderImpl &builder);
};
} // namespace detail

} // namespace mlsdk::scenariorunner
