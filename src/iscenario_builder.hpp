/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "command_types.hpp"
#include "iscenario.hpp"
#include "scenario_options.hpp"
#include "types.hpp"

#include <memory>

namespace mlsdk::scenariorunner {

struct ScenarioSpec;

/// @brief Public interface for defining and building a scenario.
///
/// Resource registration returns stable typed IDs. Callers retain these IDs
/// and use them to transfer data through the IScenario returned by build().
class IScenarioBuilder {
  public:
    virtual ~IScenarioBuilder() = default;

    virtual BufferId addBuffer(const BufferInfo &info) = 0;
    virtual ImageId addImage(const ImageInfo &info) = 0;
    virtual TensorId addTensor(const TensorInfo &info) = 0;
    virtual ShaderId addShader(const ShaderInfo &info) = 0;
    virtual RawDataId addRawData(const RawDataInfo &info) = 0;
    virtual DataGraphId addDataGraph(const DataGraphInfo &info) = 0;
    virtual GraphConstantResourceId addGraphConstant(const GraphConstantInfo &info) = 0;

    virtual ImageBarrierId addImageBarrier(const ImageBarrierInfo &info) = 0;
    virtual BufferBarrierId addBufferBarrier(const BufferBarrierInfo &info) = 0;
    virtual TensorBarrierId addTensorBarrier(const TensorBarrierInfo &info) = 0;
    virtual MemoryBarrierId addMemoryBarrier(const MemoryBarrierInfo &info) = 0;

    virtual MemoryGroupId createMemoryGroup() = 0;
    virtual void addResourceToMemoryGroup(MemoryGroupId group, MemoryResourceId resource) = 0;

    virtual void addDispatchCompute(DispatchComputeData command) = 0;
    virtual void addDispatchFragment(DispatchFragmentData command) = 0;
    virtual void addDispatchDataGraph(DispatchDataGraphData command) = 0;
    virtual void addDispatchSpirvGraph(DispatchSpirvGraphData command) = 0;
    virtual void addDispatchOpticalFlow(DispatchOpticalFlowData command) = 0;
    virtual void addDispatchBarrier(DispatchBarrierData command) = 0;
    virtual void addMarkBoundary(MarkBoundaryData command) = 0;

    /// @brief Consume the builder and return a ready-to-run scenario.
    virtual std::unique_ptr<IScenario> build(const ScenarioOptions &options) = 0;

    /// @brief Build the scenario described by a parsed scenario.json file.
    virtual std::unique_ptr<IScenario> build(const ScenarioOptions &options, ScenarioSpec &spec) = 0;
};

} // namespace mlsdk::scenariorunner
