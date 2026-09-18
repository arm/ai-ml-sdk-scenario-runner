/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "command_types.hpp"
#include "scenario.hpp"
#include "scenario_options.hpp"
#include "types.hpp"

#include <memory>

namespace mlsdk::scenariorunner {

/// @brief Public interface for defining and building a scenario.
///
/// Resource registration returns stable typed IDs. Callers retain these IDs
/// and use them to transfer data through the Scenario returned by build().
/// The builder copies each resource description and consumes command values.
/// Calling build() transfers ownership of the complete definition to the
/// returned Scenario and leaves the builder immutable.
class ScenarioBuilder {
  public:
    /// @brief Destroy the builder.
    virtual ~ScenarioBuilder() = default;

    /// @brief Register a buffer resource.
    /// @param info Buffer description to copy into the scenario.
    /// @return The ID used to refer to the buffer in commands and data transfers.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual BufferId addBuffer(const BufferInfo &info) = 0;
    /// @brief Register an image resource.
    /// @param info Image description to copy into the scenario.
    /// @return The ID used to refer to the image in commands and data transfers.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual ImageId addImage(const ImageInfo &info) = 0;
    /// @brief Register a tensor resource.
    /// @param info Tensor description to copy into the scenario.
    /// @return The ID used to refer to the tensor in commands and data transfers.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual TensorId addTensor(const TensorInfo &info) = 0;
    /// @brief Register a shader resource.
    /// @param info Shader description to copy into the scenario.
    /// @return The ID used to refer to the shader in commands.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual ShaderId addShader(const ShaderInfo &info) = 0;
    /// @brief Register raw command data.
    /// @param info Raw-data description to copy into the scenario.
    /// @return The ID used to refer to the data in commands.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual RawDataId addRawData(const RawDataInfo &info) = 0;
    /// @brief Register a VGF resource.
    /// @param info VGF description to copy into the scenario.
    /// @return The ID used to refer to the VGF in commands.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual VgfId addVgf(const VgfInfo &info) = 0;
    /// @brief Register a graph constant resource.
    /// @param info Graph-constant description to copy into the scenario.
    /// @return The ID used to refer to the constant in graph commands.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual GraphConstantResourceId addGraphConstant(const GraphConstantInfo &info) = 0;

    /// @brief Register an image barrier.
    /// @param info Barrier description containing a previously registered image ID.
    /// @return The ID used to refer to the barrier in a barrier command.
    /// @throws std::runtime_error If the image ID is invalid or the builder was consumed.
    virtual ImageBarrierId addImageBarrier(const ImageBarrierInfo &info) = 0;
    /// @brief Register a buffer barrier.
    /// @param info Barrier description containing a previously registered buffer ID.
    /// @return The ID used to refer to the barrier in a barrier command.
    /// @throws std::runtime_error If the buffer ID is invalid or the builder was consumed.
    virtual BufferBarrierId addBufferBarrier(const BufferBarrierInfo &info) = 0;
    /// @brief Register a tensor barrier.
    /// @param info Barrier description containing a previously registered tensor ID.
    /// @return The ID used to refer to the barrier in a barrier command.
    /// @throws std::runtime_error If the tensor ID is invalid or the builder was consumed.
    virtual TensorBarrierId addTensorBarrier(const TensorBarrierInfo &info) = 0;
    /// @brief Register a global memory barrier.
    /// @param info Barrier description to copy into the scenario.
    /// @return The ID used to refer to the barrier in a barrier command.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual MemoryBarrierId addMemoryBarrier(const MemoryBarrierInfo &info) = 0;

    /// @brief Create an empty memory aliasing group.
    /// @return The ID of the new memory group.
    /// @throws std::runtime_error If the builder has already been consumed.
    virtual MemoryGroupId createMemoryGroup() = 0;
    /// @brief Add a buffer, image, or tensor to a memory aliasing group.
    /// @param group Existing memory group.
    /// @param resource Previously registered memory resource.
    /// @throws std::runtime_error If either ID is invalid or the resource is already in another group.
    virtual void addResourceToMemoryGroup(MemoryGroupId group, MemoryResourceId resource) = 0;

    /// @brief Append a compute dispatch to the execution sequence.
    /// @param command Command whose referenced resources have already been registered.
    /// @throws std::runtime_error If a resource ID is invalid or the builder was consumed.
    virtual void addDispatchCompute(DispatchComputeData command) = 0;
    /// @brief Append a fragment dispatch to the execution sequence.
    /// @param command Command whose referenced resources have already been registered.
    /// @throws std::runtime_error If a resource ID is invalid or the builder was consumed.
    virtual void addDispatchFragment(DispatchFragmentData command) = 0;
    /// @brief Append a data graph dispatch to the execution sequence.
    /// @param command Command whose referenced resources have already been registered.
    /// @throws std::runtime_error If a resource ID is invalid or the builder was consumed.
    virtual void addDispatchVgf(DispatchVgfData command) = 0;
    /// @brief Append a SPIR-V™ graph dispatch to the execution sequence.
    /// @param command Command whose referenced resources have already been registered.
    /// @throws std::runtime_error If a resource ID is invalid or the builder was consumed.
    virtual void addDispatchDataGraph(DispatchDataGraphData command) = 0;
    /// @brief Append an optical flow dispatch to the execution sequence.
    /// @param command Command whose referenced resources have already been registered.
    /// @throws std::runtime_error If a resource ID is invalid or the builder was consumed.
    virtual void addDispatchOpticalFlow(DispatchOpticalFlowData command) = 0;
    /// @brief Append a set of resource barriers to the execution sequence.
    /// @param command Command whose barrier IDs have already been registered.
    /// @throws std::runtime_error If a barrier ID is invalid or the builder was consumed.
    virtual void addPipelineBarrier(PipelineBarrierData command) = 0;
    /// @brief Append a profiling boundary marker to the execution sequence.
    /// @param command Boundary whose resource IDs have already been registered.
    /// @throws std::runtime_error If a resource ID is invalid or the builder was consumed.
    virtual void addFrameBoundary(FrameBoundaryData command) = 0;

    /// @brief Consume the builder and return a ready-to-run scenario.
    /// @param options Runtime and diagnostic options for the scenario.
    /// @return A scenario that owns the registered resources and commands.
    /// @throws std::runtime_error If the scenario definition is invalid or the builder was already consumed.
    virtual std::unique_ptr<Scenario> build(const ScenarioOptions &options) = 0;
};

/// @brief Create an empty scenario builder.
/// @return A new builder with no resources or commands.
std::unique_ptr<ScenarioBuilder> createScenarioBuilder();

} // namespace mlsdk::scenariorunner
