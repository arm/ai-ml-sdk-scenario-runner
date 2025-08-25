/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "context.hpp"
#include "data_manager.hpp"
#include "perf_counter.hpp"
#include "pipeline.hpp"

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace mlsdk::scenariorunner {

/// @brief Compute command orchestrator
///
/// @note At the moment each compute handles a single shader
/// @note Not all Vulkan® implementations support push_descriptors hence we
/// register the Commands into a vector and on submission we construct the
/// command buffer and execute.
///
/// Acts as a mechanism to register pipelines or other commands to a command
/// buffer for execution.
/// Handles all scheduling and synchronization
class Compute {
  public:
    /// \brief Constructor
    ///
    /// \param ctx Device related context
    explicit Compute(Context &ctx);

    /// \brief Reset and setup resources
    void reset();

    /// \brief Register a pipeline for execution with a fence synchronization
    /// in the end
    ///
    /// \param pipeline Pipeline to register
    /// \param dataManager Data manager object to retrieve resource
    /// \param bindings List of bindings associated to each resource
    /// \param pushConstantData Pointer to push constant data to set for the pipeline
    /// \param pushConstantSize Size of push constants data in bytes
    /// \param implicitBarriers True to enable implicit barriers
    /// \param computeDispatch (Optional) Workgroup count across dimension X, Y and Z. Defaults to: 1, 1 and 1.
    void registerPipelineFenced(const Pipeline &pipeline, const DataManager &dataManager,
                                const std::vector<TypedBinding> &bindings, const char *pushConstantData,
                                size_t pushConstantSize, bool implicitBarriers, ComputeDispatch computeDispatch = {});

    /// \brief Register a timestamp query
    /// \param query Index of the query
    /// \param flag type of stage
    void registerWriteTimestamp(uint32_t query, vk::PipelineStageFlagBits2 flag);

    /// \brief Register a pipeline barrier for execution
    /// \param dispatchBarrierData image barrier dispatch descriptor
    /// \param dataManager Data manager object to retrieve resource
    void registerPipelineBarrier(const DispatchBarrierData &dispatchBarrierData, const DataManager &dataManager);

    /// \brief Submit the command buffer for execution and wait for completion
    void submitAndWaitOnFence();
    void submitAndWaitOnFence(std::vector<PerformanceCounter> &perfCounters, int iteration);

    /// \brief Setup a query pool
    /// \param nQueries Number of queries to register
    void setupQueryPool(uint32_t nQueries);

    /// \brief Create the VkFrameBoundaryEXT struct with the correct resource
    /// \param markBoundaryData MarkBoundary object
    /// \param dataManager Data manager object to retrieve resource
    void registerMarkBoundary(const MarkBoundaryData &markBoundaryData, const DataManager &dataManager);

    vk::raii::CommandBuffer &getCommandBuffer();
    void prepareCommandBuffer();

    /// \brief Write profiling data to file
    void writeProfilingFile(const std::filesystem::path &profilingPath, int iteration, int repeatCount) const;

  private:
    enum class BindPoint {
        Compute,
        DataGraph,
    };

    struct BindDescriptorSet {
        vk::PipelineLayout pipelineLayout{nullptr};
        uint32_t descriptorSetIdxGlobal;
        uint32_t descriptorSetId;
        BindPoint bindPoint;
    };

    struct BindPipeline {
        vk::Pipeline pipeline{nullptr};
        BindPoint bindPoint;
    };

    struct DataGraphDispatch {
        vk::DataGraphPipelineSessionARM session{nullptr};
    };

    struct MemoryBarrier {
        size_t memoryBarrierIdx;
        size_t imageBarrierIdx;
        size_t tensorBarrierIdx;
        size_t bufferBarrierIdx;
    };

    struct PushConstants {
        vk::PipelineLayout pipelineLayout{nullptr};
        const char *pushConstantData;
        uint32_t size;
    };

    struct WriteTimestamp {
        uint32_t query;
        vk::PipelineStageFlagBits2 flag;
    };

    struct MarkBoundary {
        vk::FrameBoundaryEXT markBoundary;
    };

    struct PushDebugMarker {
        size_t nameIdx;
    };

    struct PopDebugMarker {};

    using Command = std::variant<BindDescriptorSet, BindPipeline, ComputeDispatch, DataGraphDispatch, MemoryBarrier,
                                 PushConstants, WriteTimestamp, MarkBoundary, PushDebugMarker, PopDebugMarker>;

    struct DebugMarker;

    /// \brief Setup
    void _setup();

    void _setNextCommandBuffer();
    void _beginCommandBuffer();

    void _waitForFence();

    /// \brief Fetch the QueryPoolResults, which contain runtime cycle-timestamps used for profiling
    std::vector<uint64_t> _queryTimestamps() const;

    friend struct DebugMarker;

    Context &_ctx;
    vk::raii::CommandPool _cmdPool{nullptr};
    vk::raii::Queue _queue{nullptr};
    vk::raii::Fence _fence{nullptr};

    std::vector<vk::raii::DescriptorPool> _descriptorPools{};
    std::vector<vk::raii::DescriptorSet> _descriptorSets{};
    std::vector<std::vector<vk::MemoryBarrier2>> _memoryBarriers{};
    std::vector<std::vector<vk::TensorMemoryBarrierARM>> _tensorBarriers{};
    std::vector<std::vector<vk::ImageMemoryBarrier2>> _imageBarriers{};
    std::vector<std::vector<vk::BufferMemoryBarrier2>> _bufferBarriers{};
    std::vector<std::vector<vk::Image>> _imageArray{};
    std::vector<std::vector<vk::Buffer>> _bufferArray{};
    std::vector<std::vector<vk::TensorARM>> _tensorArray{};
    std::vector<std::unique_ptr<vk::FrameBoundaryTensorsARM>> _markBoundaryTensorArray;
    std::vector<Command> _commands{};
    std::vector<vk::raii::CommandBuffer> _cmdBufferArray{};
    std::vector<std::string> _debugMarkerNames{};

    vk::raii::QueryPool _queryPool{nullptr};
    uint32_t _nQueries{0};
#ifdef ML_SDK_ENABLE_RDOC
    bool _isRecording{false};
#endif
};
} // namespace mlsdk::scenariorunner
