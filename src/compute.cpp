/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "compute.hpp"
#include "json_writer.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include <type_traits>

namespace mlsdk::scenariorunner {

namespace {

struct PipelineBarrierDebugNameBuilder {
    template <class BarrierType> void addBarrier(const BarrierType &barrier) {
        if (!barrier.debugName().empty()) {
            _value += barrier.debugName() + ",";
        }
    }

    std::string buildDebugName() const {
        if (!_value.empty()) {
            return "barriers (" + _value.substr(0, _value.size() - 1) + ")";
        } else {
            return "";
        }
    }

  private:
    std::string _value;
};

std::vector<vk::DescriptorPoolSize> getPoolSizes(const std::vector<TypedBinding> &bindings) {
    uint32_t numBuffers = 0;
    uint32_t numTensors = 0;
    uint32_t numSampledImages = 0;
    uint32_t numImages = 0;

    for (const auto &binding : bindings) {
        switch (binding.vkDescriptorType) {
        case vk::DescriptorType::eStorageBuffer:
            numBuffers++;
            break;
        case vk::DescriptorType::eTensorARM:
            numTensors++;
            break;
        case vk::DescriptorType::eCombinedImageSampler:
            numSampledImages++;
            break;
        case vk::DescriptorType::eStorageImage:
            numImages++;
            break;
        default:
            throw std::runtime_error("Cannot count unsupported descriptor type");
        }
    }
    std::vector<vk::DescriptorPoolSize> poolSizes;

    if (numBuffers) {
        poolSizes.push_back({vk::DescriptorType::eStorageBuffer, numBuffers});
    }
    if (numTensors) {
        poolSizes.push_back({vk::DescriptorType::eTensorARM, numTensors});
    }
    if (numSampledImages) {
        poolSizes.push_back({vk::DescriptorType::eCombinedImageSampler, numSampledImages});
    }
    if (numImages) {
        poolSizes.push_back({vk::DescriptorType::eStorageImage, numImages});
    }

    return poolSizes;
}

} // namespace

struct Compute::DebugMarker {
    DebugMarker(Compute *compute, const std::string &name);
    ~DebugMarker();

  private:
    Compute *_compute;
};

Compute::Compute(Context &ctx) : _ctx(ctx) {
    const vk::CommandPoolCreateInfo cmdPoolCreateInfo({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                                      _ctx.familyQueueIdx());
    _cmdPool = _ctx.device().createCommandPool(cmdPoolCreateInfo);
    setup();
}

void Compute::setup() {
    _queue = _ctx.device().getQueue(_ctx.familyQueueIdx(), 0);
    _fence = _ctx.device().createFence({});
}

void Compute::reset() {
    _cmdBufferArray.clear();
    _commands.clear();
}

void Compute::_setNextCommandBuffer() {
    const vk::CommandBufferAllocateInfo cmdBufferAllocInfo(*_cmdPool, vk::CommandBufferLevel::ePrimary, 1);
    _cmdBufferArray.emplace_back(std::move(_ctx.device().allocateCommandBuffers(cmdBufferAllocInfo).front()));
}
void Compute::prepareCommandBuffer() {
    if (_cmdBufferArray.empty()) {
        _setNextCommandBuffer();
    }
    vk::CommandBufferBeginInfo beginInfo{};
    _cmdBufferArray.back().begin(beginInfo);
}

vk::raii::CommandBuffer &Compute::getCommandBuffer() {
    if (_cmdBufferArray.empty()) {
        throw std::runtime_error("Command buffer not initialized");
    }
    return _cmdBufferArray.back();
}

void Compute::_waitForFence() {
    mlsdk::logging::info("Wait for fence");
    const uint64_t timeout = static_cast<uint64_t>(-1);
    auto res = _ctx.device().waitForFences({*_fence}, true, timeout);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Error while waiting for fence.");
    }
}

Compute::DebugMarker::DebugMarker(Compute *compute, const std::string &name) : _compute(compute) {
    if (!_compute->_ctx.gpuDebugMarkersEnabled()) {
        return;
    }

    _compute->_commands.emplace_back(PushDebugMarker{_compute->_debugMarkerNames.size()});
    _compute->_debugMarkerNames.emplace_back(name);
}

Compute::DebugMarker::~DebugMarker() {
    if (!_compute->_ctx.gpuDebugMarkersEnabled()) {
        return;
    }

    _compute->_commands.emplace_back(PopDebugMarker{});
}

void Compute::registerPipelineFenced(const Pipeline &pipeline, const DataManager &dataManager,
                                     const std::vector<TypedBinding> &bindings, const char *pushConstantData,
                                     size_t pushConstantSize, bool implicitBarriers, uint32_t wgcx, uint32_t wgcy,
                                     uint32_t wgcz) {

    DebugMarker dbgMrk0(this, "dispatch (" + pipeline.debugName() + ")");

    // Count exact number of typed resources used by this pipeline
    std::vector<vk::DescriptorPoolSize> poolSizes = getPoolSizes(bindings);

    // Populate descriptor sets
    const uint32_t baseDescriptorSetIdxGlobal = static_cast<uint32_t>(_descriptorSets.size());
    uint32_t maxSet = 0;
    for (const auto &binding : bindings) {
        maxSet = maxSet < binding.set ? binding.set : maxSet;

        // Add new sets as needed
        while (_descriptorSets.size() <= baseDescriptorSetIdxGlobal + binding.set) {
            const vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
            _descriptorPools.emplace_back(_ctx.device(), descriptorPoolCreateInfo);

            const vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*_descriptorPools.back(),
                                                                          pipeline.descriptorSetLayout(binding.set));

            _descriptorSets.push_back(
                std::move(vk::raii::DescriptorSets(_ctx.device(), descriptorSetAllocateInfo).front()));
        }

        const vk::DescriptorSet &descSet = *_descriptorSets[baseDescriptorSetIdxGlobal + binding.set];
        if (dataManager.hasBuffer(binding.resourceRef)) {
            const vk::Buffer &buf = dataManager.getBuffer(binding.resourceRef).buffer();
            const vk::DescriptorBufferInfo info(buf, 0, vk::WholeSize);
            vk::WriteDescriptorSet dwrite(descSet, static_cast<uint32_t>(binding.id), 0, 1, binding.vkDescriptorType,
                                          {}, &info);
            _ctx.device().updateDescriptorSets(vk::ArrayProxy<vk::WriteDescriptorSet>(dwrite), {});
        } else if (dataManager.hasTensor(binding.resourceRef)) {
            const vk::WriteDescriptorSetTensorARM info(1, &dataManager.getTensor(binding.resourceRef).tensorView());
            const vk::WriteDescriptorSet dwrite(descSet, static_cast<uint32_t>(binding.id), 0, 1,
                                                binding.vkDescriptorType, {}, {}, {}, &info);
            _ctx.device().updateDescriptorSets(vk::ArrayProxy<vk::WriteDescriptorSet>(dwrite), {});
        } else if (dataManager.hasImage(binding.resourceRef)) {
            vk::ImageView imageView;
            const Image &image = dataManager.getImage(binding.resourceRef);

            if (binding.lod.has_value()) {
                imageView = image.imageView(binding.lod.value());
            } else {
                imageView = image.imageView();
            }
            const vk::DescriptorImageInfo info(image.sampler(), imageView, image.getImageLayout());
            const vk::WriteDescriptorSet dwrite(descSet, static_cast<uint32_t>(binding.id), 0, 1,
                                                binding.vkDescriptorType, &info);
            _ctx.device().updateDescriptorSets(vk::ArrayProxy<vk::WriteDescriptorSet>(dwrite), {});
        }
    }

    const auto bindPoint = pipeline.isDataGraphPipeline() ? BindPoint::DataGraph : BindPoint::Compute;

    _commands.emplace_back(BindPipeline{pipeline.pipeline(), bindPoint});

    for (uint32_t setId = 0; setId <= maxSet; setId++) {
        _commands.emplace_back(
            BindDescriptorSet{pipeline.pipelineLayout(), baseDescriptorSetIdxGlobal + setId, setId, bindPoint});
    }

    if (pushConstantData != nullptr) {
        _commands.emplace_back(
            PushConstants{pipeline.pipelineLayout(), pushConstantData, static_cast<uint32_t>(pushConstantSize)});
    }
    if (pipeline.isDataGraphPipeline()) {
        _commands.emplace_back(DataGraphDispatch{pipeline.session()});
    } else {
        _commands.emplace_back(ComputeDispatch{wgcx, wgcy, wgcz});
    }

    if (implicitBarriers) {
        // Set an implicit memory barrier
        const uint32_t memoryBarrierIdx = static_cast<uint32_t>(_memoryBarriers.size());
        const uint32_t imageBarrierIdx = static_cast<uint32_t>(_imageBarriers.size());
        const uint32_t tensorBarrierIdx = static_cast<uint32_t>(_tensorBarriers.size());
        const uint32_t bufferBarrierIdx = static_cast<uint32_t>(_bufferBarriers.size());

        auto accessFlag =
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eHostWrite;
        _memoryBarriers.emplace_back(
            std::vector<vk::MemoryBarrier2>{vk::MemoryBarrier2(vk::PipelineStageFlagBits2::eAllCommands, accessFlag,
                                                               vk::PipelineStageFlagBits2::eAllCommands, accessFlag)});
        _imageBarriers.emplace_back(std::vector<vk::ImageMemoryBarrier2>{});
        _tensorBarriers.emplace_back(std::vector<vk::TensorMemoryBarrierARM>{});
        _bufferBarriers.emplace_back(std::vector<vk::BufferMemoryBarrier2>{});

        DebugMarker dbgMrk1(this, "barriers (pipeline implicit)");
        _commands.emplace_back(MemoryBarrier{memoryBarrierIdx, imageBarrierIdx, tensorBarrierIdx, bufferBarrierIdx});
    }
}

void Compute::registerWriteTimestamp(uint32_t query, vk::PipelineStageFlagBits2 flag) {
    _commands.emplace_back(WriteTimestamp{query, flag});
}

void Compute::registerPipelineBarrier(const DispatchBarrierDesc &dispatchBarrierDescs, const DataManager &dataManager) {
    const uint32_t memoryBarrierIdx = static_cast<uint32_t>(_memoryBarriers.size());
    const uint32_t imageBarrierIdx = static_cast<uint32_t>(_imageBarriers.size());
    const uint32_t tensorBarrierIdx = static_cast<uint32_t>(_tensorBarriers.size());
    const uint32_t bufferBarrierIdx = static_cast<uint32_t>(_bufferBarriers.size());

    PipelineBarrierDebugNameBuilder debugNameBuilder;

    // Populate each individual barrier struct based on the guids
    std::vector<vk::MemoryBarrier2> memoryBarriers{};
    for (auto &memoryBarrierRef : dispatchBarrierDescs.memoryBarriersRef) {
        const auto &memoryBarrier = dataManager.getMemoryBarrier(memoryBarrierRef);
        debugNameBuilder.addBarrier(memoryBarrier);
        memoryBarriers.emplace_back(memoryBarrier.memoryBarrier());
    }
    _memoryBarriers.emplace_back(memoryBarriers);

    std::vector<vk::ImageMemoryBarrier2> imageBarriers{};
    for (auto &imageBarrierRef : dispatchBarrierDescs.imageBarriersRef) {
        const auto &imageBarrier = dataManager.getImageBarrier(imageBarrierRef);
        debugNameBuilder.addBarrier(imageBarrier);
        imageBarriers.emplace_back(imageBarrier.imageBarrier());
    }
    _imageBarriers.emplace_back(imageBarriers);

    std::vector<vk::TensorMemoryBarrierARM> tensorBarriers{};
    for (auto &tensorBarrierRef : dispatchBarrierDescs.tensorBarriersRef) {
        const auto &tensorBarrier = dataManager.getTensorBarrier(tensorBarrierRef);
        debugNameBuilder.addBarrier(tensorBarrier);
        tensorBarriers.emplace_back(tensorBarrier.tensorBarrier());
    }
    _tensorBarriers.emplace_back(tensorBarriers);

    std::vector<vk::BufferMemoryBarrier2> bufferBarriers{};
    for (auto &bufferBarrierRef : dispatchBarrierDescs.bufferBarriersRef) {
        if (dataManager.hasBufferBarrier(bufferBarrierRef)) {
            bufferBarriers.emplace_back(dataManager.getBufferBarrier(bufferBarrierRef).bufferBarrier());
        } else {
            throw std::runtime_error("Cannot find Buffer memory barrier");
        }
    }
    _bufferBarriers.emplace_back(bufferBarriers);

    DebugMarker dbgMrk0(this, debugNameBuilder.buildDebugName());
    _commands.emplace_back(MemoryBarrier{memoryBarrierIdx, imageBarrierIdx, tensorBarrierIdx, bufferBarrierIdx});
}

void Compute::registerMarkBoundary(const MarkBoundaryDesc &markBoundaryDesc, const DataManager &dataManager) {

    for (auto &resourceRef : markBoundaryDesc.resources) {
        if (!(dataManager.hasImage(resourceRef) || dataManager.hasBuffer(resourceRef) ||
              dataManager.hasTensor(resourceRef))) {
            throw std::runtime_error("Unsupported resource");
        }
    }

    std::vector<vk::Image> imageHandles;
    imageHandles.reserve(markBoundaryDesc.resources.size());

    for (auto &resourceRef : markBoundaryDesc.resources) {
        if (dataManager.hasImage(resourceRef)) {
            const auto image = dataManager.getImage(resourceRef).image();
            imageHandles.emplace_back(image);
        }
    }
    _imageArray.emplace_back(std::move(imageHandles));

    std::vector<vk::Buffer> bufferHandles;
    bufferHandles.reserve(markBoundaryDesc.resources.size());

    for (auto &resourceRef : markBoundaryDesc.resources) {
        if (dataManager.hasBuffer(resourceRef)) {
            const auto buffer = dataManager.getBuffer(resourceRef).buffer();
            bufferHandles.emplace_back(buffer);
        }
    }
    _bufferArray.emplace_back(std::move(bufferHandles));
    vk::FrameBoundaryEXT markBoundary;
    markBoundary.sType = vk::StructureType::eFrameBoundaryEXT;

    markBoundary.flags = vk::FrameBoundaryFlagBitsEXT::eFrameEnd;
    markBoundary.frameID = markBoundaryDesc.frameId;
    markBoundary.pImages = _imageArray.back().data();
    markBoundary.imageCount = static_cast<uint32_t>(_imageArray.back().size());
    markBoundary.pBuffers = _bufferArray.back().data();
    markBoundary.bufferCount = static_cast<uint32_t>(_bufferArray.back().size());
    std::vector<vk::TensorARM> tensorHandles;
    tensorHandles.reserve(markBoundaryDesc.resources.size());

    for (auto &resourceRef : markBoundaryDesc.resources) {
        if (dataManager.hasTensor(resourceRef)) {
            auto tensor = dataManager.getTensor(resourceRef).tensor();
            tensorHandles.emplace_back(tensor);
        }
    }
    _tensorArray.emplace_back(std::move(tensorHandles));
    vk::FrameBoundaryTensorsARM markBoundaryTensor;
    markBoundaryTensor.sType = vk::StructureType::eFrameBoundaryTensorsARM;
    markBoundaryTensor.pTensors = _tensorArray.back().data();
    markBoundaryTensor.tensorCount = static_cast<uint32_t>(_tensorArray.back().size());
    markBoundaryTensor.pNext = nullptr;
    _markBoundaryTensorArray.emplace_back(std::make_unique<vk::FrameBoundaryTensorsARM>(markBoundaryTensor));
    if (_tensorArray.back().size() > 0) {
        markBoundary.pNext = &(*_markBoundaryTensorArray.back());
    } else {
        markBoundary.pNext = nullptr;
    }
    _commands.emplace_back(MarkBoundary{std::move(markBoundary)});
}

void Compute::submitAndWaitOnFence() {
    // Unused arguments
    std::vector<PerformanceCounter> perfCounters{};
    submitAndWaitOnFence(perfCounters, 0);
}

void Compute::submitAndWaitOnFence(std::vector<PerformanceCounter> &perfCounters, int iteration) {
    // Reset query pool
    perfCounters.emplace_back("Reset Query Pool. Iteration: " + std::to_string(iteration + 1), "Run Scenario").start();
    if (*_queryPool) {
        _queryPool.reset(0, _nQueries);
    }
    perfCounters.back().stop();
    // Create command buffer vector
    perfCounters.emplace_back("Creating Command Buffer. Iteration: " + std::to_string(iteration + 1), "Run Scenario")
        .start();
    const vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    _setNextCommandBuffer();
    _cmdBufferArray.back().begin(CmdBufferBeginInfo);
    for (auto &cmd : _commands) {
        if (std::holds_alternative<BindDescriptorSet>(cmd)) {
            auto &typedCmd = std::get<BindDescriptorSet>(cmd);
            const vk::DescriptorSet &descSet = *_descriptorSets[typedCmd.descriptorSetIdxGlobal];

            vk::PipelineBindPoint bindPoint = (typedCmd.bindPoint == BindPoint::DataGraph)
                                                  ? vk::PipelineBindPoint::eDataGraphARM
                                                  : vk::PipelineBindPoint::eCompute;

            _cmdBufferArray.back().bindDescriptorSets(bindPoint, typedCmd.pipelineLayout, typedCmd.descriptorSetId,
                                                      vk::ArrayProxy<vk::DescriptorSet>(descSet),
                                                      vk::ArrayProxy<uint32_t>());
        } else if (std::holds_alternative<BindPipeline>(cmd)) {
            auto &typedCmd = std::get<BindPipeline>(cmd);

            vk::PipelineBindPoint bindPoint = (typedCmd.bindPoint == BindPoint::DataGraph)
                                                  ? vk::PipelineBindPoint::eDataGraphARM
                                                  : vk::PipelineBindPoint::eCompute;

            _cmdBufferArray.back().bindPipeline(bindPoint, typedCmd.pipeline);
        } else if (std::holds_alternative<ComputeDispatch>(cmd)) {
            mlsdk::logging::info("Dispatch compute");
            auto &typedCmd = std::get<ComputeDispatch>(cmd);
            _cmdBufferArray.back().dispatch(typedCmd.gwcx, typedCmd.gwcy, typedCmd.gwcz);
        } else if (std::holds_alternative<DataGraphDispatch>(cmd)) {
            mlsdk::logging::info("Dispatch graph");
            auto &typedCmd = std::get<DataGraphDispatch>(cmd);
            _cmdBufferArray.back().dispatchDataGraphARM(typedCmd.session);
        } else if (std::holds_alternative<MemoryBarrier>(cmd)) {
            auto &typedCmd = std::get<MemoryBarrier>(cmd);
            auto &memoryBarriers = _memoryBarriers[typedCmd.memoryBarrierIdx];
            auto &imageBarriers = _imageBarriers[typedCmd.imageBarrierIdx];
            auto &tensorBarriers = _tensorBarriers[typedCmd.tensorBarrierIdx];
            auto &bufferBarriers = _bufferBarriers[typedCmd.bufferBarrierIdx];

            void *dependencyInfoExt = nullptr;
            auto tensorDependencyInfo =
                vk::TensorDependencyInfoARM(static_cast<uint32_t>(tensorBarriers.size()), tensorBarriers.data());
            if (!tensorBarriers.empty()) {
                dependencyInfoExt = &tensorDependencyInfo;
            }

            _cmdBufferArray.back().pipelineBarrier2(vk::DependencyInfo(
                (vk::DependencyFlags)0, memoryBarriers, bufferBarriers, imageBarriers, dependencyInfoExt));
        } else if (std::holds_alternative<PushConstants>(cmd)) {
            auto &typedCmd = std::get<PushConstants>(cmd);
            _cmdBufferArray.back().pushConstants<char>(typedCmd.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                                       vk::ArrayProxy(typedCmd.size, typedCmd.pushConstantData));
        } else if (std::holds_alternative<WriteTimestamp>(cmd)) {
            if (*_queryPool) {
                const auto &typedCmd = std::get<WriteTimestamp>(cmd);
                _cmdBufferArray.back().writeTimestamp2(typedCmd.flag, *_queryPool, typedCmd.query);
            }
        } else if (std::holds_alternative<MarkBoundary>(cmd)) {
            auto &typeCmd = std::get<MarkBoundary>(cmd);
            _cmdBufferArray.back().end();
            vk::SubmitInfo submitInfo({}, {}, *_cmdBufferArray.back(), {}, &typeCmd.markBoundary);
            _queue.submit(submitInfo, *_fence);
            _waitForFence();
            _setNextCommandBuffer();
            _fence = _ctx.device().createFence({});
            _cmdBufferArray.back().begin(CmdBufferBeginInfo);
        } else if (std::holds_alternative<PushDebugMarker>(cmd)) {
            _cmdBufferArray.back().beginDebugUtilsLabelEXT(
                vk::DebugUtilsLabelEXT{_debugMarkerNames[std::get<PushDebugMarker>(cmd).nameIdx].c_str()});
        } else if (std::holds_alternative<PopDebugMarker>(cmd)) {
            _cmdBufferArray.back().endDebugUtilsLabelEXT();
        } else {
            throw std::runtime_error("Unsupported compute command");
        }
    }
    _cmdBufferArray.back().end();
    perfCounters.back().stop();
    // Run commands
    perfCounters.emplace_back("Submit Commands. Iteration: " + std::to_string(iteration + 1), "Run Scenario").start();
    vk::SubmitInfo submitInfo({}, {}, *_cmdBufferArray.back());
    _queue.submit(submitInfo, *_fence);
    perfCounters.back().stop();

    // Wait to finish
    perfCounters.emplace_back("Wait for Fence. Iteration: " + std::to_string(iteration + 1), "Run Scenario").start();
    _waitForFence();
    perfCounters.back().stop();
}

void Compute::setupQueryPool(uint32_t nQueries) {
    _nQueries = nQueries;
    const vk::QueryPoolCreateInfo queryPoolCreateInfo({}, vk::QueryType::eTimestamp, _nQueries);
    _queryPool = vk::raii::QueryPool(_ctx.device(), queryPoolCreateInfo);
    _queryPool.reset(0, _nQueries);
}

std::vector<uint64_t> Compute::_queryTimestamps() const {
    if (*_queryPool) {
        auto [_, queryPair] = _queryPool.getResults<uint64_t>(0, _nQueries, _nQueries * sizeof(uint64_t),
                                                              static_cast<vk::DeviceSize>(sizeof(uint64_t)),
                                                              vk::QueryResultFlagBits::e64);
        return queryPair;
    } else {
        throw std::runtime_error("Failed to retrieve timestamps, since the query pool is empty");
    }
}

void Compute::writeProfilingFile(const std::filesystem::path &profilingPath, int iteration, int repeatCount) const {
    std::vector<uint64_t> timestamps = _queryTimestamps();
    VkPhysicalDeviceLimits physicalDeviceLimits = _ctx.physicalDevice().getProperties().limits;
    float timestampPeriod = physicalDeviceLimits.timestampPeriod;
    std::vector<std::string> profiledCommands;
    for (const auto &command : _commands) {
        if (std::holds_alternative<ComputeDispatch>(command)) {
            profiledCommands.push_back("ComputeDispatch");
        } else if (std::holds_alternative<DataGraphDispatch>(command)) {
            profiledCommands.push_back("DataGraphDispatch");
        }
    }
    writeProfilingData(timestamps, timestampPeriod, profiledCommands, profilingPath, iteration, repeatCount);
}

} // namespace mlsdk::scenariorunner
