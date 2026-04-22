/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "compute.hpp"
#include "json_writer.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>

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
        }
        return {};
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
    _queue = _ctx.device().getQueue(_ctx.familyQueueIdx(), 0);
    _fence = _ctx.device().createFence({});
}

void Compute::_resetFence() { _ctx.device().resetFences({*_fence}); }

void Compute::reset() {
    _cmdBufferArray.clear();
    _cmdPool.reset(vk::CommandPoolResetFlagBits::eReleaseResources);
    _resetFence();
}

void Compute::_setNextCommandBuffer() {
    const vk::CommandBufferAllocateInfo cmdBufferAllocInfo(*_cmdPool, vk::CommandBufferLevel::ePrimary, 1);
    _cmdBufferArray.emplace_back(std::move(_ctx.device().allocateCommandBuffers(cmdBufferAllocInfo).front()));
}

void Compute::_beginCommandBuffer() {
    const vk::CommandBufferBeginInfo commandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit, // flags
    };
    _cmdBufferArray.back().begin(commandBufferBeginInfo);
}

void Compute::prepareCommandBuffer() {
    if (_cmdBufferArray.empty()) {
        _setNextCommandBuffer();
    }
    _beginCommandBuffer();
}

vk::raii::CommandBuffer &Compute::getCommandBuffer() {
    if (_cmdBufferArray.empty()) {
        throw std::runtime_error("Command buffer not initialized");
    }
    return _cmdBufferArray.back();
}

void Compute::_waitForFence() {
    mlsdk::logging::info("Wait for fence");
    const auto timeout = WAIT_FOR_FENCE_TIMEOUT;
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

void Compute::_addDescriptorSets(const uint32_t baseDescriptorSetIdxGlobal, uint32_t set,
                                 const std::vector<vk::DescriptorPoolSize> &poolSizes, const Pipeline &pipeline) {
    // Add new sets as needed
    while (_descriptorSets.size() <= baseDescriptorSetIdxGlobal + set) {
        const auto nextSet = static_cast<uint32_t>(_descriptorSets.size() - baseDescriptorSetIdxGlobal);
        const vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes);
        _descriptorPools.emplace_back(_ctx.device(), descriptorPoolCreateInfo);

        const vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*_descriptorPools.back(),
                                                                      pipeline.descriptorSetLayout(nextSet));

        _descriptorSets.push_back(
            std::move(vk::raii::DescriptorSets(_ctx.device(), descriptorSetAllocateInfo).front()));
    }
}

void Compute::_updateDescriptorSets(const vk::DescriptorSet &descSet, const TypedBinding &binding,
                                    const IResourceViewer &resourceViewer) {
    if (resourceViewer.hasBuffer()) {
        const vk::Buffer &buf = resourceViewer.getBuffer().buffer();
        const vk::DescriptorBufferInfo info(buf, 0, vk::WholeSize);
        vk::WriteDescriptorSet dwrite(descSet, static_cast<uint32_t>(binding.id), 0, 1, binding.vkDescriptorType, {},
                                      &info);
        _ctx.device().updateDescriptorSets(vk::ArrayProxy<vk::WriteDescriptorSet>(dwrite), {});
    } else if (resourceViewer.hasTensor()) {
        const vk::WriteDescriptorSetTensorARM info(1, &resourceViewer.getTensor().tensorView());
        const vk::WriteDescriptorSet dwrite(descSet, static_cast<uint32_t>(binding.id), 0, 1, binding.vkDescriptorType,
                                            {}, {}, {}, &info);
        _ctx.device().updateDescriptorSets(vk::ArrayProxy<vk::WriteDescriptorSet>(dwrite), {});
    } else if (resourceViewer.hasImage()) {
        vk::ImageView imageView;
        const Image &image = resourceViewer.getImage();

        if (binding.lod.has_value()) {
            imageView = image.imageView(binding.lod.value());
        } else {
            imageView = image.imageView();
        }
        const vk::DescriptorImageInfo info(image.sampler(), imageView, image.getImageLayout());
        const vk::WriteDescriptorSet dwrite(descSet, static_cast<uint32_t>(binding.id), 0, 1, binding.vkDescriptorType,
                                            &info);
        _ctx.device().updateDescriptorSets(vk::ArrayProxy<vk::WriteDescriptorSet>(dwrite), {});
    }
}

void Compute::createPipeline(const PipelineCreateArguments &args, const ShaderInfo &shaderInfo, const uint32_t *spvCode,
                             size_t spvSize) {
    const Pipeline::CommonArguments commonArgs{_ctx, args.debugName, args.bindings, args.pipelineCache};
    (void)_pipelines.emplace_back(commonArgs, shaderInfo, spvCode, spvSize);
}

void Compute::createPipeline(const PipelineCreateArguments &args, uint32_t segmentIndex, const VgfView &vgfView,
                             const DataManager &dataManager) {
    const Pipeline::CommonArguments commonArgs{_ctx, args.debugName, args.bindings, args.pipelineCache};
    (void)_pipelines.emplace_back(commonArgs, segmentIndex, vgfView, dataManager);
}

void Compute::createPipeline(const PipelineCreateArguments &args, const ShaderInfo &shaderInfo,
                             const DataManager &dataManager, const std::vector<GraphConstantInfo> &constants) {
    const Pipeline::CommonArguments commonArgs{_ctx, args.debugName, args.bindings, args.pipelineCache};
    (void)_pipelines.emplace_back(commonArgs, shaderInfo, dataManager, constants);
}

void Compute::createPipeline(const PipelineCreateArguments &args, const ShaderInfo &vertexShaderInfo,
                             const ShaderInfo &fragmentShaderInfo,
                             const std::vector<vk::Format> &colorAttachmentFormats) {
    const Pipeline::CommonArguments commonArgs{_ctx, args.debugName, args.bindings, args.pipelineCache};
    (void)_pipelines.emplace_back(commonArgs, vertexShaderInfo, fragmentShaderInfo, colorAttachmentFormats);
}

void Compute::createPipeline(const PipelineCreateArguments &args, const DataManager &dataManager,
                             const TypedBinding &inputSearch, const TypedBinding &inputTemplate,
                             const TypedBinding &outputFlow, const std::optional<TypedBinding> &inputHintMV,
                             const std::optional<TypedBinding> &outputCost,
                             vk::DataGraphOpticalFlowPerformanceLevelARM performanceLevel,
                             vk::DataGraphOpticalFlowGridSizeFlagsARM gridSize, uint32_t inputWidth,
                             uint32_t inputHeight) {
    const std::vector<TypedBinding> emptyBindings{};
    const Pipeline::CommonArguments commonArgs{_ctx, args.debugName, emptyBindings, args.pipelineCache};
    (void)_pipelines.emplace_back(commonArgs, dataManager, inputSearch, inputTemplate, outputFlow, inputHintMV,
                                  outputCost, performanceLevel, gridSize, inputWidth, inputHeight);
}

void Compute::_registerPipelineFencedCommon(const DataManager &dataManager, const std::vector<TypedBinding> &bindings,
                                            const char *pushConstantData, size_t pushConstantSize) {
    const auto &pipeline = _pipelines.back();
    DebugMarker dbgMrk0(this, "dispatch (" + pipeline.debugName() + ")");

    // Count exact number of typed resources used by this pipeline
    const auto poolSizes = getPoolSizes(bindings);

    // Populate descriptor sets
    const auto baseDescriptorSetIdxGlobal = static_cast<uint32_t>(_descriptorSets.size());
    uint32_t maxSet = 0;
    for (const auto &binding : bindings) {
        maxSet = maxSet < binding.set ? binding.set : maxSet;

        _addDescriptorSets(baseDescriptorSetIdxGlobal, binding.set, poolSizes, pipeline);

        const auto &descSet = *_descriptorSets[baseDescriptorSetIdxGlobal + binding.set];
        const DataManagerResourceViewer resourceViewer(dataManager, binding.resourceRef);
        _updateDescriptorSets(descSet, binding, resourceViewer);
    }

    _addBinds(pipeline, maxSet, baseDescriptorSetIdxGlobal);
    _addPushConstants(pushConstantData, pipeline, pushConstantSize);
}

void Compute::registerPipelineFenced(const DataManager &dataManager, const std::vector<TypedBinding> &bindings,
                                     const char *pushConstantData, size_t pushConstantSize, bool implicitBarriers,
                                     ComputeDispatch computeDispatch,
                                     std::optional<OpticalFlowDispatchInfo> opticalFlowDispatchInfo) {
    _registerPipelineFencedCommon(dataManager, bindings, pushConstantData, pushConstantSize);
    _addDispatch(computeDispatch, opticalFlowDispatchInfo);

    if (implicitBarriers) {
        _addImplicitBarriers();
    }
}

void Compute::registerPipelineFenced(const DataManager &dataManager, const std::vector<TypedBinding> &bindings,
                                     const char *pushConstantData, size_t pushConstantSize, bool implicitBarriers,
                                     const GraphicsDispatchInfo &graphicsDispatch) {
    _registerPipelineFencedCommon(dataManager, bindings, pushConstantData, pushConstantSize);
    _addGraphicsDispatch(graphicsDispatch);

    if (implicitBarriers) {
        _addImplicitBarriers();
    }
}

void Compute::_addBinds(const Pipeline &pipeline, const uint32_t maxSet, const uint32_t baseDescriptorSetIdxGlobal) {
    auto bindPoint = BindPoint::Compute;
    if (pipeline.isDataGraphPipeline()) {
        bindPoint = BindPoint::DataGraph;
    } else if (pipeline.isGraphicsPipeline()) {
        bindPoint = BindPoint::Graphics;
    }

    _commands.emplace_back(BindPipeline{pipeline.pipeline(), bindPoint});

    for (uint32_t setId = 0; setId <= maxSet; setId++) {
        _commands.emplace_back(
            BindDescriptorSet{pipeline.pipelineLayout(), baseDescriptorSetIdxGlobal + setId, setId, bindPoint});
    }
}

void Compute::_addPushConstants(const char *pushConstantData, const Pipeline &pipeline, const size_t pushConstantSize) {
    if (pushConstantData != nullptr && pushConstantSize > 0 && pipeline.pushConstantStages()) {
        _commands.emplace_back(PushConstants{pipeline.pipelineLayout(), pushConstantData,
                                             static_cast<uint32_t>(pushConstantSize), pipeline.pushConstantStages()});
    }
}

void Compute::_addDispatch(const ComputeDispatch &computeDispatch,
                           const std::optional<OpticalFlowDispatchInfo> &dispatchInfo) {
    const auto &pipeline = _pipelines.back();
    if (pipeline.isDataGraphPipeline()) {
        DataGraphDispatch dispatch{pipeline.session()};
        if (dispatchInfo.has_value()) {
            dispatch.dispatchInfo = dispatchInfo;
        }
        _commands.emplace_back(dispatch);
    } else {
        _commands.emplace_back(computeDispatch);
    }
}

void Compute::_addGraphicsDispatch(const GraphicsDispatchInfo &graphicsDispatch) {
    _commands.emplace_back(GraphicsDispatch{graphicsDispatch});
}

void Compute::_addImplicitBarriers() {
    // Set an implicit memory barrier
    const auto memoryBarrierIdx = static_cast<uint32_t>(_memoryBarriers.size());
    const auto imageBarrierIdx = static_cast<uint32_t>(_imageBarriers.size());
    const auto tensorBarrierIdx = static_cast<uint32_t>(_tensorBarriers.size());
    const auto bufferBarrierIdx = static_cast<uint32_t>(_bufferBarriers.size());

    auto accessFlag = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    _memoryBarriers.emplace_back(std::vector<vk::MemoryBarrier2>{vk::MemoryBarrier2(
        vk::PipelineStageFlagBits2::eAllCommands, accessFlag, vk::PipelineStageFlagBits2::eAllCommands, accessFlag)});
    _imageBarriers.emplace_back(std::vector<vk::ImageMemoryBarrier2>{});
    _tensorBarriers.emplace_back(std::vector<vk::TensorMemoryBarrierARM>{});
    _bufferBarriers.emplace_back(std::vector<vk::BufferMemoryBarrier2>{});

    DebugMarker dbgMrk1(this, "barriers (pipeline implicit)");
    _commands.emplace_back(MemoryBarrier{memoryBarrierIdx, imageBarrierIdx, tensorBarrierIdx, bufferBarrierIdx});
}

void Compute::registerWriteTimestamp(uint32_t query, vk::PipelineStageFlagBits2 flag) {
    _commands.emplace_back(WriteTimestamp{query, flag});
}

void Compute::registerPipelineBarrier(const DispatchBarrierData &dispatchBarrierData, const DataManager &dataManager) {
    const auto memoryBarrierIdx = static_cast<uint32_t>(_memoryBarriers.size());
    const auto imageBarrierIdx = static_cast<uint32_t>(_imageBarriers.size());
    const auto tensorBarrierIdx = static_cast<uint32_t>(_tensorBarriers.size());
    const auto bufferBarrierIdx = static_cast<uint32_t>(_bufferBarriers.size());

    PipelineBarrierDebugNameBuilder debugNameBuilder;

    // Populate each individual barrier struct based on the guids
    std::vector<vk::MemoryBarrier2> memoryBarriers{};
    for (const auto &memoryBarrierRef : dispatchBarrierData.memoryBarriers) {
        const auto &memoryBarrier = dataManager.getMemoryBarrier(memoryBarrierRef);
        debugNameBuilder.addBarrier(memoryBarrier);
        memoryBarriers.emplace_back(memoryBarrier.memoryBarrier());
    }
    _memoryBarriers.emplace_back(memoryBarriers);

    std::vector<vk::ImageMemoryBarrier2> imageBarriers{};
    for (const auto &imageBarrierRef : dispatchBarrierData.imageBarriers) {
        const auto &imageBarrier = dataManager.getImageBarrier(imageBarrierRef);
        debugNameBuilder.addBarrier(imageBarrier);
        imageBarriers.emplace_back(imageBarrier.imageBarrier());
    }
    _imageBarriers.emplace_back(imageBarriers);

    std::vector<vk::TensorMemoryBarrierARM> tensorBarriers{};
    for (const auto &tensorBarrierRef : dispatchBarrierData.tensorBarriers) {
        const auto &tensorBarrier = dataManager.getTensorBarrier(tensorBarrierRef);
        debugNameBuilder.addBarrier(tensorBarrier);
        tensorBarriers.emplace_back(tensorBarrier.tensorBarrier());
    }
    _tensorBarriers.emplace_back(tensorBarriers);

    std::vector<vk::BufferMemoryBarrier2> bufferBarriers{};
    for (const auto &bufferBarrierRef : dispatchBarrierData.bufferBarriers) {
        const auto &bufferBarrier = dataManager.getBufferBarrier(bufferBarrierRef);
        bufferBarriers.emplace_back(bufferBarrier.bufferBarrier());
    }
    _bufferBarriers.emplace_back(bufferBarriers);

    DebugMarker dbgMrk0(this, debugNameBuilder.buildDebugName());
    _commands.emplace_back(MemoryBarrier{memoryBarrierIdx, imageBarrierIdx, tensorBarrierIdx, bufferBarrierIdx});
}

vk::FrameBoundaryEXT Compute::_createFrameBoundary() {
    vk::FrameBoundaryEXT frameBoundary;
    frameBoundary.flags = vk::FrameBoundaryFlagBitsEXT::eFrameEnd;
    frameBoundary.frameID = 0;
    frameBoundary.pImages = _imageArray.back().data();
    frameBoundary.imageCount = static_cast<uint32_t>(_imageArray.back().size());
    frameBoundary.pBuffers = _bufferArray.back().data();
    frameBoundary.bufferCount = static_cast<uint32_t>(_bufferArray.back().size());

    vk::FrameBoundaryTensorsARM frameBoundaryTensor;
    frameBoundaryTensor.pTensors = _tensorArray.back().data();
    frameBoundaryTensor.tensorCount = static_cast<uint32_t>(_tensorArray.back().size());
    frameBoundaryTensor.pNext = nullptr;
    _markBoundaryTensorArray.emplace_back(std::make_unique<vk::FrameBoundaryTensorsARM>(frameBoundaryTensor));
    if (!_tensorArray.back().empty()) {
        frameBoundary.pNext = &(*_markBoundaryTensorArray.back());
    } else {
        frameBoundary.pNext = nullptr;
    }
    return frameBoundary;
}

void Compute::_addMarkBoundary() { _commands.emplace_back(MarkBoundary{_createFrameBoundary()}); }

void Compute::registerMarkBoundary(const MarkBoundaryData &markBoundaryData, const DataManager &dataManager) {
    std::vector<vk::Image> imageHandles;
    imageHandles.reserve(markBoundaryData.images.size());
    for (const auto &resourceRef : markBoundaryData.images) {
        auto image = dataManager.getImage(resourceRef).image();
        imageHandles.emplace_back(image);
    }
    _imageArray.emplace_back(std::move(imageHandles));

    std::vector<vk::Buffer> bufferHandles;
    bufferHandles.reserve(markBoundaryData.buffers.size());
    for (const auto &resourceRef : markBoundaryData.buffers) {
        auto buffer = dataManager.getBuffer(resourceRef).buffer();
        bufferHandles.emplace_back(buffer);
    }
    _bufferArray.emplace_back(std::move(bufferHandles));

    std::vector<vk::TensorARM> tensorHandles;
    tensorHandles.reserve(markBoundaryData.tensors.size());
    for (const auto &resourceRef : markBoundaryData.tensors) {
        auto tensor = dataManager.getTensor(resourceRef).tensor();
        tensorHandles.emplace_back(tensor);
    }
    _tensorArray.emplace_back(std::move(tensorHandles));

    _addMarkBoundary();
}

void Compute::submitAndWaitOnFence() {
    // Unused arguments
    std::vector<PerformanceCounter> perfCounters{};
    submitAndWaitOnFence(perfCounters, 0);
}

vk::PipelineBindPoint Compute::_getBindPoint(BindPoint bindPoint) {
    vk::PipelineBindPoint vkBindPoint;
    switch (bindPoint) {
    case BindPoint::DataGraph:
        vkBindPoint = vk::PipelineBindPoint::eDataGraphARM;
        break;
    case BindPoint::Graphics:
        vkBindPoint = vk::PipelineBindPoint::eGraphics;
        break;
    default:
        vkBindPoint = vk::PipelineBindPoint::eCompute;
        break;
    }

    return vkBindPoint;
}

void Compute::_createCmdBuffer() {
    _resetFence();
    _setNextCommandBuffer();
    _beginCommandBuffer();

    // If a frame boundary command is present, also add one after initial setup
    bool hasFrameBoundary = false;
    for (auto &cmd : _commands) {
        if (std::holds_alternative<MarkBoundary>(cmd)) {
            hasFrameBoundary = true;
        }
    }
    if (hasFrameBoundary && _repeatNumber == 0) {
        _cmdBufferArray.back().end();

        auto frameBoundary = _createFrameBoundary();
        frameBoundary.frameID = _repeatNumber++;

        vk::SubmitInfo submitInfo({}, {}, *_cmdBufferArray.back(), {}, &frameBoundary);
        _queue.submit(submitInfo, *_fence);
        _waitForFence();
        _resetFence();
        _setNextCommandBuffer();
        _beginCommandBuffer();
    }

    for (auto &cmd : _commands) {
        if (std::holds_alternative<BindDescriptorSet>(cmd)) {
            auto &typedCmd = std::get<BindDescriptorSet>(cmd);
            const vk::DescriptorSet &descSet = *_descriptorSets[typedCmd.descriptorSetIdxGlobal];

            auto bindPoint = _getBindPoint(typedCmd.bindPoint);
            _cmdBufferArray.back().bindDescriptorSets(bindPoint, typedCmd.pipelineLayout, typedCmd.descriptorSetId,
                                                      vk::ArrayProxy<vk::DescriptorSet>(descSet),
                                                      vk::ArrayProxy<uint32_t>());
        } else if (std::holds_alternative<BindPipeline>(cmd)) {
            auto &typedCmd = std::get<BindPipeline>(cmd);
            auto bindPoint = _getBindPoint(typedCmd.bindPoint);
            _cmdBufferArray.back().bindPipeline(bindPoint, typedCmd.pipeline);
        } else if (std::holds_alternative<ComputeDispatch>(cmd)) {
            mlsdk::logging::info("Dispatch compute");
            auto &typedCmd = std::get<ComputeDispatch>(cmd);
            _cmdBufferArray.back().dispatch(typedCmd.gwcx, typedCmd.gwcy, typedCmd.gwcz);
        } else if (std::holds_alternative<DataGraphDispatch>(cmd)) {
            mlsdk::logging::info("Dispatch graph");
            auto &typedCmd = std::get<DataGraphDispatch>(cmd);
            if (typedCmd.dispatchInfo.has_value()) {
                vk::DataGraphPipelineOpticalFlowDispatchInfoARM opticalFlowInfo{};
                opticalFlowInfo.setFlags(typedCmd.dispatchInfo->opticalFlowFlags);
                opticalFlowInfo.setMeanFlowL1NormHint(typedCmd.dispatchInfo->meanFlowL1NormHint);

                vk::DataGraphPipelineDispatchInfoARM dispatchInfo{};
                dispatchInfo.setFlags(vk::DataGraphPipelineDispatchFlagsARM{});
                dispatchInfo.setPNext(&opticalFlowInfo);

                _cmdBufferArray.back().dispatchDataGraphARM(typedCmd.session, &dispatchInfo);
            } else {
                _cmdBufferArray.back().dispatchDataGraphARM(typedCmd.session);
            }
        } else if (std::holds_alternative<GraphicsDispatch>(cmd)) {
            mlsdk::logging::info("Dispatch graphics");
            auto &typedCmd = std::get<GraphicsDispatch>(cmd);

            std::vector<vk::RenderingAttachmentInfo> colorAttachmentInfos;
            colorAttachmentInfos.reserve(typedCmd.info.colorAttachments.size());
            for (const auto &attachment : typedCmd.info.colorAttachments) {
                vk::RenderingAttachmentInfo attachmentInfo{};
                attachmentInfo.setImageView(attachment.view);
                attachmentInfo.setImageLayout(attachment.layout);
                attachmentInfo.setLoadOp(vk::AttachmentLoadOp::eDontCare);
                attachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
                colorAttachmentInfos.push_back(attachmentInfo);
            }

            vk::RenderingInfo renderingInfo{};
            renderingInfo.setRenderArea(vk::Rect2D({0, 0}, typedCmd.info.extent));
            renderingInfo.setLayerCount(1);
            renderingInfo.setColorAttachmentCount(static_cast<uint32_t>(colorAttachmentInfos.size()));
            if (colorAttachmentInfos.empty()) {
                renderingInfo.setPColorAttachments(nullptr);
            } else {
                renderingInfo.setPColorAttachments(colorAttachmentInfos.data());
            }

            _cmdBufferArray.back().beginRendering(renderingInfo);
            vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(typedCmd.info.extent.width),
                                  static_cast<float>(typedCmd.info.extent.height), 0.0f, 1.0f);
            vk::Rect2D scissor({0, 0}, typedCmd.info.extent);
            _cmdBufferArray.back().setViewport(0, viewport);
            _cmdBufferArray.back().setScissor(0, scissor);
            _cmdBufferArray.back().draw(3, 1, 0, 0);
            _cmdBufferArray.back().endRendering();
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
            _cmdBufferArray.back().pushConstants<char>(typedCmd.pipelineLayout, typedCmd.stages, 0,
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

            typeCmd.markBoundary.frameID = _repeatNumber++;

            _queue.submit(submitInfo, *_fence);
            _waitForFence();
            _resetFence();
            _setNextCommandBuffer();
            _beginCommandBuffer();
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
}

void Compute::submitAndWaitOnFence(std::vector<PerformanceCounter> &perfCounters, int iteration) {
    auto iterationStr = std::to_string(iteration + 1);
    // Reset query pool
    {
        PerfCounterGuard guard(perfCounters, "Reset Query Pool. Iteration: " + iterationStr, "Run Scenario", false);
        if (*_queryPool) {
            _queryPool.reset(0, _nQueries);
        }
    }

    // Create command buffer vector
    {
        PerfCounterGuard guard(perfCounters, "Creating Command Buffer. Iteration: " + iterationStr, "Run Scenario",
                               false);
        _createCmdBuffer();
    }

    // Run commands
    {
        vk::SubmitInfo submitInfo({}, {}, *_cmdBufferArray.back());
        PerfCounterGuard guard(perfCounters, "Submit Commands. Iteration: " + iterationStr, "Run Scenario", false);
        _queue.submit(submitInfo, *_fence);
    }

    // Wait to finish
    {
        PerfCounterGuard guard(perfCounters, "Wait for Fence. Iteration: " + iterationStr, "Run Scenario", false);
        _waitForFence();
        _cmdBufferArray.clear();
    }
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
    }
    throw std::runtime_error("Failed to retrieve timestamps, since the query pool is empty");
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
        } else if (std::holds_alternative<GraphicsDispatch>(command)) {
            profiledCommands.push_back("GraphicsDispatch");
        }
    }
    std::vector<uint64_t> memoryUsages;
    for (const auto &pipeline : _pipelines) {
        if (pipeline.isDataGraphPipeline()) {
            memoryUsages.push_back(pipeline.getDataGraphPipelineMemoryRequirement());
        }
    }
    writeProfilingData(timestamps, timestampPeriod, profiledCommands, memoryUsages, profilingPath, iteration,
                       repeatCount);
}

void Compute::sessionRAMsDump(const std::filesystem::path &sessionRAMsDumpDir) const {
    uint32_t graphPipelineIdx = 0;
    for (const auto &pipeline : _pipelines) {
        const auto &sessionMemory = pipeline.sessionMemory();
        const auto &sessionMemoryDataSizes = pipeline.sessionMemoryDataSizes();

        for (size_t i = 0; i < sessionMemory.size(); i++) {
            const std::string neStatsFileName =
                "Graph_Pipeline_" + std::to_string(graphPipelineIdx++) + "_Session_RAM_" + std::to_string(i) + ".txt";
            std::ofstream fs;
            fs.open(sessionRAMsDumpDir / neStatsFileName);

            const vk::raii::DeviceMemory &deviceMemory = sessionMemory.at(i);
            uint64_t dataSize = sessionMemoryDataSizes.at(i);

            auto *dst = reinterpret_cast<unsigned char *>(deviceMemory.mapMemory(0, vk::WholeSize));

            fs << std::hex << std::uppercase;
            fs.fill('0');

            for (size_t j = 0; j < dataSize; j++) {
                if ((j % 16) == 0) {
                    fs << std::endl << std::setw(8) << j << ":   ";
                }
                fs << std::setw(2) << static_cast<unsigned>(dst[j]) << " ";
            }

            deviceMemory.unmapMemory();
            fs.close();
        }
        mlsdk::logging::info("Session RAM dump stored");
    }
}

} // namespace mlsdk::scenariorunner
