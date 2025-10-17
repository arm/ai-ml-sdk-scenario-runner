/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "barrier.hpp"
#include "utils.hpp"

namespace mlsdk::scenariorunner {

namespace {
vk::AccessFlags2 convertAccessFlags(MemoryAccess access) {
    switch (access) {
    case (MemoryAccess::MemoryWrite):
        return vk::AccessFlagBits2::eMemoryWrite;
    case (MemoryAccess::MemoryRead):
        return vk::AccessFlagBits2::eMemoryRead;
    case (MemoryAccess::GraphWrite):
        return vk::AccessFlagBits2::eDataGraphWriteARM;
    case (MemoryAccess::GraphRead):
        return vk::AccessFlagBits2::eDataGraphReadARM;
    case (MemoryAccess::ComputeShaderWrite):
        return vk::AccessFlagBits2::eShaderWrite;
    case (MemoryAccess::ComputeShaderRead):
        return vk::AccessFlagBits2::eShaderRead;
    default: {
        throw std::runtime_error("Invalid barrier access flag");
    }
    }
}

vk::PipelineStageFlagBits2 convertStageFlag(PipelineStage stage) {
    switch (stage) {
    case (PipelineStage::Graph):
        return vk::PipelineStageFlagBits2::eDataGraphARM;
    case (PipelineStage::Compute):
        return vk::PipelineStageFlagBits2::eComputeShader;
    case (PipelineStage::All):
        return vk::PipelineStageFlagBits2::eAllCommands;
    default: {
        throw std::runtime_error("Invalid barrier stage flag");
    }
    }
}

vk::Flags<vk::PipelineStageFlagBits2> convertStageFlags(const std::vector<PipelineStage> &stages) {
    vk::Flags<vk::PipelineStageFlagBits2> result = vk::PipelineStageFlagBits2::eNone;
    for (auto stage : stages) {
        result = result | convertStageFlag(stage);
    }
    return result;
}

vk::ImageLayout convertImageLayout(ImageLayout layout) {
    switch (layout) {
    case (ImageLayout::TensorAliasing):
        return vk::ImageLayout::eTensorAliasingARM;
    case (ImageLayout::General):
        return vk::ImageLayout::eGeneral;
    case (ImageLayout::Undefined):
        return vk::ImageLayout::eUndefined;
    default: {
        throw std::runtime_error("Invalid image barrier layout");
    }
    }
}
} // namespace

VulkanImageBarrier::VulkanImageBarrier(const ImageBarrierData &imageBarrierData)
    : _debugName{imageBarrierData.debugName} {
    _imageBarrier.srcAccessMask = convertAccessFlags(imageBarrierData.srcAccess);
    _imageBarrier.dstAccessMask = convertAccessFlags(imageBarrierData.dstAccess);
    _imageBarrier.srcStageMask = convertStageFlags(imageBarrierData.srcStages);
    _imageBarrier.dstStageMask = convertStageFlags(imageBarrierData.dstStages);
    _imageBarrier.oldLayout = convertImageLayout(imageBarrierData.oldLayout);
    _imageBarrier.newLayout = convertImageLayout(imageBarrierData.newLayout);
    _imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    _imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    _imageBarrier.image = imageBarrierData.image;
    _imageBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    _imageBarrier.subresourceRange.baseMipLevel = imageBarrierData.imageRange.baseMipLevel;
    _imageBarrier.subresourceRange.levelCount = imageBarrierData.imageRange.levelCount;
    _imageBarrier.subresourceRange.baseArrayLayer = imageBarrierData.imageRange.baseArrayLayer;
    _imageBarrier.subresourceRange.layerCount = imageBarrierData.imageRange.layerCount;
}

const vk::ImageMemoryBarrier2 &VulkanImageBarrier::imageBarrier() const { return _imageBarrier; }
const std::string &VulkanImageBarrier::debugName() const { return _debugName; }

VulkanTensorBarrier::VulkanTensorBarrier(const TensorBarrierData &tensorBarrierData)
    : _debugName{tensorBarrierData.debugName} {
    _tensorBarrier.srcAccessMask = convertAccessFlags(tensorBarrierData.srcAccess);
    _tensorBarrier.dstAccessMask = convertAccessFlags(tensorBarrierData.dstAccess);
    _tensorBarrier.srcStageMask = convertStageFlags(tensorBarrierData.srcStages);
    _tensorBarrier.dstStageMask = convertStageFlags(tensorBarrierData.dstStages);
    _tensorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    _tensorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    _tensorBarrier.tensor = tensorBarrierData.tensor;
}

const vk::TensorMemoryBarrierARM &VulkanTensorBarrier::tensorBarrier() const { return _tensorBarrier; }
const std::string &VulkanTensorBarrier::debugName() const { return _debugName; }

VulkanMemoryBarrier::VulkanMemoryBarrier(const MemoryBarrierData &memoryBarrierData)
    : _debugName{memoryBarrierData.debugName} {
    _memoryBarrier.srcAccessMask = convertAccessFlags(memoryBarrierData.srcAccess);
    _memoryBarrier.dstAccessMask = convertAccessFlags(memoryBarrierData.dstAccess);
    _memoryBarrier.srcStageMask = convertStageFlags(memoryBarrierData.srcStages);
    _memoryBarrier.dstStageMask = convertStageFlags(memoryBarrierData.dstStages);
}

const vk::MemoryBarrier2 &VulkanMemoryBarrier::memoryBarrier() const { return _memoryBarrier; }
const std::string &VulkanMemoryBarrier::debugName() const { return _debugName; }

VulkanBufferBarrier::VulkanBufferBarrier(const BufferBarrierData &bufferBarrierData)
    : _debugName{bufferBarrierData.debugName} {
    _bufferBarrier.srcAccessMask = convertAccessFlags(bufferBarrierData.srcAccess);
    _bufferBarrier.dstAccessMask = convertAccessFlags(bufferBarrierData.dstAccess);
    _bufferBarrier.srcStageMask = convertStageFlags(bufferBarrierData.srcStages);
    _bufferBarrier.dstStageMask = convertStageFlags(bufferBarrierData.dstStages);
    _bufferBarrier.offset = bufferBarrierData.offset;
    _bufferBarrier.size = bufferBarrierData.size;
    _bufferBarrier.buffer = bufferBarrierData.buffer;
}

const vk::BufferMemoryBarrier2 &VulkanBufferBarrier::bufferBarrier() const { return _bufferBarrier; }
const std::string &VulkanBufferBarrier::debugName() const { return _debugName; }
} // namespace mlsdk::scenariorunner
