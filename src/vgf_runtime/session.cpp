/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "session.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace mlsdk::vgf_runtime {
namespace {

std::vector<std::vector<DescriptorBindingInfo>> splitBindingsBySet(const std::vector<DescriptorBindingInfo> &bindings) {
    std::vector<std::vector<DescriptorBindingInfo>> sets;
    for (const auto &binding : bindings) {
        while (sets.size() <= binding.set) {
            sets.emplace_back();
        }
        sets[binding.set].push_back(binding);
    }
    return sets;
}

std::vector<vk::DescriptorSetLayout>
rawLayouts(const std::vector<vk::raii::DescriptorSetLayout> &descriptorSetLayouts) {
    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.reserve(descriptorSetLayouts.size());
    for (const auto &layout : descriptorSetLayouts) {
        layouts.push_back(*layout);
    }
    return layouts;
}

uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t memoryTypeBits,
                        vk::MemoryPropertyFlags requiredFlags) {
    const auto memoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool supportsType = (memoryTypeBits & (uint32_t{1} << i)) != 0;
        const bool hasFlags = (memoryProperties.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags;
        if (supportsType && hasFlags) {
            return i;
        }
    }
    throw std::runtime_error("Cannot find a compatible memory type");
}

void waitForFence(const vk::raii::Device &device, const vk::raii::Fence &fence) {
    const auto result = device.waitForFences(*fence, true, std::numeric_limits<uint64_t>::max());
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("vkWaitForFences failed with VkResult " +
                                 std::to_string(static_cast<int32_t>(result)));
    }
}

vk::PipelineBindPoint bindPoint(ModuleType type) {
    switch (type) {
    case ModuleType::GRAPH:
        return vk::PipelineBindPoint::eDataGraphARM;
    case ModuleType::SHADER:
        return vk::PipelineBindPoint::eCompute;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

vk::PipelineStageFlags2 pipelineStage(ModuleType type) {
    switch (type) {
    case ModuleType::GRAPH:
        return vk::PipelineStageFlagBits2::eDataGraphARM;
    case ModuleType::SHADER:
        return vk::PipelineStageFlagBits2::eComputeShader;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

vk::AccessFlags2 readAccess(ModuleType type) {
    switch (type) {
    case ModuleType::GRAPH:
        return vk::AccessFlagBits2::eDataGraphReadARM;
    case ModuleType::SHADER:
        return vk::AccessFlagBits2::eShaderRead;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

vk::AccessFlags2 writeAccess(ModuleType type) {
    switch (type) {
    case ModuleType::GRAPH:
        return vk::AccessFlagBits2::eDataGraphWriteARM;
    case ModuleType::SHADER:
        return vk::AccessFlagBits2::eShaderWrite;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

} // namespace

struct Session::BoundTensor {
    DescriptorBindingInfo binding;
    vk::TensorARM tensor{nullptr};
    vk::raii::TensorViewARM tensorView{nullptr};
};

struct Session::BoundBuffer {
    DescriptorBindingInfo binding;
    vk::Buffer buffer{nullptr};
};

struct Session::SegmentState {
    // Common members
    ModuleType type = ModuleType::UNKNOWN;
    vk::raii::ShaderModule shaderModule{nullptr};
    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
    vk::raii::PipelineLayout pipelineLayout{nullptr};
    vk::raii::Pipeline pipeline{nullptr};
    std::vector<vk::raii::DeviceMemory> sessionMemory;
    vk::raii::DescriptorPool descriptorPool{nullptr};
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    std::vector<DescriptorBindingInfo> bindings;
    // Graph
    vk::raii::DataGraphPipelineSessionARM graphSession{nullptr};
    // Shader
    std::array<uint32_t, 3> dispatchShape = {};
};

Session::Session(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device,
                 uint32_t queueFamilyIndex, const vk::raii::Queue &queue, const VGF &vgf)
    : physicalDevice_(physicalDevice), device_(device), vgf_(vgf), queueFamilyIndex_(queueFamilyIndex), queue_(queue) {}

Session::~Session() = default;

const Session::BoundTensor *Session::findBoundTensor(uint32_t resourceIndex) const {
    const auto tensor =
        std::find_if(boundTensors_.rbegin(), boundTensors_.rend(), [resourceIndex](const auto &boundTensor) {
            return boundTensor.binding.resourceIndex == resourceIndex;
        });
    return tensor != boundTensors_.rend() ? &*tensor : nullptr;
}

const Session::BoundBuffer *Session::findBoundBuffer(uint32_t resourceIndex) const {
    const auto buffer =
        std::find_if(boundBuffers_.rbegin(), boundBuffers_.rend(), [resourceIndex](const auto &boundBuffer) {
            return boundBuffer.binding.resourceIndex == resourceIndex;
        });
    return buffer != boundBuffers_.rend() ? &*buffer : nullptr;
}

void Session::updateDescriptorSets(const std::vector<vk::raii::DescriptorSet> &descriptorSets,
                                   const std::vector<DescriptorBindingInfo> &bindings) const {
    for (const auto &binding : bindings) {
        switch (binding.descriptorType) {
        case vk::DescriptorType::eTensorARM: {
            const auto *const tensor = findBoundTensor(binding.resourceIndex);
            const auto tensorView = *tensor->tensorView;
            const vk::WriteDescriptorSetTensorARM tensorInfo(1, &tensorView);
            const vk::WriteDescriptorSet write(*descriptorSets[binding.set], binding.binding, 0, 1,
                                               binding.descriptorType, nullptr, nullptr, nullptr, &tensorInfo);
            device_.updateDescriptorSets(write, nullptr);
            break;
        }
        case vk::DescriptorType::eStorageBuffer: {
            const auto *const buffer = findBoundBuffer(binding.resourceIndex);
            const vk::DescriptorBufferInfo bufferInfo(buffer->buffer, 0, vk::WholeSize);
            const vk::WriteDescriptorSet write(*descriptorSets[binding.set], binding.binding, 0, 1,
                                               binding.descriptorType, nullptr, &bufferInfo);
            device_.updateDescriptorSets(write, nullptr);
            break;
        }
        default:
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(binding.descriptorType)));
        }
    }
}

void Session::insertSegmentBarrier(vk::raii::CommandBuffer &commandBuffer, const SegmentState &producer,
                                   const SegmentState &consumer) const {
    std::vector<vk::TensorMemoryBarrierARM> tensorBarriers;
    std::vector<vk::BufferMemoryBarrier2> bufferBarriers;
    std::vector<uint32_t> barrierResourceIndices;

    for (const auto &producerBinding : producer.bindings) {
        if ((producerBinding.resourceCategory != ResourceCategory::OUTPUT &&
             producerBinding.resourceCategory != ResourceCategory::INTERMEDIATE) ||
            std::find(barrierResourceIndices.begin(), barrierResourceIndices.end(), producerBinding.resourceIndex) !=
                barrierResourceIndices.end()) {
            continue;
        }

        switch (producerBinding.descriptorType) {
        case vk::DescriptorType::eTensorARM: {
            const auto *const tensor = findBoundTensor(producerBinding.resourceIndex);
            vk::TensorMemoryBarrierARM tensorBarrier;
            tensorBarrier.srcStageMask = pipelineStage(producer.type);
            tensorBarrier.srcAccessMask = writeAccess(producer.type);
            tensorBarrier.dstStageMask = pipelineStage(consumer.type);
            tensorBarrier.dstAccessMask = readAccess(consumer.type) | writeAccess(consumer.type);
            tensorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tensorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tensorBarrier.tensor = tensor->tensor;

            tensorBarriers.push_back(tensorBarrier);
            barrierResourceIndices.push_back(producerBinding.resourceIndex);
            break;
        }
        case vk::DescriptorType::eStorageBuffer: {
            const auto *const buffer = findBoundBuffer(producerBinding.resourceIndex);
            vk::BufferMemoryBarrier2 bufferBarrier;
            bufferBarrier.srcStageMask = pipelineStage(producer.type);
            bufferBarrier.srcAccessMask = writeAccess(producer.type);
            bufferBarrier.dstStageMask = pipelineStage(consumer.type);
            bufferBarrier.dstAccessMask = readAccess(consumer.type) | writeAccess(consumer.type);
            bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarrier.buffer = buffer->buffer;
            bufferBarrier.offset = 0;
            bufferBarrier.size = vk::WholeSize;

            bufferBarriers.push_back(bufferBarrier);
            barrierResourceIndices.push_back(producerBinding.resourceIndex);
            break;
        }
        default:
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(producerBinding.descriptorType)));
        }
    }

    if (tensorBarriers.empty() && bufferBarriers.empty()) {
        return;
    }

    vk::TensorDependencyInfoARM tensorDependencyInfo(static_cast<uint32_t>(tensorBarriers.size()),
                                                     tensorBarriers.data());
    vk::DependencyInfo dependencyInfo;
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = bufferBarriers.data();
    if (!tensorBarriers.empty()) {
        dependencyInfo.pNext = &tensorDependencyInfo;
    }
    commandBuffer.pipelineBarrier2(dependencyInfo);
}

void Session::bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding) {
    const auto resource = vgf_.getResource(binding.resourceIndex);
    const vk::TensorViewCreateInfoARM viewCreateInfo({}, *tensor, resource.format);
    boundTensors_.push_back({binding, *tensor, vk::raii::TensorViewARM(device_, viewCreateInfo)});
}

void Session::bindBuffer(const vk::raii::Buffer &buffer, DescriptorBindingInfo binding) {
    boundBuffers_.push_back({binding, *buffer});
}

void Session::configureSegment(uint32_t segmentIndex) {
    const auto segment = vgf_.getSegment(segmentIndex);
    if (segment.type != ModuleType::GRAPH && segment.type != ModuleType::SHADER) {
        throw std::runtime_error("Session only supports VGF data graph and compute shader segments");
    }

    auto &state = segments_.emplace_back();
    state.type = segment.type;
    state.bindings = vgf_.getDescriptorBindings(segmentIndex);
    for (const auto &binding : state.bindings) {
        if (binding.descriptorType != vk::DescriptorType::eStorageBuffer &&
            binding.descriptorType != vk::DescriptorType::eTensorARM) {
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(binding.descriptorType)));
        }
    }

    const auto bindingSets = splitBindingsBySet(state.bindings);
    state.descriptorSetLayouts.reserve(bindingSets.size());
    for (const auto &setBindings : bindingSets) {
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
        layoutBindings.reserve(setBindings.size());
        for (const auto &binding : setBindings) {
            layoutBindings.emplace_back(binding.binding, binding.descriptorType, 1, vk::ShaderStageFlagBits::eAll);
        }
        state.descriptorSetLayouts.emplace_back(device_, vk::DescriptorSetLayoutCreateInfo({}, layoutBindings));
    }

    const auto descriptorSetLayouts = rawLayouts(state.descriptorSetLayouts);
    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, descriptorSetLayouts);
    state.pipelineLayout = vk::raii::PipelineLayout(device_, pipelineLayoutCreateInfo);

    const auto module = vgf_.getSPIRVModule(segment.moduleIndex);
    const vk::ShaderModuleCreateInfo shaderCreateInfo({}, module.code.size() * sizeof(uint32_t), module.code.data());
    state.shaderModule = vk::raii::ShaderModule(device_, shaderCreateInfo);

    if (segment.type == ModuleType::SHADER) {
        const auto dispatchShape = vgf_.getDispatchShape(segmentIndex);
        if (dispatchShape.size() != state.dispatchShape.size() || dispatchShape[0] == 0 || dispatchShape[1] == 0 ||
            dispatchShape[2] == 0) {
            throw std::runtime_error("VGF compute shader segments must have a non-zero 3D dispatch shape");
        }
        std::copy(dispatchShape.begin(), dispatchShape.end(), state.dispatchShape.begin());

        const vk::PipelineShaderStageCreateInfo shaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                                      *state.shaderModule, module.entryPoint.c_str());
        const vk::ComputePipelineCreateInfo pipelineCreateInfo({}, shaderStageCreateInfo, *state.pipelineLayout);
        const vk::raii::PipelineCache *pipelineCache = nullptr;
        state.pipeline = vk::raii::Pipeline(device_, pipelineCache, pipelineCreateInfo);
    } else {
        std::vector<vk::TensorDescriptionARM> tensorDescriptions;
        std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
        tensorDescriptions.reserve(state.bindings.size());
        resourceInfos.reserve(state.bindings.size());
        for (const auto &binding : state.bindings) {
            const auto resource = vgf_.getResource(binding.resourceIndex);
            tensorDescriptions.emplace_back(vk::TensorTilingARM::eLinear, resource.format,
                                            static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                                            resource.stride.empty() ? nullptr : resource.stride.data(),
                                            vk::TensorUsageFlagBitsARM::eDataGraph);
            resourceInfos.emplace_back(binding.set, binding.binding, 0, &tensorDescriptions.back());
        }

        const uint32_t numConstants = vgf_.getNumConstants(segmentIndex);
        std::vector<vk::TensorDescriptionARM> constantTensorDescriptions;
        std::vector<vk::DataGraphPipelineConstantARM> constants;
        constantTensorDescriptions.reserve(numConstants);
        constants.reserve(numConstants);
        for (uint32_t constantIndex = 0; constantIndex < numConstants; ++constantIndex) {
            const auto constant = vgf_.getConstant(segmentIndex, constantIndex);
            if (constant.sparsityDimension >= 0) {
                throw std::runtime_error("Sparse VGF graph constants are not supported");
            }
            constantTensorDescriptions.emplace_back(vk::TensorTilingARM::eLinear, constant.format,
                                                    static_cast<uint32_t>(constant.shape.size()), constant.shape.data(),
                                                    constant.stride.empty() ? nullptr : constant.stride.data(),
                                                    vk::TensorUsageFlagBitsARM::eDataGraph);
            constants.emplace_back(constant.index, constant.data.data(), &constantTensorDescriptions.back());
        }

        const vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleInfo(
            *state.shaderModule, module.entryPoint.c_str(), nullptr, static_cast<uint32_t>(constants.size()),
            constants.data(), nullptr);
        const vk::DataGraphPipelineCreateInfoARM pipelineCreateInfo({}, *state.pipelineLayout,
                                                                    static_cast<uint32_t>(resourceInfos.size()),
                                                                    resourceInfos.data(), &shaderModuleInfo);
        const vk::raii::DeferredOperationKHR deferredOperation(nullptr);
        const vk::raii::PipelineCache *pipelineCache = nullptr;
        state.pipeline = vk::raii::Pipeline(device_, deferredOperation, pipelineCache, pipelineCreateInfo);

        const vk::DataGraphPipelineSessionCreateInfoARM sessionCreateInfo({}, *state.pipeline);
        state.graphSession = vk::raii::DataGraphPipelineSessionARM(device_, sessionCreateInfo);

        const vk::DataGraphPipelineSessionBindPointRequirementsInfoARM bindPointInfo(*state.graphSession);
        const auto bindPointRequirements = device_.getDataGraphPipelineSessionBindPointRequirementsARM(bindPointInfo);
        std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM> bindInfos;
        for (const auto &bindPointRequirement : bindPointRequirements) {
            if (bindPointRequirement.bindPointType != vk::DataGraphPipelineSessionBindPointTypeARM::eMemory) {
                continue;
            }

            for (uint32_t objectIndex = 0; objectIndex < bindPointRequirement.numObjects; ++objectIndex) {
                const vk::DataGraphPipelineSessionMemoryRequirementsInfoARM memoryInfo(
                    *state.graphSession, bindPointRequirement.bindPoint, objectIndex);
                const auto memReqs = device_.getDataGraphPipelineSessionMemoryRequirementsARM(memoryInfo);
                if (memReqs.memoryRequirements.size == 0) {
                    continue;
                }

                const auto memoryType = findMemoryType(physicalDevice_, memReqs.memoryRequirements.memoryTypeBits,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal);
                const vk::MemoryAllocateInfo allocateInfo(memReqs.memoryRequirements.size, memoryType);
                state.sessionMemory.emplace_back(device_, allocateInfo);
                bindInfos.emplace_back(*state.graphSession, bindPointRequirement.bindPoint, objectIndex,
                                       *state.sessionMemory.back());
            }
        }
        if (!bindInfos.empty()) {
            device_.bindDataGraphPipelineSessionMemoryARM(bindInfos);
        }
    }

    std::map<vk::DescriptorType, uint32_t> descriptorCounts;
    for (const auto &binding : state.bindings) {
        ++descriptorCounts[binding.descriptorType];
    }
    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.reserve(descriptorCounts.size());
    for (const auto &[type, count] : descriptorCounts) {
        poolSizes.emplace_back(type, count);
    }
    state.descriptorPool =
        vk::raii::DescriptorPool(device_, {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                           static_cast<uint32_t>(state.descriptorSetLayouts.size()), poolSizes});
    state.descriptorSets = device_.allocateDescriptorSets({*state.descriptorPool, descriptorSetLayouts});
}

void Session::configure() {
    if (configured_) {
        throw std::runtime_error("Session::configure() must only be called once");
    }

    segments_.reserve(vgf_.getNumSegments());
    for (uint32_t segmentIndex = 0; segmentIndex < vgf_.getNumSegments(); ++segmentIndex) {
        configureSegment(segmentIndex);
    }

    commandPool_ =
        vk::raii::CommandPool(device_, {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex_});
    commandBuffer_ =
        std::move(device_.allocateCommandBuffers({*commandPool_, vk::CommandBufferLevel::ePrimary, 1}).front());
    fence_ = vk::raii::Fence(device_, {vk::FenceCreateFlagBits::eSignaled});
    configured_ = true;
}

void Session::run() {
    if (!configured_) {
        throw std::runtime_error("Session::configure() must be called before Session::run()");
    }

    for (const auto &segment : segments_) {
        updateDescriptorSets(segment.descriptorSets, segment.bindings);
    }

    waitForFence(device_, fence_);
    device_.resetFences(*fence_);
    commandBuffer_.reset();

    commandBuffer_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    for (size_t segmentIndex = 0; segmentIndex < segments_.size(); ++segmentIndex) {
        const auto &segment = segments_[segmentIndex];
        const auto pipelineBindPoint = bindPoint(segment.type);
        for (uint32_t set = 0; set < static_cast<uint32_t>(segment.descriptorSets.size()); ++set) {
            commandBuffer_.bindDescriptorSets(pipelineBindPoint, *segment.pipelineLayout, set,
                                              *segment.descriptorSets[set], nullptr);
        }
        commandBuffer_.bindPipeline(pipelineBindPoint, *segment.pipeline);
        if (segment.type == ModuleType::GRAPH) {
            commandBuffer_.dispatchDataGraphARM(*segment.graphSession);
        } else {
            commandBuffer_.dispatch(segment.dispatchShape[0], segment.dispatchShape[1], segment.dispatchShape[2]);
        }

        if (segmentIndex + 1 < segments_.size()) {
            insertSegmentBarrier(commandBuffer_, segment, segments_[segmentIndex + 1]);
        }
    }
    commandBuffer_.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer_);
    queue_.submit(submitInfo, *fence_);
    waitForFence(device_, fence_);
}

} // namespace mlsdk::vgf_runtime
