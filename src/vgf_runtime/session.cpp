/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "session.hpp"

#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace mlsdk::vgf_runtime {
namespace {

// Only support single segment vgf
constexpr uint32_t segmentIndex = 0;

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

template <typename BoundTensors>
void updateDescriptorSets(const vk::raii::Device &device, const std::vector<vk::raii::DescriptorSet> &descriptorSets,
                          const BoundTensors &boundTensors) {
    for (const auto &boundTensor : boundTensors) {
        const auto tensorView = *boundTensor.tensorView;
        const vk::WriteDescriptorSetTensorARM tensorInfo(1, &tensorView);
        const vk::WriteDescriptorSet write(*descriptorSets[boundTensor.binding.set], boundTensor.binding.binding, 0, 1,
                                           boundTensor.binding.descriptorType, nullptr, nullptr, nullptr, &tensorInfo);
        device.updateDescriptorSets(write, nullptr);
    }
}

} // namespace

Session::Session(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device,
                 uint32_t queueFamilyIndex, const vk::raii::Queue &queue, const VGF &vgf)
    : physicalDevice_(physicalDevice), device_(device), vgf_(vgf), queueFamilyIndex_(queueFamilyIndex), queue_(queue) {}

Session::~Session() = default;

void Session::bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding) {
    const auto resource = vgf_.getResource(binding.resourceIndex);
    const vk::TensorViewCreateInfoARM viewCreateInfo({}, *tensor, resource.format);
    boundTensors_.push_back({binding, vk::raii::TensorViewARM(device_, viewCreateInfo)});
}

void Session::configure() {
    if (configured_) {
        throw std::runtime_error("Session::configure() must only be called once");
    }
    if (vgf_.getNumSegments() != 1) {
        throw std::runtime_error("Session only supports a single VGF segment");
    }

    const auto segment = vgf_.getSegment(segmentIndex);
    if (segment.type != ModuleType::GRAPH) {
        throw std::runtime_error("Session only supports VGF data graph segments");
    }
    if (vgf_.getNumConstants(segmentIndex) != 0) {
        throw std::runtime_error("Session does not support VGF graph constants");
    }
    const auto bindings = vgf_.getDescriptorBindings(segmentIndex);

    const auto bindingSets = splitBindingsBySet(bindings);
    descriptorSetLayouts_.reserve(bindingSets.size());
    for (const auto &setBindings : bindingSets) {
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
        layoutBindings.reserve(setBindings.size());
        for (const auto &binding : setBindings) {
            layoutBindings.emplace_back(binding.binding, binding.descriptorType, 1, vk::ShaderStageFlagBits::eAll);
        }
        descriptorSetLayouts_.emplace_back(device_, vk::DescriptorSetLayoutCreateInfo({}, layoutBindings));
    }

    const auto descriptorSetLayouts = rawLayouts(descriptorSetLayouts_);
    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, descriptorSetLayouts);
    pipelineLayout_ = vk::raii::PipelineLayout(device_, pipelineLayoutCreateInfo);

    std::vector<vk::TensorDescriptionARM> tensorDescriptions;
    std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
    tensorDescriptions.reserve(bindings.size());
    resourceInfos.reserve(bindings.size());
    for (const auto &binding : bindings) {
        const auto resource = vgf_.getResource(binding.resourceIndex);
        tensorDescriptions.emplace_back(vk::TensorTilingARM::eLinear, resource.format,
                                        static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                                        resource.stride.empty() ? nullptr : resource.stride.data(),
                                        vk::TensorUsageFlagBitsARM::eDataGraph);
        resourceInfos.emplace_back(binding.set, binding.binding, 0, &tensorDescriptions.back());
    }

    const auto module = vgf_.getSPIRVModule(segment.moduleIndex);
    const vk::ShaderModuleCreateInfo shaderCreateInfo({}, module.code.size() * sizeof(uint32_t), module.code.data());
    shaderModule_ = vk::raii::ShaderModule(device_, shaderCreateInfo);

    const vk::DataGraphPipelineShaderModuleCreateInfoARM shaderModuleInfo(*shaderModule_, module.entryPoint.c_str(),
                                                                          nullptr, 0, nullptr, nullptr);
    const vk::DataGraphPipelineCreateInfoARM pipelineCreateInfo(
        {}, *pipelineLayout_, static_cast<uint32_t>(resourceInfos.size()), resourceInfos.data(), &shaderModuleInfo);
    const vk::raii::DeferredOperationKHR deferredOperation(nullptr);
    const vk::raii::PipelineCache *pipelineCache = nullptr;
    pipeline_ = vk::raii::Pipeline(device_, deferredOperation, pipelineCache, pipelineCreateInfo);

    const vk::DataGraphPipelineSessionCreateInfoARM sessionCreateInfo({}, *pipeline_);
    graphSession_ = vk::raii::DataGraphPipelineSessionARM(device_, sessionCreateInfo);

    const vk::DataGraphPipelineSessionBindPointRequirementsInfoARM bindPointInfo(*graphSession_);
    const auto bindPointRequirements = device_.getDataGraphPipelineSessionBindPointRequirementsARM(bindPointInfo);
    std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM> bindInfos;
    for (const auto &bindPointRequirement : bindPointRequirements) {
        if (bindPointRequirement.bindPointType != vk::DataGraphPipelineSessionBindPointTypeARM::eMemory) {
            continue;
        }

        for (uint32_t objectIndex = 0; objectIndex < bindPointRequirement.numObjects; ++objectIndex) {
            const vk::DataGraphPipelineSessionMemoryRequirementsInfoARM memoryInfo(
                *graphSession_, bindPointRequirement.bindPoint, objectIndex);
            const auto memReqs = device_.getDataGraphPipelineSessionMemoryRequirementsARM(memoryInfo);
            if (memReqs.memoryRequirements.size == 0) {
                continue;
            }

            const auto memoryType = findMemoryType(physicalDevice_, memReqs.memoryRequirements.memoryTypeBits,
                                                   vk::MemoryPropertyFlagBits::eDeviceLocal);
            const vk::MemoryAllocateInfo allocateInfo(memReqs.memoryRequirements.size, memoryType);
            sessionMemory_.emplace_back(device_, allocateInfo);
            bindInfos.emplace_back(*graphSession_, bindPointRequirement.bindPoint, objectIndex, *sessionMemory_.back());
        }
    }
    if (!bindInfos.empty()) {
        device_.bindDataGraphPipelineSessionMemoryARM(bindInfos);
    }

    std::map<vk::DescriptorType, uint32_t> descriptorCounts;
    for (const auto &binding : bindings) {
        ++descriptorCounts[binding.descriptorType];
    }
    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.reserve(descriptorCounts.size());
    for (const auto &[type, count] : descriptorCounts) {
        poolSizes.emplace_back(type, count);
    }
    descriptorPool_ =
        vk::raii::DescriptorPool(device_, {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                           static_cast<uint32_t>(descriptorSetLayouts_.size()), poolSizes});
    descriptorSets_ = device_.allocateDescriptorSets({*descriptorPool_, descriptorSetLayouts});

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

    updateDescriptorSets(device_, descriptorSets_, boundTensors_);

    waitForFence(device_, fence_);
    device_.resetFences(*fence_);
    commandBuffer_.reset();

    commandBuffer_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    for (uint32_t set = 0; set < static_cast<uint32_t>(descriptorSets_.size()); ++set) {
        commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eDataGraphARM, *pipelineLayout_, set,
                                          *descriptorSets_[set], nullptr);
    }
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eDataGraphARM, *pipeline_);
    commandBuffer_.dispatchDataGraphARM(*graphSession_);
    commandBuffer_.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer_);
    queue_.submit(submitInfo, *fence_);
    waitForFence(device_, fence_);
}

} // namespace mlsdk::vgf_runtime
