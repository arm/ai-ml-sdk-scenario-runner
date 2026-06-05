/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <vgf_runtime/runtime.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

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

vk::PipelineBindPoint bindPoint(vgflib::ModuleType type) {
    switch (type) {
    case vgflib::ModuleType::GRAPH:
        return vk::PipelineBindPoint::eDataGraphARM;
    case vgflib::ModuleType::COMPUTE:
        return vk::PipelineBindPoint::eCompute;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

vk::PipelineStageFlags2 pipelineStage(vgflib::ModuleType type) {
    switch (type) {
    case vgflib::ModuleType::GRAPH:
        return vk::PipelineStageFlagBits2::eDataGraphARM;
    case vgflib::ModuleType::COMPUTE:
        return vk::PipelineStageFlagBits2::eComputeShader;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

vk::AccessFlags2 readAccess(vgflib::ModuleType type) {
    switch (type) {
    case vgflib::ModuleType::GRAPH:
        return vk::AccessFlagBits2::eDataGraphReadARM;
    case vgflib::ModuleType::COMPUTE:
        return vk::AccessFlagBits2::eShaderRead;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

vk::AccessFlags2 writeAccess(vgflib::ModuleType type) {
    switch (type) {
    case vgflib::ModuleType::GRAPH:
        return vk::AccessFlagBits2::eDataGraphWriteARM;
    case vgflib::ModuleType::COMPUTE:
        return vk::AccessFlagBits2::eShaderWrite;
    default:
        throw std::runtime_error("Unsupported VGF module type");
    }
}

vk::DeviceSize formatByteSize(vk::Format format) {
    switch (format) {
    case vk::Format::eR8Sint:
        return 1;
    case vk::Format::eR32Sint:
    case vk::Format::eR32Sfloat:
        return 4;
    default:
        throw std::runtime_error("Session does not support storage buffer format " +
                                 std::to_string(static_cast<uint32_t>(format)));
    }
}

vk::DeviceSize resourceByteSize(const ResourceInfo &resource) {
    const auto elementSize = formatByteSize(resource.format);

    if (!resource.stride.empty()) {
        vk::DeviceSize size = elementSize;
        for (uint32_t i = 0; i < resource.shape.size(); ++i) {
            size +=
                static_cast<vk::DeviceSize>(resource.shape[i] - 1) * static_cast<vk::DeviceSize>(resource.stride[i]);
        }
        return size;
    }

    vk::DeviceSize elements = elementSize;
    for (const int64_t dimension : resource.shape) {
        elements *= static_cast<vk::DeviceSize>(dimension);
    }
    return elements;
}

std::string resourceCategoryName(vgflib::ResourceCategory category) {
    switch (category) {
    case vgflib::ResourceCategory::INPUT:
        return "input";
    case vgflib::ResourceCategory::OUTPUT:
        return "output";
    case vgflib::ResourceCategory::INTERMEDIATE:
        return "intermediate";
    case vgflib::ResourceCategory::CONSTANT:
        return "constant";
    default:
        return "unknown";
    }
}

} // namespace

struct Session::Impl {
    struct BoundTensor {
        DescriptorBindingInfo binding;
        vk::TensorARM tensor{nullptr};
        vk::raii::TensorViewARM tensorView{nullptr};
        BoundMemoryInfo memory{};
    };

    struct BoundBuffer {
        DescriptorBindingInfo binding;
        vk::Buffer buffer{nullptr};
        BoundMemoryInfo memory{};
    };

    struct SegmentState {
        // Common members
        vgflib::ModuleType type;
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

    Impl(const vk::raii::PhysicalDevice &physicalDeviceIn, const vk::raii::Device &deviceIn,
         uint32_t queueFamilyIndexIn, const vk::raii::Queue &queueIn, const VGF &vgfIn)
        : physicalDevice(physicalDeviceIn), device(deviceIn), vgf(vgfIn), queueFamilyIndex(queueFamilyIndexIn),
          queue(queueIn) {}

    const BoundTensor *findBoundTensor(uint32_t resourceIndex) const;
    const BoundBuffer *findBoundBuffer(uint32_t resourceIndex) const;
    const BoundTensor *findBoundTensorInAliasGroup(uint32_t aliasGroupId) const;
    const BoundBuffer *findBoundBufferInAliasGroup(uint32_t aliasGroupId) const;
    void updateDescriptorSets(const std::vector<vk::raii::DescriptorSet> &descriptorSets,
                              const std::vector<DescriptorBindingInfo> &bindings) const;
    void insertSegmentBarrier(vk::raii::CommandBuffer &commandBuffer, const SegmentState &producer,
                              const SegmentState &consumer) const;
    void configureSegment(uint32_t segmentIndex);
    void allocateResources();
    vk::raii::TensorARM createIntermediateTensor(const DescriptorBindingInfo &binding) const;
    vk::raii::Buffer createIntermediateBuffer(const DescriptorBindingInfo &binding) const;
    void allocateIntermediateTensor(const DescriptorBindingInfo &binding);
    void allocateIntermediateBuffer(const DescriptorBindingInfo &binding);
    void allocateAliasedResources(const std::vector<DescriptorBindingInfo> &bindings);
    void addBoundTensor(vk::TensorARM tensor, DescriptorBindingInfo binding,
                        BoundMemoryInfo memory = BoundMemoryInfo());
    void addBoundBuffer(vk::Buffer buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory = BoundMemoryInfo());

    void bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding, BoundMemoryInfo memory);
    void bindBuffer(const vk::raii::Buffer &buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory);

    void configure();

    void run();

    const vk::raii::PhysicalDevice &physicalDevice;
    const vk::raii::Device &device;
    const VGF &vgf;

    uint32_t queueFamilyIndex = 0;
    const vk::raii::Queue &queue;

    std::vector<vk::raii::DeviceMemory> ownedMemory;
    std::vector<vk::raii::TensorARM> ownedTensors;
    std::vector<vk::raii::Buffer> ownedBuffers;
    std::vector<BoundTensor> boundTensors;
    std::vector<BoundBuffer> boundBuffers;
    std::vector<SegmentState> segments;

    vk::raii::CommandPool commandPool{nullptr};
    vk::raii::CommandBuffer commandBuffer{nullptr};
    vk::raii::Fence fence{nullptr};
    bool configured = false;
};

Session::Session(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device,
                 uint32_t queueFamilyIndex, const vk::raii::Queue &queue, const VGF &vgf)
    : impl_(std::make_unique<Impl>(physicalDevice, device, queueFamilyIndex, queue, vgf)) {}

Session::~Session() = default;

const Session::Impl::BoundTensor *Session::Impl::findBoundTensor(uint32_t resourceIndex) const {
    const auto tensor =
        std::find_if(boundTensors.rbegin(), boundTensors.rend(), [resourceIndex](const auto &boundTensor) {
            return boundTensor.binding.resourceIndex == resourceIndex;
        });
    if (tensor == boundTensors.rend()) {
        throw std::runtime_error("No tensor bound for VGF resource " + std::to_string(resourceIndex));
    }
    return &*tensor;
}

const Session::Impl::BoundBuffer *Session::Impl::findBoundBuffer(uint32_t resourceIndex) const {
    const auto buffer =
        std::find_if(boundBuffers.rbegin(), boundBuffers.rend(), [resourceIndex](const auto &boundBuffer) {
            return boundBuffer.binding.resourceIndex == resourceIndex;
        });
    if (buffer == boundBuffers.rend()) {
        throw std::runtime_error("No buffer bound for VGF resource " + std::to_string(resourceIndex));
    }
    return &*buffer;
}

const Session::Impl::BoundTensor *Session::Impl::findBoundTensorInAliasGroup(uint32_t aliasGroupId) const {
    const auto tensor =
        std::find_if(boundTensors.rbegin(), boundTensors.rend(), [this, aliasGroupId](const auto &boundTensor) {
            const auto resource = vgf.getResource(boundTensor.binding.resourceIndex);
            return resource.aliasGroupId.has_value() && *resource.aliasGroupId == aliasGroupId;
        });
    return tensor != boundTensors.rend() ? &*tensor : nullptr;
}

const Session::Impl::BoundBuffer *Session::Impl::findBoundBufferInAliasGroup(uint32_t aliasGroupId) const {
    const auto buffer =
        std::find_if(boundBuffers.rbegin(), boundBuffers.rend(), [this, aliasGroupId](const auto &boundBuffer) {
            const auto resource = vgf.getResource(boundBuffer.binding.resourceIndex);
            return resource.aliasGroupId.has_value() && *resource.aliasGroupId == aliasGroupId;
        });
    return buffer != boundBuffers.rend() ? &*buffer : nullptr;
}

void Session::Impl::updateDescriptorSets(const std::vector<vk::raii::DescriptorSet> &descriptorSets,
                                         const std::vector<DescriptorBindingInfo> &bindings) const {
    for (const auto &binding : bindings) {
        switch (binding.descriptorType) {
        case vk::DescriptorType::eTensorARM: {
            const auto *const tensor = findBoundTensor(binding.resourceIndex);
            const auto tensorView = *tensor->tensorView;
            const vk::WriteDescriptorSetTensorARM tensorInfo(1, &tensorView);
            const vk::WriteDescriptorSet write(*descriptorSets[binding.set], binding.binding, 0, 1,
                                               binding.descriptorType, nullptr, nullptr, nullptr, &tensorInfo);
            device.updateDescriptorSets(write, nullptr);
            break;
        }
        case vk::DescriptorType::eStorageBuffer: {
            const auto *const buffer = findBoundBuffer(binding.resourceIndex);
            const vk::DescriptorBufferInfo bufferInfo(buffer->buffer, 0, vk::WholeSize);
            const vk::WriteDescriptorSet write(*descriptorSets[binding.set], binding.binding, 0, 1,
                                               binding.descriptorType, nullptr, &bufferInfo);
            device.updateDescriptorSets(write, nullptr);
            break;
        }
        default:
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(binding.descriptorType)));
        }
    }
}

void Session::Impl::insertSegmentBarrier(vk::raii::CommandBuffer &commandBuffer, const SegmentState &producer,
                                         const SegmentState &consumer) const {
    std::vector<vk::MemoryBarrier2> memoryBarriers;
    std::vector<vk::TensorMemoryBarrierARM> tensorBarriers;
    std::vector<vk::BufferMemoryBarrier2> bufferBarriers;
    std::vector<uint32_t> barrierAliasGroupIds;
    std::vector<uint32_t> barrierResourceIndices;

    for (const auto &producerBinding : producer.bindings) {
        if ((producerBinding.resourceCategory != vgflib::ResourceCategory::OUTPUT &&
             producerBinding.resourceCategory != vgflib::ResourceCategory::INTERMEDIATE) ||
            std::find(barrierResourceIndices.begin(), barrierResourceIndices.end(), producerBinding.resourceIndex) !=
                barrierResourceIndices.end()) {
            continue;
        }

        const auto resource = vgf.getResource(producerBinding.resourceIndex);
        if (resource.aliasGroupId.has_value()) {
            if (std::find(barrierAliasGroupIds.begin(), barrierAliasGroupIds.end(), *resource.aliasGroupId) !=
                barrierAliasGroupIds.end()) {
                continue;
            }

            vk::MemoryBarrier2 memoryBarrier;
            memoryBarrier.srcStageMask = pipelineStage(producer.type);
            memoryBarrier.srcAccessMask = writeAccess(producer.type);
            memoryBarrier.dstStageMask = pipelineStage(consumer.type);
            memoryBarrier.dstAccessMask = readAccess(consumer.type) | writeAccess(consumer.type);

            memoryBarriers.push_back(memoryBarrier);
            barrierAliasGroupIds.push_back(*resource.aliasGroupId);
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

    if (memoryBarriers.empty() && tensorBarriers.empty() && bufferBarriers.empty()) {
        return;
    }

    vk::TensorDependencyInfoARM tensorDependencyInfo(static_cast<uint32_t>(tensorBarriers.size()),
                                                     tensorBarriers.data());
    vk::DependencyInfo dependencyInfo;
    dependencyInfo.memoryBarrierCount = static_cast<uint32_t>(memoryBarriers.size());
    dependencyInfo.pMemoryBarriers = memoryBarriers.data();
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = bufferBarriers.data();
    if (!tensorBarriers.empty()) {
        dependencyInfo.pNext = &tensorDependencyInfo;
    }
    commandBuffer.pipelineBarrier2(dependencyInfo);
}

void Session::Impl::addBoundTensor(vk::TensorARM tensor, DescriptorBindingInfo binding, BoundMemoryInfo memory) {
    const auto resource = vgf.getResource(binding.resourceIndex);
    const vk::TensorViewCreateInfoARM viewCreateInfo({}, tensor, resource.format);
    boundTensors.push_back({binding, tensor, vk::raii::TensorViewARM(device, viewCreateInfo), memory});
}

void Session::Impl::addBoundBuffer(vk::Buffer buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory) {
    boundBuffers.push_back({binding, buffer, memory});
}

void Session::Impl::bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding,
                               BoundMemoryInfo memory) {
    const auto resource = vgf.getResource(binding.resourceIndex);
    if (resource.category != vgflib::ResourceCategory::INPUT && resource.category != vgflib::ResourceCategory::OUTPUT) {
        throw std::runtime_error(std::string("VGF ") + resourceCategoryName(resource.category) + " resource " +
                                 std::to_string(binding.resourceIndex) + " must not be manually bound");
    }
    addBoundTensor(*tensor, binding, memory);
}

void Session::Impl::bindBuffer(const vk::raii::Buffer &buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory) {
    const auto resource = vgf.getResource(binding.resourceIndex);
    if (resource.category != vgflib::ResourceCategory::INPUT && resource.category != vgflib::ResourceCategory::OUTPUT) {
        throw std::runtime_error(std::string("VGF ") + resourceCategoryName(resource.category) + " resource " +
                                 std::to_string(binding.resourceIndex) + " must not be manually bound");
    }
    addBoundBuffer(*buffer, binding, memory);
}

vk::raii::TensorARM Session::Impl::createIntermediateTensor(const DescriptorBindingInfo &binding) const {
    const auto resource = vgf.getResource(binding.resourceIndex);

    const vk::TensorDescriptionARM description(vk::TensorTilingARM::eLinear, resource.format,
                                               static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                                               resource.stride.empty() ? nullptr : resource.stride.data(),
                                               vk::TensorUsageFlagBitsARM::eDataGraph);
    const vk::TensorCreateInfoARM createInfo({}, &description, vk::SharingMode::eExclusive);
    return vk::raii::TensorARM(device, createInfo);
}

vk::raii::Buffer Session::Impl::createIntermediateBuffer(const DescriptorBindingInfo &binding) const {
    const auto resource = vgf.getResource(binding.resourceIndex);
    return vk::raii::Buffer(
        device, vk::BufferCreateInfo({}, resourceByteSize(resource), vk::BufferUsageFlagBits::eStorageBuffer));
}

void Session::Impl::allocateIntermediateTensor(const DescriptorBindingInfo &binding) {
    auto ownedTensor = createIntermediateTensor(binding);
    const auto memoryRequirements =
        device.getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*ownedTensor));
    const auto memoryType = findMemoryType(physicalDevice, memoryRequirements.memoryRequirements.memoryTypeBits,
                                           vk::MemoryPropertyFlagBits::eDeviceLocal);
    ownedMemory.emplace_back(device, vk::MemoryAllocateInfo(memoryRequirements.memoryRequirements.size, memoryType));
    device.bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*ownedTensor, *ownedMemory.back(), 0));

    ownedTensors.push_back(std::move(ownedTensor));
    addBoundTensor(ownedTensors.back(), binding);
}

void Session::Impl::allocateIntermediateBuffer(const DescriptorBindingInfo &binding) {
    auto ownedBuffer = createIntermediateBuffer(binding);
    const auto memoryRequirements = ownedBuffer.getMemoryRequirements();
    const auto memoryType =
        findMemoryType(physicalDevice, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    ownedMemory.emplace_back(device, vk::MemoryAllocateInfo(memoryRequirements.size, memoryType));
    ownedBuffer.bindMemory(*ownedMemory.back(), 0);

    ownedBuffers.push_back(std::move(ownedBuffer));
    addBoundBuffer(ownedBuffers.back(), binding);
}

void Session::Impl::allocateAliasedResources(const std::vector<DescriptorBindingInfo> &bindings) {
    std::vector<std::pair<DescriptorBindingInfo, vk::raii::TensorARM>> tensors;
    std::vector<std::pair<DescriptorBindingInfo, vk::raii::Buffer>> buffers;

    const auto aliasGroupId = *vgf.getResource(bindings.front().resourceIndex).aliasGroupId;

    uint32_t memoryTypeBits = std::numeric_limits<uint32_t>::max();
    vk::DeviceSize memorySize = 0;
    for (const auto &binding : bindings) {
        if (binding.descriptorType == vk::DescriptorType::eTensorARM) {
            // Add already bound input/output tensor
            if (const auto *const aliasedTensor = findBoundTensorInAliasGroup(aliasGroupId)) {
                addBoundTensor(aliasedTensor->tensor, binding, aliasedTensor->memory);
                continue;
            }
            if (binding.resourceCategory != vgflib::ResourceCategory::INTERMEDIATE) {
                throw std::runtime_error("No manually bound tensor for aliased VGF resource " +
                                         std::to_string(binding.resourceIndex));
            }
            const auto *const aliasedBuffer = findBoundBufferInAliasGroup(aliasGroupId);
            auto ownedTensor = createIntermediateTensor(binding);
            // If buffer in alias group already bound
            if (aliasedBuffer != nullptr) {
                if (aliasedBuffer->memory.memory == nullptr) {
                    throw std::runtime_error("Manually bound buffer for aliased VGF resource " +
                                             std::to_string(aliasedBuffer->binding.resourceIndex) +
                                             " must include memory info");
                }
                device.bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*ownedTensor, aliasedBuffer->memory.memory,
                                                                       aliasedBuffer->memory.offset));
                ownedTensors.push_back(std::move(ownedTensor));
                addBoundTensor(*ownedTensors.back(), binding, aliasedBuffer->memory);
            } else {
                const auto memoryRequirements =
                    device.getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*ownedTensor))
                        .memoryRequirements;
                memoryTypeBits &= memoryRequirements.memoryTypeBits;
                memorySize = std::max(memorySize, memoryRequirements.size);
                tensors.emplace_back(binding, std::move(ownedTensor));
            }
        } else if (binding.descriptorType == vk::DescriptorType::eStorageBuffer) {
            // Add already bound input/output buffer
            if (const auto *const aliasedBuffer = findBoundBufferInAliasGroup(aliasGroupId)) {
                addBoundBuffer(aliasedBuffer->buffer, binding, aliasedBuffer->memory);
                continue;
            }
            if (binding.resourceCategory != vgflib::ResourceCategory::INTERMEDIATE) {
                throw std::runtime_error("No manually bound buffer for aliased VGF resource " +
                                         std::to_string(binding.resourceIndex));
            }
            const auto *const aliasedTensor = findBoundTensorInAliasGroup(aliasGroupId);
            auto ownedBuffer = createIntermediateBuffer(binding);
            // If tensor in alias group already bound
            if (aliasedTensor != nullptr) {
                if (aliasedTensor->memory.memory == nullptr) {
                    throw std::runtime_error("Manually bound tensor for aliased VGF resource " +
                                             std::to_string(aliasedTensor->binding.resourceIndex) +
                                             " must include memory info");
                }
                ownedBuffer.bindMemory(aliasedTensor->memory.memory, aliasedTensor->memory.offset);
                ownedBuffers.push_back(std::move(ownedBuffer));
                addBoundBuffer(*ownedBuffers.back(), binding, aliasedTensor->memory);
            } else {
                const auto memoryRequirements = ownedBuffer.getMemoryRequirements();
                memoryTypeBits &= memoryRequirements.memoryTypeBits;
                memorySize = std::max(memorySize, memoryRequirements.size);
                buffers.emplace_back(binding, std::move(ownedBuffer));
            }
        } else {
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(binding.descriptorType)));
        }
    }

    if (tensors.empty() && buffers.empty()) {
        return;
    }

    // Allocate and add fully intermediate aliased resources
    const auto memoryType = findMemoryType(physicalDevice, memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    ownedMemory.emplace_back(device, vk::MemoryAllocateInfo(memorySize, memoryType));
    const vk::DeviceMemory memory = *ownedMemory.back();

    for (auto &[binding, ownedTensor] : tensors) {
        device.bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*ownedTensor, memory, 0));
        ownedTensors.push_back(std::move(ownedTensor));
        addBoundTensor(*ownedTensors.back(), binding, {*ownedMemory.back(), 0, memorySize});
    }

    for (auto &[binding, ownedBuffer] : buffers) {
        ownedBuffer.bindMemory(memory, 0);
        ownedBuffers.push_back(std::move(ownedBuffer));
        addBoundBuffer(*ownedBuffers.back(), binding, {*ownedMemory.back(), 0, memorySize});
    }
}

void Session::Impl::allocateResources() {
    std::vector<uint32_t> allocatedResourceIndices;
    std::map<uint32_t, std::vector<DescriptorBindingInfo>> aliasGroups;
    for (const auto &segment : segments) {
        for (const auto &binding : segment.bindings) {
            // Skip if already present
            if (std::find(allocatedResourceIndices.begin(), allocatedResourceIndices.end(), binding.resourceIndex) !=
                allocatedResourceIndices.end()) {
                continue;
            }

            // Only allocate non-aliased resources in current loop
            const auto resource = vgf.getResource(binding.resourceIndex);
            if (binding.resourceCategory != vgflib::ResourceCategory::INTERMEDIATE) {
                // If aliased Input/Output
                if (resource.aliasGroupId.has_value()) {
                    aliasGroups[*resource.aliasGroupId].push_back(binding);
                }
                allocatedResourceIndices.push_back(binding.resourceIndex);
                continue;
            }

            if (resource.aliasGroupId.has_value()) {
                aliasGroups[*resource.aliasGroupId].push_back(binding);
                allocatedResourceIndices.push_back(binding.resourceIndex);
                continue;
            }

            switch (binding.descriptorType) {
            case vk::DescriptorType::eTensorARM:
                allocateIntermediateTensor(binding);
                break;
            case vk::DescriptorType::eStorageBuffer:
                allocateIntermediateBuffer(binding);
                break;
            default:
                throw std::runtime_error("Session does not support descriptor type " +
                                         std::to_string(static_cast<uint32_t>(binding.descriptorType)));
            }
            allocatedResourceIndices.push_back(binding.resourceIndex);
        }
    }

    for (const auto &aliasGroup : aliasGroups) {
        allocateAliasedResources(aliasGroup.second);
    }
}

void Session::Impl::configureSegment(uint32_t segmentIndex) {
    const auto segment = vgf.getSegment(segmentIndex);
    if (segment.type != vgflib::ModuleType::GRAPH && segment.type != vgflib::ModuleType::COMPUTE) {
        throw std::runtime_error("Session only supports VGF data graph and compute shader segments");
    }

    auto &state = segments.emplace_back();
    state.type = segment.type;
    state.bindings = vgf.getDescriptorBindings(segmentIndex);
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
        state.descriptorSetLayouts.emplace_back(device, vk::DescriptorSetLayoutCreateInfo({}, layoutBindings));
    }

    const auto descriptorSetLayouts = rawLayouts(state.descriptorSetLayouts);
    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, descriptorSetLayouts);
    state.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCreateInfo);

    const auto module = vgf.getSPIRVModule(segment.moduleIndex);
    const vk::ShaderModuleCreateInfo shaderCreateInfo({}, module.code.size() * sizeof(uint32_t), module.code.data());
    state.shaderModule = vk::raii::ShaderModule(device, shaderCreateInfo);

    if (segment.type == vgflib::ModuleType::COMPUTE) {
        const auto dispatchShape = vgf.getDispatchShape(segmentIndex);
        if (dispatchShape.size() != state.dispatchShape.size() || dispatchShape[0] == 0 || dispatchShape[1] == 0 ||
            dispatchShape[2] == 0) {
            throw std::runtime_error("VGF compute shader segments must have a non-zero 3D dispatch shape");
        }
        std::copy(dispatchShape.begin(), dispatchShape.end(), state.dispatchShape.begin());

        const vk::PipelineShaderStageCreateInfo shaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                                      *state.shaderModule, module.entryPoint.c_str());
        const vk::ComputePipelineCreateInfo pipelineCreateInfo({}, shaderStageCreateInfo, *state.pipelineLayout);
        const vk::raii::PipelineCache *pipelineCache = nullptr;
        state.pipeline = vk::raii::Pipeline(device, pipelineCache, pipelineCreateInfo);
    } else {
        std::vector<vk::TensorDescriptionARM> tensorDescriptions;
        std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
        tensorDescriptions.reserve(state.bindings.size());
        resourceInfos.reserve(state.bindings.size());
        for (const auto &binding : state.bindings) {
            const auto resource = vgf.getResource(binding.resourceIndex);
            tensorDescriptions.emplace_back(vk::TensorTilingARM::eLinear, resource.format,
                                            static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                                            resource.stride.empty() ? nullptr : resource.stride.data(),
                                            vk::TensorUsageFlagBitsARM::eDataGraph);
            resourceInfos.emplace_back(binding.set, binding.binding, 0, &tensorDescriptions.back());
        }

        const uint32_t numConstants = vgf.getNumConstants(segmentIndex);
        std::vector<vk::TensorDescriptionARM> constantTensorDescriptions;
        std::vector<vk::DataGraphPipelineConstantARM> constants;
        constantTensorDescriptions.reserve(numConstants);
        constants.reserve(numConstants);
        for (uint32_t constantIndex = 0; constantIndex < numConstants; ++constantIndex) {
            const auto constant = vgf.getConstant(segmentIndex, constantIndex);
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
        state.pipeline = vk::raii::Pipeline(device, deferredOperation, pipelineCache, pipelineCreateInfo);

        const vk::DataGraphPipelineSessionCreateInfoARM sessionCreateInfo({}, *state.pipeline);
        state.graphSession = vk::raii::DataGraphPipelineSessionARM(device, sessionCreateInfo);

        const vk::DataGraphPipelineSessionBindPointRequirementsInfoARM bindPointInfo(*state.graphSession);
        const auto bindPointRequirements = device.getDataGraphPipelineSessionBindPointRequirementsARM(bindPointInfo);
        std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM> bindInfos;
        for (const auto &bindPointRequirement : bindPointRequirements) {
            if (bindPointRequirement.bindPointType != vk::DataGraphPipelineSessionBindPointTypeARM::eMemory) {
                continue;
            }

            for (uint32_t objectIndex = 0; objectIndex < bindPointRequirement.numObjects; ++objectIndex) {
                const vk::DataGraphPipelineSessionMemoryRequirementsInfoARM memoryInfo(
                    *state.graphSession, bindPointRequirement.bindPoint, objectIndex);
                const auto memReqs = device.getDataGraphPipelineSessionMemoryRequirementsARM(memoryInfo);
                if (memReqs.memoryRequirements.size == 0) {
                    continue;
                }

                const auto memoryType = findMemoryType(physicalDevice, memReqs.memoryRequirements.memoryTypeBits,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal);
                const vk::MemoryAllocateInfo allocateInfo(memReqs.memoryRequirements.size, memoryType);
                state.sessionMemory.emplace_back(device, allocateInfo);
                bindInfos.emplace_back(*state.graphSession, bindPointRequirement.bindPoint, objectIndex,
                                       *state.sessionMemory.back());
            }
        }
        if (!bindInfos.empty()) {
            device.bindDataGraphPipelineSessionMemoryARM(bindInfos);
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
        vk::raii::DescriptorPool(device, {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                          static_cast<uint32_t>(state.descriptorSetLayouts.size()), poolSizes});
    state.descriptorSets = device.allocateDescriptorSets({*state.descriptorPool, descriptorSetLayouts});
}

void Session::Impl::configure() {
    if (configured) {
        throw std::runtime_error("Session::configure() must only be called once");
    }

    segments.reserve(vgf.getNumSegments());
    for (uint32_t segmentIndex = 0; segmentIndex < vgf.getNumSegments(); ++segmentIndex) {
        configureSegment(segmentIndex);
    }
    allocateResources();

    commandPool = vk::raii::CommandPool(device, {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex});
    commandBuffer =
        std::move(device.allocateCommandBuffers({*commandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    fence = vk::raii::Fence(device, {vk::FenceCreateFlagBits::eSignaled});
    configured = true;
}

void Session::Impl::run() {
    if (!configured) {
        throw std::runtime_error("Session::configure() must be called before Session::run()");
    }

    for (const auto &segment : segments) {
        updateDescriptorSets(segment.descriptorSets, segment.bindings);
    }

    waitForFence(device, fence);
    device.resetFences(*fence);
    commandBuffer.reset();

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
        const auto &segment = segments[segmentIndex];
        const auto pipelineBindPoint = bindPoint(segment.type);
        for (uint32_t set = 0; set < static_cast<uint32_t>(segment.descriptorSets.size()); ++set) {
            commandBuffer.bindDescriptorSets(pipelineBindPoint, *segment.pipelineLayout, set,
                                             *segment.descriptorSets[set], nullptr);
        }
        commandBuffer.bindPipeline(pipelineBindPoint, *segment.pipeline);
        if (segment.type == vgflib::ModuleType::GRAPH) {
            commandBuffer.dispatchDataGraphARM(*segment.graphSession);
        } else {
            commandBuffer.dispatch(segment.dispatchShape[0], segment.dispatchShape[1], segment.dispatchShape[2]);
        }

        if (segmentIndex + 1 < segments.size()) {
            insertSegmentBarrier(commandBuffer, segment, segments[segmentIndex + 1]);
        }
    }
    commandBuffer.end();

    const vk::SubmitInfo submitInfo({}, {}, *commandBuffer);
    queue.submit(submitInfo, *fence);
    waitForFence(device, fence);
}

void Session::bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding, BoundMemoryInfo memory) {
    impl_->bindTensor(tensor, binding, memory);
}

void Session::bindBuffer(const vk::raii::Buffer &buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory) {
    impl_->bindBuffer(buffer, binding, memory);
}

void Session::configure() { impl_->configure(); }

void Session::run() { impl_->run(); }

} // namespace mlsdk::vgf_runtime
