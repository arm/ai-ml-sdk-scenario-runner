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

void validateImageFormat(vk::Format format) {
    if (format != vk::Format::eR8G8B8A8Snorm) {
        throw std::runtime_error("Session only supports eR8G8B8A8Snorm VGF image resources");
    }
}

vk::Extent3D imageExtent(const ResourceInfo &resource) {
    validateImageFormat(resource.format);
    if (resource.shape.size() == 4 && resource.shape[0] == 1 && resource.shape[3] == 4) {
        return {static_cast<uint32_t>(resource.shape[2]), static_cast<uint32_t>(resource.shape[1]), 1};
    }
    throw std::runtime_error("Session only supports 4D NHWC VGF image resources with batch 1 and 4 channels");
}

vk::ImageLayout imageLayout(vk::DescriptorType descriptorType) {
    switch (descriptorType) {
    case vk::DescriptorType::eCombinedImageSampler:
        return vk::ImageLayout::eShaderReadOnlyOptimal;
    case vk::DescriptorType::eStorageImage:
        return vk::ImageLayout::eGeneral;
    default:
        throw std::runtime_error("Descriptor type is not an image descriptor");
    }
}

vk::ImageUsageFlags imageUsage(vk::DescriptorType descriptorType, bool aliased) {
    vk::ImageUsageFlags usage{};
    switch (descriptorType) {
    case vk::DescriptorType::eCombinedImageSampler:
        usage |= vk::ImageUsageFlagBits::eSampled;
        break;
    case vk::DescriptorType::eStorageImage:
        usage |= vk::ImageUsageFlagBits::eStorage;
        break;
    default:
        throw std::runtime_error("Descriptor type is not an image descriptor");
    }
    if (aliased) {
        usage |= vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                 vk::ImageUsageFlagBits::eTensorAliasingARM;
    }
    return usage;
}

vk::AccessFlags2 imageAccess(vgflib::ModuleType type, vk::DescriptorType descriptorType) {
    if (descriptorType == vk::DescriptorType::eCombinedImageSampler) {
        return readAccess(type);
    }
    if (descriptorType == vk::DescriptorType::eStorageImage) {
        return readAccess(type) | writeAccess(type);
    }
    throw std::runtime_error("Descriptor type is not an image descriptor");
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

    struct BoundImage {
        DescriptorBindingInfo binding;
        vk::Image image{nullptr};
        vk::raii::ImageView imageView{nullptr};
        vk::raii::Sampler sampler{nullptr};
        BoundMemoryInfo memory{};
        vk::ImageLayout currentLayout = vk::ImageLayout::eUndefined;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
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
    const BoundImage *findBoundImage(uint32_t resourceIndex) const;
    const BoundImage *findBoundImageInAliasGroup(uint32_t aliasGroupId) const;
    void updateDescriptorSets(const std::vector<vk::raii::DescriptorSet> &descriptorSets,
                              const std::vector<DescriptorBindingInfo> &bindings) const;
    void insertInitialImageLayoutTransitions(vk::raii::CommandBuffer &commandBuffer);
    void insertSegmentBarrier(vk::raii::CommandBuffer &commandBuffer, const SegmentState &producer,
                              const SegmentState &consumer) const;
    void configureSegment(uint32_t segmentIndex);
    void allocateResources();
    vk::raii::TensorARM createIntermediateTensor(const DescriptorBindingInfo &binding) const;
    vk::raii::Buffer createIntermediateBuffer(const DescriptorBindingInfo &binding) const;
    vk::raii::Image createIntermediateImage(const DescriptorBindingInfo &binding, bool aliased) const;
    void allocateIntermediateTensor(const DescriptorBindingInfo &binding);
    void allocateIntermediateBuffer(const DescriptorBindingInfo &binding);
    void allocateIntermediateImage(const DescriptorBindingInfo &binding);
    void allocateAliasedResources(const std::vector<DescriptorBindingInfo> &bindings);
    void addBoundTensor(vk::TensorARM tensor, DescriptorBindingInfo binding,
                        BoundMemoryInfo memory = BoundMemoryInfo());
    void addBoundBuffer(vk::Buffer buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory = BoundMemoryInfo());
    void addBoundImage(vk::Image image, DescriptorBindingInfo binding, BoundMemoryInfo memory = BoundMemoryInfo(),
                       vk::ImageLayout currentLayout = vk::ImageLayout::eUndefined);

    void bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding, BoundMemoryInfo memory);
    void bindBuffer(const vk::raii::Buffer &buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory);
    void bindImage(const vk::raii::Image &image, DescriptorBindingInfo binding, BoundMemoryInfo memory,
                   vk::ImageLayout currentLayout);

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
    std::vector<vk::raii::Image> ownedImages;
    std::vector<BoundTensor> boundTensors;
    std::vector<BoundBuffer> boundBuffers;
    std::vector<BoundImage> boundImages;
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

const Session::Impl::BoundImage *Session::Impl::findBoundImage(uint32_t resourceIndex) const {
    const auto image = std::find_if(boundImages.rbegin(), boundImages.rend(), [resourceIndex](const auto &boundImage) {
        return boundImage.binding.resourceIndex == resourceIndex;
    });
    if (image == boundImages.rend()) {
        throw std::runtime_error("No image bound for VGF resource " + std::to_string(resourceIndex));
    }
    return &*image;
}

const Session::Impl::BoundImage *Session::Impl::findBoundImageInAliasGroup(uint32_t aliasGroupId) const {
    const auto image =
        std::find_if(boundImages.rbegin(), boundImages.rend(), [this, aliasGroupId](const auto &boundImage) {
            const auto resource = vgf.getResource(boundImage.binding.resourceIndex);
            return resource.aliasGroupId.has_value() && *resource.aliasGroupId == aliasGroupId;
        });
    return image != boundImages.rend() ? &*image : nullptr;
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
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eStorageImage: {
            const auto *const image = findBoundImage(binding.resourceIndex);
            const vk::DescriptorImageInfo imageInfo(
                binding.descriptorType == vk::DescriptorType::eCombinedImageSampler ? *image->sampler : vk::Sampler(),
                *image->imageView, image->layout);
            const vk::WriteDescriptorSet write(*descriptorSets[binding.set], binding.binding, 0, 1,
                                               binding.descriptorType, &imageInfo);
            device.updateDescriptorSets(write, nullptr);
            break;
        }
        default:
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(binding.descriptorType)));
        }
    }
}

void Session::Impl::insertInitialImageLayoutTransitions(vk::raii::CommandBuffer &commandBuffer) {
    std::vector<vk::ImageMemoryBarrier2> imageBarriers;
    std::vector<vk::Image> transitionedImages;
    imageBarriers.reserve(boundImages.size());

    for (auto &boundImage : boundImages) {
        if (std::find(transitionedImages.begin(), transitionedImages.end(), boundImage.image) !=
            transitionedImages.end()) {
            continue;
        }

        const auto firstConsumer = std::find_if(segments.begin(), segments.end(), [&boundImage](const auto &segment) {
            return std::any_of(segment.bindings.begin(), segment.bindings.end(), [&boundImage](const auto &binding) {
                return binding.resourceIndex == boundImage.binding.resourceIndex;
            });
        });
        if (firstConsumer == segments.end()) {
            throw std::runtime_error("No segment uses VGF image resource " +
                                     std::to_string(boundImage.binding.resourceIndex));
        }

        vk::ImageMemoryBarrier2 imageBarrier;
        imageBarrier.srcStageMask = boundImage.currentLayout == vk::ImageLayout::eUndefined
                                        ? vk::PipelineStageFlagBits2::eTopOfPipe
                                        : vk::PipelineStageFlagBits2::eAllCommands;
        imageBarrier.srcAccessMask = boundImage.currentLayout == vk::ImageLayout::eUndefined
                                         ? vk::AccessFlags2{}
                                         : vk::AccessFlagBits2::eMemoryWrite;
        imageBarrier.dstStageMask = pipelineStage(firstConsumer->type);
        imageBarrier.dstAccessMask = imageAccess(firstConsumer->type, boundImage.binding.descriptorType);
        imageBarrier.oldLayout = boundImage.currentLayout;
        imageBarrier.newLayout = boundImage.layout;
        imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.image = boundImage.image;
        imageBarrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        imageBarriers.push_back(imageBarrier);
        transitionedImages.push_back(boundImage.image);
        boundImage.currentLayout = boundImage.layout;
    }

    if (imageBarriers.empty()) {
        return;
    }

    vk::DependencyInfo dependencyInfo;
    dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
    dependencyInfo.pImageMemoryBarriers = imageBarriers.data();
    commandBuffer.pipelineBarrier2(dependencyInfo);
}

void Session::Impl::insertSegmentBarrier(vk::raii::CommandBuffer &commandBuffer, const SegmentState &producer,
                                         const SegmentState &consumer) const {
    std::vector<vk::MemoryBarrier2> memoryBarriers;
    std::vector<vk::TensorMemoryBarrierARM> tensorBarriers;
    std::vector<vk::BufferMemoryBarrier2> bufferBarriers;
    std::vector<vk::ImageMemoryBarrier2> imageBarriers;
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
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eStorageImage: {
            const auto *const image = findBoundImage(producerBinding.resourceIndex);
            vk::ImageMemoryBarrier2 imageBarrier;
            imageBarrier.srcStageMask = pipelineStage(producer.type);
            imageBarrier.srcAccessMask = writeAccess(producer.type);
            imageBarrier.dstStageMask = pipelineStage(consumer.type);
            imageBarrier.dstAccessMask = readAccess(consumer.type) | writeAccess(consumer.type);
            imageBarrier.oldLayout = image->layout;
            imageBarrier.newLayout = image->layout;
            imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.image = image->image;
            imageBarrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

            imageBarriers.push_back(imageBarrier);
            barrierResourceIndices.push_back(producerBinding.resourceIndex);
            break;
        }
        default:
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(producerBinding.descriptorType)));
        }
    }

    if (memoryBarriers.empty() && tensorBarriers.empty() && bufferBarriers.empty() && imageBarriers.empty()) {
        return;
    }

    vk::TensorDependencyInfoARM tensorDependencyInfo(static_cast<uint32_t>(tensorBarriers.size()),
                                                     tensorBarriers.data());
    vk::DependencyInfo dependencyInfo;
    dependencyInfo.memoryBarrierCount = static_cast<uint32_t>(memoryBarriers.size());
    dependencyInfo.pMemoryBarriers = memoryBarriers.data();
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = bufferBarriers.data();
    dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
    dependencyInfo.pImageMemoryBarriers = imageBarriers.data();
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

void Session::Impl::addBoundImage(vk::Image image, DescriptorBindingInfo binding, BoundMemoryInfo memory,
                                  vk::ImageLayout currentLayout) {
    const auto resource = vgf.getResource(binding.resourceIndex);
    validateImageFormat(resource.format);
    const vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    const vk::ImageViewCreateInfo viewCreateInfo({}, image, vk::ImageViewType::e2D, resource.format, {},
                                                 subresourceRange);

    vk::raii::Sampler sampler(nullptr);
    if (binding.descriptorType == vk::DescriptorType::eCombinedImageSampler) {
        const auto samplerConfig = resource.samplerConfig.value_or(ResourceInfo::SamplerConfig{});
        const vk::SamplerCreateInfo samplerCreateInfo(
            {}, samplerConfig.magFilter, samplerConfig.minFilter, vk::SamplerMipmapMode::eNearest,
            samplerConfig.addressModeU, samplerConfig.addressModeV, vk::SamplerAddressMode::eClampToEdge, 0.0F, false,
            1.0F, false, vk::CompareOp::eNever, 0.0F, 0.0F, samplerConfig.borderColor);
        sampler = vk::raii::Sampler(device, samplerCreateInfo);
    }

    boundImages.push_back({binding, image, vk::raii::ImageView(device, viewCreateInfo), std::move(sampler), memory,
                           currentLayout, imageLayout(binding.descriptorType)});
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

void Session::Impl::bindImage(const vk::raii::Image &image, DescriptorBindingInfo binding, BoundMemoryInfo memory,
                              vk::ImageLayout currentLayout) {
    const auto resource = vgf.getResource(binding.resourceIndex);
    if (resource.category != vgflib::ResourceCategory::INPUT && resource.category != vgflib::ResourceCategory::OUTPUT) {
        throw std::runtime_error(std::string("VGF ") + resourceCategoryName(resource.category) + " resource " +
                                 std::to_string(binding.resourceIndex) + " must not be manually bound");
    }
    addBoundImage(*image, binding, memory, currentLayout);
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

vk::raii::Image Session::Impl::createIntermediateImage(const DescriptorBindingInfo &binding, bool aliased) const {
    const auto resource = vgf.getResource(binding.resourceIndex);
    const vk::ImageCreateInfo createInfo({}, vk::ImageType::e2D, resource.format, imageExtent(resource), 1, 1,
                                         vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                                         imageUsage(binding.descriptorType, aliased), vk::SharingMode::eExclusive, {},
                                         vk::ImageLayout::eUndefined);
    return vk::raii::Image(device, createInfo);
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

void Session::Impl::allocateIntermediateImage(const DescriptorBindingInfo &binding) {
    auto ownedImage = createIntermediateImage(binding, false);
    const auto memoryRequirements = ownedImage.getMemoryRequirements();
    const auto memoryType =
        findMemoryType(physicalDevice, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    ownedMemory.emplace_back(device, vk::MemoryAllocateInfo(memoryRequirements.size, memoryType));
    ownedImage.bindMemory(*ownedMemory.back(), 0);

    ownedImages.push_back(std::move(ownedImage));
    addBoundImage(ownedImages.back(), binding, {*ownedMemory.back(), 0, memoryRequirements.size});
}

void Session::Impl::allocateAliasedResources(const std::vector<DescriptorBindingInfo> &bindings) {
    std::vector<std::pair<DescriptorBindingInfo, vk::raii::TensorARM>> tensors;
    std::vector<std::pair<DescriptorBindingInfo, vk::raii::Buffer>> buffers;
    std::vector<std::pair<DescriptorBindingInfo, vk::raii::Image>> images;

    const auto aliasGroupId = *vgf.getResource(bindings.front().resourceIndex).aliasGroupId;

    std::optional<BoundMemoryInfo> aliasedMemory;
    auto setAliasedMemory = [&](BoundMemoryInfo memory) {
        if (memory.memory != nullptr) {
            aliasedMemory = memory;
        }
    };
    const auto *const aliasedTensor = findBoundTensorInAliasGroup(aliasGroupId);
    const auto *const aliasedBuffer = findBoundBufferInAliasGroup(aliasGroupId);
    const auto *const aliasedImage = findBoundImageInAliasGroup(aliasGroupId);
    if (aliasedTensor != nullptr) {
        setAliasedMemory(aliasedTensor->memory);
    }
    if (aliasedBuffer != nullptr) {
        setAliasedMemory(aliasedBuffer->memory);
    }
    if (aliasedImage != nullptr) {
        setAliasedMemory(aliasedImage->memory);
    }
    const bool hasBoundAlias = aliasedTensor != nullptr || aliasedBuffer != nullptr || aliasedImage != nullptr;
    const auto requireAliasedMemory = [&](const DescriptorBindingInfo &binding) {
        if (hasBoundAlias && !aliasedMemory.has_value()) {
            throw std::runtime_error("Manually bound aliases must provide memory for aliased VGF resource " +
                                     std::to_string(binding.resourceIndex));
        }
    };

    uint32_t memoryTypeBits = std::numeric_limits<uint32_t>::max();
    vk::DeviceSize memorySize = 0;
    for (const auto &binding : bindings) {
        if (binding.descriptorType == vk::DescriptorType::eTensorARM) {
            if (binding.resourceCategory != vgflib::ResourceCategory::INTERMEDIATE) {
                if (aliasedTensor != nullptr) {
                    addBoundTensor(aliasedTensor->tensor, binding, aliasedTensor->memory);
                    continue;
                }
                throw std::runtime_error("No manually bound tensor for aliased VGF resource " +
                                         std::to_string(binding.resourceIndex));
            }
            requireAliasedMemory(binding);
            auto ownedTensor = createIntermediateTensor(binding);
            if (aliasedMemory.has_value()) {
                device.bindTensorMemoryARM(
                    vk::BindTensorMemoryInfoARM(*ownedTensor, aliasedMemory->memory, aliasedMemory->offset));
                ownedTensors.push_back(std::move(ownedTensor));
                addBoundTensor(*ownedTensors.back(), binding, *aliasedMemory);
            } else {
                const auto memoryRequirements =
                    device.getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*ownedTensor))
                        .memoryRequirements;
                memoryTypeBits &= memoryRequirements.memoryTypeBits;
                memorySize = std::max(memorySize, memoryRequirements.size);
                tensors.emplace_back(binding, std::move(ownedTensor));
            }
        } else if (binding.descriptorType == vk::DescriptorType::eStorageBuffer) {
            if (binding.resourceCategory != vgflib::ResourceCategory::INTERMEDIATE) {
                if (aliasedBuffer != nullptr) {
                    addBoundBuffer(aliasedBuffer->buffer, binding, aliasedBuffer->memory);
                    continue;
                }
                throw std::runtime_error("No manually bound buffer for aliased VGF resource " +
                                         std::to_string(binding.resourceIndex));
            }
            requireAliasedMemory(binding);
            auto ownedBuffer = createIntermediateBuffer(binding);
            if (aliasedMemory.has_value()) {
                ownedBuffer.bindMemory(aliasedMemory->memory, aliasedMemory->offset);
                ownedBuffers.push_back(std::move(ownedBuffer));
                addBoundBuffer(*ownedBuffers.back(), binding, *aliasedMemory);
            } else {
                const auto memoryRequirements = ownedBuffer.getMemoryRequirements();
                memoryTypeBits &= memoryRequirements.memoryTypeBits;
                memorySize = std::max(memorySize, memoryRequirements.size);
                buffers.emplace_back(binding, std::move(ownedBuffer));
            }
        } else if (binding.descriptorType == vk::DescriptorType::eCombinedImageSampler ||
                   binding.descriptorType == vk::DescriptorType::eStorageImage) {
            if (binding.resourceCategory != vgflib::ResourceCategory::INTERMEDIATE) {
                if (aliasedImage != nullptr) {
                    addBoundImage(aliasedImage->image, binding, aliasedImage->memory, aliasedImage->currentLayout);
                    continue;
                }
                throw std::runtime_error("No manually bound image for aliased VGF resource " +
                                         std::to_string(binding.resourceIndex));
            }
            requireAliasedMemory(binding);
            auto ownedImage = createIntermediateImage(binding, true);
            if (aliasedMemory.has_value()) {
                ownedImage.bindMemory(aliasedMemory->memory, aliasedMemory->offset);
                ownedImages.push_back(std::move(ownedImage));
                addBoundImage(*ownedImages.back(), binding, *aliasedMemory);
            } else {
                const auto memoryRequirements = ownedImage.getMemoryRequirements();
                memoryTypeBits &= memoryRequirements.memoryTypeBits;
                memorySize = std::max(memorySize, memoryRequirements.size);
                images.emplace_back(binding, std::move(ownedImage));
            }
        } else {
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(binding.descriptorType)));
        }
    }

    if (tensors.empty() && buffers.empty() && images.empty()) {
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

    for (auto &[binding, ownedImage] : images) {
        ownedImage.bindMemory(memory, 0);
        ownedImages.push_back(std::move(ownedImage));
        addBoundImage(*ownedImages.back(), binding, {*ownedMemory.back(), 0, memorySize});
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
            case vk::DescriptorType::eCombinedImageSampler:
            case vk::DescriptorType::eStorageImage:
                allocateIntermediateImage(binding);
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
    const auto module = vgf.getSPIRVModule(segment.moduleIndex);
    for (const auto &binding : state.bindings) {
        if (binding.descriptorType != vk::DescriptorType::eStorageBuffer &&
            binding.descriptorType != vk::DescriptorType::eTensorARM &&
            binding.descriptorType != vk::DescriptorType::eCombinedImageSampler &&
            binding.descriptorType != vk::DescriptorType::eStorageImage) {
            throw std::runtime_error("Session does not support descriptor type " +
                                     std::to_string(static_cast<uint32_t>(binding.descriptorType)));
        }
        if (binding.descriptorType == vk::DescriptorType::eCombinedImageSampler ||
            binding.descriptorType == vk::DescriptorType::eStorageImage) {
            validateImageFormat(vgf.getResource(binding.resourceIndex).format);
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
        std::vector<vk::DataGraphPipelineResourceInfoImageLayoutARM> imageLayouts;
        std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
        tensorDescriptions.reserve(state.bindings.size());
        imageLayouts.reserve(state.bindings.size());
        resourceInfos.reserve(state.bindings.size());
        for (const auto &binding : state.bindings) {
            const auto resource = vgf.getResource(binding.resourceIndex);
            const auto tensorTiling = binding.descriptorType == vk::DescriptorType::eCombinedImageSampler ||
                                              binding.descriptorType == vk::DescriptorType::eStorageImage
                                          ? vk::TensorTilingARM::eOptimal
                                          : vk::TensorTilingARM::eLinear;
            tensorDescriptions.emplace_back(
                tensorTiling, resource.format, static_cast<uint32_t>(resource.shape.size()), resource.shape.data(),
                resource.stride.empty() ? nullptr : resource.stride.data(), vk::TensorUsageFlagBitsARM::eDataGraph);
            if (binding.descriptorType == vk::DescriptorType::eCombinedImageSampler ||
                binding.descriptorType == vk::DescriptorType::eStorageImage) {
                imageLayouts.emplace_back(imageLayout(binding.descriptorType), &tensorDescriptions.back());
                resourceInfos.emplace_back(binding.set, binding.binding, 0, &imageLayouts.back());
            } else {
                resourceInfos.emplace_back(binding.set, binding.binding, 0, &tensorDescriptions.back());
            }
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
    insertInitialImageLayoutTransitions(commandBuffer);
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

void Session::bindImage(const vk::raii::Image &image, DescriptorBindingInfo binding, BoundMemoryInfo memory) {
    impl_->bindImage(image, binding, memory, imageLayout(binding.descriptorType));
}

void Session::bindImage(const vk::raii::Image &image, DescriptorBindingInfo binding, vk::ImageLayout currentLayout,
                        BoundMemoryInfo memory) {
    impl_->bindImage(image, binding, memory, currentLayout);
}

void Session::configure() { impl_->configure(); }

void Session::run() { impl_->run(); }

} // namespace mlsdk::vgf_runtime
