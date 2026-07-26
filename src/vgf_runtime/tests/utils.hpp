/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../src/utils.hpp"

#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vgf_runtime::test {

struct MaxpoolSpvasmBindings {
    uint32_t inputSet;
    uint32_t inputBinding;
    uint32_t outputSet;
    uint32_t outputBinding;
};

inline void replaceAll(std::string &text, std::string_view from, std::string_view to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

inline std::vector<uint32_t> assembleSpirv(std::string_view text) {
    spvtools::SpirvTools tools{SPV_ENV_UNIVERSAL_1_6};
    if (!tools.IsValid()) {
        throw std::runtime_error("Failed to instantiate SPIR-V tools");
    }

    std::string diagnostics;
    tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &position,
                                 const char *message) {
        diagnostics += std::to_string(position.line) + ":" + std::to_string(position.column) + ": " + message + "\n";
    });

    std::vector<uint32_t> spirvModule;
    if (!tools.Assemble(std::string(text), &spirvModule)) {
        throw std::runtime_error("Failed to assemble SPIR-V program\n" + diagnostics);
    }

    if (!tools.Validate(spirvModule)) {
        throw std::runtime_error("Failed to validate SPIR-V program\n" + diagnostics);
    }

    return spirvModule;
}

inline std::vector<uint32_t> assembleMaxpoolSpirvFromTemplate(std::string_view name, const char *templatePath,
                                                              MaxpoolSpvasmBindings bindings) {
    std::ifstream templateFile(templatePath);
    std::string spvasm((std::istreambuf_iterator<char>(templateFile)), {});
    replaceAll(spvasm, "INPUT_SET", std::to_string(bindings.inputSet));
    replaceAll(spvasm, "INPUT_BINDING", std::to_string(bindings.inputBinding));
    replaceAll(spvasm, "OUTPUT_SET", std::to_string(bindings.outputSet));
    replaceAll(spvasm, "OUTPUT_BINDING", std::to_string(bindings.outputBinding));

    try {
        return assembleSpirv(spvasm);
    } catch (const std::runtime_error &error) {
        throw std::runtime_error("Failed to assemble SPIR-V test asset " + std::string(name) + ": " + error.what());
    }
}

inline std::vector<uint32_t> assembleMaxpool16x16To8x8Spirv(std::string_view name, MaxpoolSpvasmBindings bindings) {
    return assembleMaxpoolSpirvFromTemplate(name, VGF_RUNTIME_MAXPOOL_16X16_TO_8X8_SPVASM, bindings);
}

inline std::vector<uint32_t> assembleMaxpool8x8To4x4Spirv(std::string_view name, MaxpoolSpvasmBindings bindings) {
    return assembleMaxpoolSpirvFromTemplate(name, VGF_RUNTIME_MAXPOOL_8X8_TO_4X4_SPVASM, bindings);
}

inline std::vector<uint32_t> assembleAddInt32BuffersSpirv() {
    std::ifstream templateFile(VGF_RUNTIME_ADD_INT32_BUFFERS_SPVASM);
    std::string spvasm((std::istreambuf_iterator<char>(templateFile)), {});
    return assembleSpirv(spvasm);
}

inline bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, const char *name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto &extension) {
        return std::string_view(extension.extensionName.data()) == name;
    });
}

inline uint32_t findDataGraphQueueFamily(const vk::raii::PhysicalDevice &physicalDevice) {
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); ++i) {
        const auto requiredFlags = vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eCompute;
        if ((queueFamilies[i].queueFlags & requiredFlags) == requiredFlags) {
            return i;
        }
    }
    return UINT32_MAX;
}

class RuntimeSessionExecutionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const vk::ApplicationInfo applicationInfo("vgf-runtime-test", 1, nullptr, 0, VK_API_VERSION_1_3);
        instance = vk::raii::Instance(context, vk::InstanceCreateInfo({}, &applicationInfo));

        for (auto &candidate : vk::raii::PhysicalDevices(instance)) {
            const auto extensions = candidate.enumerateDeviceExtensionProperties();
            if (!hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME)) {
                continue;
            }
            const auto candidateQueueFamilyIndex = findDataGraphQueueFamily(candidate);
            if (candidateQueueFamilyIndex != UINT32_MAX) {
                physicalDevice = candidate;
                queueFamilyIndex = candidateQueueFamilyIndex;
                break;
            }
        }
        if (queueFamilyIndex == UINT32_MAX) {
            GTEST_SKIP() << "No Vulkan device with VK_ARM_data_graph, VK_ARM_tensors, and compute queue support";
        }

        const float queuePriority = 1.0F;
        const vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.shaderInt16 = true;
        deviceFeatures.shaderInt64 = true;

        vulkan12Features.storageBuffer8BitAccess = true;
        vulkan12Features.shaderInt8 = true;
        vulkan12Features.vulkanMemoryModel = true;

        vulkan13Features.synchronization2 = true;
        vulkan13Features.maintenance4 = true;
        vulkan13Features.pipelineCreationCacheControl = true;
        vulkan13Features.pNext = &vulkan12Features;

        tensorFeatures.tensors = true;
        tensorFeatures.shaderTensorAccess = true;
        tensorFeatures.tensorNonPacked = true;
        tensorFeatures.pNext = &vulkan13Features;

        dataGraphFeatures.dataGraph = true;
        dataGraphFeatures.dataGraphShaderModule = true;
        dataGraphFeatures.pNext = &tensorFeatures;

        void *featureChain = &dataGraphFeatures;
        const auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
        std::vector<const char *> deviceExtensions = {VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_TENSORS_EXTENSION_NAME};
        if (hasExtension(extensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME)) {
            replicatedCompositesFeatures.shaderReplicatedComposites = true;
            replicatedCompositesFeatures.pNext = featureChain;
            featureChain = &replicatedCompositesFeatures;
            deviceExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        }
        device = vk::raii::Device(
            physicalDevice,
            {vk::DeviceCreateFlags(), queueCreateInfo, {}, deviceExtensions, &deviceFeatures, featureChain});
        queue = device.getQueue(queueFamilyIndex, 0);
    }

    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    vk::raii::Queue queue{nullptr};
    uint32_t queueFamilyIndex = UINT32_MAX;

    vk::PhysicalDeviceVulkan12Features vulkan12Features;
    vk::PhysicalDeviceVulkan13Features vulkan13Features;
    vk::PhysicalDeviceTensorFeaturesARM tensorFeatures;
    vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeatures;
    vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures;
};

struct Tensor {
    Tensor(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, vk::Format format,
           const std::vector<int64_t> &shape)
        : shape(shape) {
        const vk::TensorDescriptionARM description(vk::TensorTilingARM::eLinear, format,
                                                   static_cast<uint32_t>(this->shape.size()), this->shape.data(),
                                                   nullptr, vk::TensorUsageFlagBitsARM::eDataGraph);
        const vk::TensorCreateInfoARM createInfo({}, &description, vk::SharingMode::eExclusive);
        tensor = vk::raii::TensorARM(device, createInfo);

        const auto memoryRequirements =
            device.getTensorMemoryRequirementsARM(vk::TensorMemoryRequirementsInfoARM(*tensor));
        memorySize = memoryRequirements.memoryRequirements.size;
        const auto memoryType = detail::vulkan_helpers::findMemoryType(
            physicalDevice, memoryRequirements.memoryRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        memory = vk::raii::DeviceMemory(device, {memorySize, memoryType});
        device.bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*tensor, *memory, 0));
    }

    size_t numElements() const { return Tensor::numElements(shape); }

    static size_t numElements(const std::vector<int64_t> &shape) {
        return static_cast<size_t>(detail::utils::elementCount(shape));
    }

    void fill(int8_t value, size_t elements) const {
        void *data = memory.mapMemory(0, memorySize);
        std::memset(data, 0, static_cast<size_t>(memorySize));
        std::fill_n(static_cast<int8_t *>(data), elements, value);
        memory.unmapMemory();
    }

    void write(const std::vector<int8_t> &values) const {
        void *data = memory.mapMemory(0, memorySize);
        std::memset(data, 0, static_cast<size_t>(memorySize));
        std::copy(values.begin(), values.end(), static_cast<int8_t *>(data));
        memory.unmapMemory();
    }

    std::vector<int8_t> read(size_t elements) const {
        const void *data = memory.mapMemory(0, memorySize);
        const auto *begin = static_cast<const int8_t *>(data);
        std::vector<int8_t> result(begin, begin + elements);
        memory.unmapMemory();
        return result;
    }

    std::vector<int64_t> shape;
    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::TensorARM tensor{nullptr};
    vk::DeviceSize memorySize = 0;
};

struct Buffer {
    Buffer(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, vk::DeviceSize size)
        : buffer(device, vk::BufferCreateInfo({}, size, vk::BufferUsageFlagBits::eStorageBuffer)) {
        const auto memoryRequirements = buffer.getMemoryRequirements();
        memorySize = memoryRequirements.size;
        const auto memoryType = detail::vulkan_helpers::findMemoryType(
            physicalDevice, memoryRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        memory = vk::raii::DeviceMemory(device, {memorySize, memoryType});
        buffer.bindMemory(*memory, 0);
    }

    void write(const std::vector<int32_t> &values) const {
        void *data = memory.mapMemory(0, memorySize);
        std::memset(data, 0, static_cast<size_t>(memorySize));
        std::copy(values.begin(), values.end(), static_cast<int32_t *>(data));
        memory.unmapMemory();
    }

    std::vector<int32_t> read(size_t elements) const {
        const void *data = memory.mapMemory(0, memorySize);
        const auto *begin = static_cast<const int32_t *>(data);
        std::vector<int32_t> result(begin, begin + elements);
        memory.unmapMemory();
        return result;
    }

    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::Buffer buffer{nullptr};
    vk::DeviceSize memorySize = 0;
};

inline std::vector<int32_t> addVectors(const std::vector<int32_t> &lhs, const std::vector<int32_t> &rhs) {
    std::vector<int32_t> result(lhs.size());
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::plus<>());
    return result;
}

inline std::vector<int32_t> int32WordsFromBytes(const std::vector<int8_t> &bytes, size_t words) {
    std::vector<int32_t> result(words);
    for (size_t word = 0; word < words; ++word) {
        uint32_t value = 0;
        for (size_t byte = 0; byte < sizeof(int32_t); ++byte) {
            value |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[word * sizeof(int32_t) + byte])) << (byte * 8);
        }
        result[word] = static_cast<int32_t>(value);
    }
    return result;
}

inline std::vector<int8_t> makeMaxpoolInput(const std::vector<int64_t> &shape, uint32_t seed = 0) {
    const auto batch = shape[0];
    const auto height = shape[1];
    const auto width = shape[2];
    const auto channels = shape[3];

    std::vector<int8_t> input(Tensor::numElements(shape));
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                for (int64_t c = 0; c < channels; ++c) {
                    const auto index = static_cast<size_t>(((n * height + h) * width + w) * channels + c);
                    input[index] = static_cast<int8_t>((n * 17 + h * 13 + w * 7 + c * 3 + seed) % 97);
                }
            }
        }
    }
    return input;
}

inline std::vector<int8_t> expectedMaxpool(const std::vector<int8_t> &input, const std::vector<int64_t> &shape) {
    const auto batch = shape[0];
    const auto inputHeight = shape[1];
    const auto inputWidth = shape[2];
    const auto channels = shape[3];
    const auto outputHeight = inputHeight / 2;
    const auto outputWidth = inputWidth / 2;

    std::vector<int8_t> expected(Tensor::numElements({batch, outputHeight, outputWidth, channels}));
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t h = 0; h < outputHeight; ++h) {
            for (int64_t w = 0; w < outputWidth; ++w) {
                for (int64_t c = 0; c < channels; ++c) {
                    const auto firstInputIndex =
                        static_cast<size_t>(((n * inputHeight + h * 2) * inputWidth + w * 2) * channels + c);
                    int8_t maxValue = input[firstInputIndex];
                    for (int64_t kh = 0; kh < 2; ++kh) {
                        for (int64_t kw = 0; kw < 2; ++kw) {
                            const auto inputIndex = static_cast<size_t>(
                                ((n * inputHeight + h * 2 + kh) * inputWidth + w * 2 + kw) * channels + c);
                            maxValue = std::max(maxValue, input[inputIndex]);
                        }
                    }
                    expected[static_cast<size_t>(((n * outputHeight + h) * outputWidth + w) * channels + c)] = maxValue;
                }
            }
        }
    }
    return expected;
}

} // namespace vgf_runtime::test
