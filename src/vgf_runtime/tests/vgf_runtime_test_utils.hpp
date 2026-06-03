/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "vgf/encoder.hpp"

#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlsdk::vgf_runtime::test {

struct MaxpoolSpvasmBindings {
    uint32_t inputSet;
    uint32_t inputBinding;
    uint32_t outputSet;
    uint32_t outputBinding;
};

inline std::string writeVgf(const std::function<void(mlsdk::vgflib::Encoder &)> &populate) {
    auto encoder = mlsdk::vgflib::CreateEncoder(VK_HEADER_VERSION);
    populate(*encoder);
    encoder->Finish();

    std::stringstream stream;
    EXPECT_TRUE(encoder->WriteTo(stream));
    return stream.str();
}

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

    std::vector<uint32_t> spirvModule;
    if (!tools.Assemble(std::string(text), &spirvModule)) {
        throw std::runtime_error("Failed to assemble SPIR-V program");
    }

    if (!tools.Validate(spirvModule)) {
        throw std::runtime_error("Failed to validate SPIR-V program");
    }

    return spirvModule;
}

inline std::vector<uint32_t> assembleMaxpoolSpirv(std::string_view name, MaxpoolSpvasmBindings bindings) {
    std::ifstream templateFile(VGF_RUNTIME_MAXPOOL_16X16_TO_8X8_SPVASM);
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

inline std::string makeMaxpoolVgf() {
    const auto &code = assembleMaxpoolSpirv("maxpool_set0", {0, 0, 1, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool", "main", code);
        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({inputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "maxpool_graph_segment", {inputSet, outputSet}, {inputBinding}, {outputBinding},
                               {});
    });
}

inline bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, const char *name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto &extension) {
        return std::string_view(extension.extensionName.data()) == name;
    });
}

inline uint32_t findDataGraphQueueFamily(const vk::raii::PhysicalDevice &physicalDevice) {
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); ++i) {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eDataGraphARM) {
            return i;
        }
    }
    return UINT32_MAX;
}

inline uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t memoryTypeBits,
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
        const auto memoryType =
            findMemoryType(physicalDevice, memoryRequirements.memoryRequirements.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        memory = vk::raii::DeviceMemory(device, {memorySize, memoryType});
        device.bindTensorMemoryARM(vk::BindTensorMemoryInfoARM(*tensor, *memory, 0));
    }

    size_t numElements() const { return Tensor::numElements(shape); }

    static size_t numElements(const std::vector<int64_t> &shape) {
        size_t count = 1;
        for (const auto dim : shape) {
            count *= static_cast<size_t>(dim);
        }
        return count;
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

} // namespace mlsdk::vgf_runtime::test
