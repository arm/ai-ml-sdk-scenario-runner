/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"
#include "glsl_compiler.hpp"
#include "logging.hpp"

#include "vgf/vulkan_helpers.generated.hpp"

#include "vulkan/vulkan_format_traits.hpp"

#include <fstream>
#include <limits>
#include <numeric>

namespace mlsdk::scenariorunner {

uint32_t numComponentsFromVkFormat(vk::Format format) { return vk::componentCount(format); }

namespace {
bool isPow2(uint32_t value) { return ((value & (~(value - 1))) == value); }

} // namespace

uint32_t elementSizeFromVkFormat(vk::Format format) {

    uint32_t value = vk::blockSize(format);

    if (isPow2(value)) {
        return value;
    }

    // Round up to next power of 2
    uint32_t countBits = 1; // start with offset to select next power of 2
    while (value >>= 1) {
        countBits++;
    }
    return 1 << countBits;
}

vk::Format getVkFormatFromString(const std::string &format) {
    if (format == "VK_FORMAT_R8_BOOL_ARM") {
        return vk::Format::eR8BoolARM;
    } else if (format == "VK_FORMAT_R8_UINT") {
        return vk::Format::eR8Uint;
    } else if (format == "VK_FORMAT_R8_SINT") {
        return vk::Format::eR8Sint;
    } else if (format == "VK_FORMAT_R8_SNORM") {
        return vk::Format::eR8Snorm;
    } else if (format == "VK_FORMAT_R16_UINT") {
        return vk::Format::eR16Uint;
    } else if (format == "VK_FORMAT_R16_SINT") {
        return vk::Format::eR16Sint;
    } else if (format == "VK_FORMAT_R8G8_SINT") {
        return vk::Format::eR8G8Sint;
    } else if (format == "VK_FORMAT_R8G8_UNORM") {
        return vk::Format::eR8G8Unorm;
    } else if (format == "VK_FORMAT_R8G8B8_SINT") {
        return vk::Format::eR8G8B8Sint;
    } else if (format == "VK_FORMAT_R32_SINT") {
        return vk::Format::eR32Sint;
    } else if (format == "VK_FORMAT_R16_SFLOAT") {
        return vk::Format::eR16Sfloat;
    } else if (format == "VK_FORMAT_R32_SFLOAT") {
        return vk::Format::eR32Sfloat;
    } else if (format == "VK_FORMAT_B8G8R8A8_UNORM") {
        return vk::Format::eB8G8R8A8Unorm;
    } else if (format == "VK_FORMAT_R8G8B8A8_UNORM") {
        return vk::Format::eR8G8B8A8Unorm;
    } else if (format == "VK_FORMAT_R64_SINT") {
        return vk::Format::eR64Sint;
    } else if (format == "VK_FORMAT_R8G8B8A8_SNORM") {
        return vk::Format::eR8G8B8A8Snorm;
    } else if (format == "VK_FORMAT_R8G8B8_SNORM") {
        return vk::Format::eR8G8B8Snorm;
    } else if (format == "VK_FORMAT_R8G8B8A8_SINT") {
        return vk::Format::eR8G8B8A8Sint;
    } else if (format == "VK_FORMAT_R16G16B16A16_UNORM") {
        return vk::Format::eR16G16B16A16Unorm;
    } else if (format == "VK_FORMAT_R16G16B16A16_SNORM") {
        return vk::Format::eR16G16B16A16Snorm;
    } else if (format == "VK_FORMAT_R16G16B16A16_SFLOAT") {
        return vk::Format::eR16G16B16A16Sfloat;
    } else if (format == "VK_FORMAT_R16G16B16A16_SINT") {
        return vk::Format::eR16G16B16A16Sint;
    } else if (format == "VK_FORMAT_R32G32B32A32_SFLOAT") {
        return vk::Format::eR32G32B32A32Sfloat;
    } else if (format == "VK_FORMAT_R16G16_SFLOAT") {
        return vk::Format::eR16G16Sfloat;
    } else if (format == "VK_FORMAT_B10G11R11_UFLOAT_PACK32") {
        return vk::Format::eB10G11R11UfloatPack32;
    } else if (format == "VK_FORMAT_D32_SFLOAT_S8_UINT") {
        return vk::Format::eD32SfloatS8Uint;
    } else if (format == "VK_FORMAT_R8_UNORM") {
        return vk::Format::eR8Unorm;
    } else if (format == "VK_FORMAT_R32_UINT") {
        return vk::Format::eR32Uint;
    } else {
        throw std::runtime_error("Unknown VkFormat: " + format);
    }
}

vk::ImageAspectFlags getImageAspectMaskForVkFormat(vk::Format format) {
    if (format == vk::Format::eD32Sfloat) {
        return vk::ImageAspectFlagBits::eDepth;
    }

    return vk::ImageAspectFlagBits::eColor;
}

const vgfutils::numpy::DType getDTypeFromVkFormat(vk::Format format) {
    if (numComponentsFromVkFormat(format) != 1) {
        throw std::runtime_error("More than 1 components from VkFormat: " +
                                 vgflib::FormatTypeToName(vgflib::ToFormatType(format)));
    }

    char const *numeric = componentNumericFormat(format, 0);
    char encoding = vgfutils::numpy::numpyTypeEncoding(numeric);
    uint32_t size = elementSizeFromVkFormat(format);

    if (encoding == '?') {
        throw std::runtime_error("Unsupported VkFormat: " + vgflib::FormatTypeToName(vgflib::ToFormatType(format)));
    }
    return vgfutils::numpy::DType(encoding, size);
}

uint64_t totalElementsFromShape(const std::vector<int64_t> &shape) {
    return static_cast<uint64_t>(
        std::abs(std::accumulate(shape.cbegin(), shape.cend(), int64_t(1), std::multiplies<int64_t>())));
}

uint32_t findMemoryIdx(const Context &ctx, uint32_t memTypeBits, vk::MemoryPropertyFlags required) {
    const vk::PhysicalDeviceMemoryProperties memProps = ctx.physicalDevice().getMemoryProperties();

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((1U << i) & memTypeBits) {
            const vk::MemoryType &memType = memProps.memoryTypes[i];
            if ((memType.propertyFlags & required) == required) {
                return i;
            }
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

std::vector<uint32_t> readShaderCode(const ShaderInfo &shaderInfo) {
    switch (shaderInfo.shaderType) {
    case ShaderType::SPIR_V: {
        std::ifstream shaderFile{shaderInfo.src, std::ios::binary | std::ios::ate};
        if (!shaderFile.is_open()) {
            throw std::runtime_error("Cannot open SPIR-V shader file.");
        }
        shaderFile.exceptions(std::ios::badbit);

        std::streamsize codeSize = shaderFile.tellg();
        if (codeSize < 0) {
            throw std::runtime_error("Failed to get shader file size.");
        }
        shaderFile.seekg(0);
        std::vector<uint32_t> code(static_cast<size_t>(codeSize) / sizeof(uint32_t), 0);
        shaderFile.read(reinterpret_cast<char *>(code.data()), codeSize);
        return code;
    }
    case ShaderType::GLSL: {
        std::ifstream shaderFile{shaderInfo.src};
        if (!shaderFile.is_open()) {
            throw std::runtime_error("Cannot open GLSL shader file.");
        }
        shaderFile.exceptions(std::ios::badbit);
        std::string content((std::istreambuf_iterator<char>(shaderFile)), (std::istreambuf_iterator<char>()));
        auto spirv = GlslCompiler::get().compile(content, shaderInfo.buildOpts, shaderInfo.includeDirs);
        if (!spirv.first.empty()) {
            throw std::runtime_error("Compilation error\n" + spirv.first);
        }
        return spirv.second;
    }
    default:
        throw std::runtime_error("Unknown shader type");
    }
}

void SPIRVMessageConsumer(spv_message_level_t level, const char *, const spv_position_t &position,
                          const char *message) {
    mlsdk::logging::LogLevel logLevel;
    switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
        logLevel = mlsdk::logging::LogLevel::Error;
        break;
    case SPV_MSG_WARNING:
        logLevel = mlsdk::logging::LogLevel::Warning;
        break;
    default:
        logLevel = mlsdk::logging::LogLevel::Info;
        break;
    }

    mlsdk::logging::log("SPVTools", logLevel, "line:" + std::to_string(position.index) + ": " + message);
}

} // namespace mlsdk::scenariorunner
