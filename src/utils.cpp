/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"
#include "glsl_compiler.hpp"
#ifdef SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT
#    include "hlsl_compiler.hpp"
#endif
#include "logging.hpp"

#include "vgf/vulkan_helpers.generated.hpp"

#include "vulkan/vulkan_format_traits.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
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
    }
    if (format == "VK_FORMAT_R8_UINT") {
        return vk::Format::eR8Uint;
    }
    if (format == "VK_FORMAT_R8_SINT") {
        return vk::Format::eR8Sint;
    }
    if (format == "VK_FORMAT_R8_SNORM") {
        return vk::Format::eR8Snorm;
    }
    if (format == "VK_FORMAT_R16_UINT") {
        return vk::Format::eR16Uint;
    }
    if (format == "VK_FORMAT_R16_SINT") {
        return vk::Format::eR16Sint;
    }
    if (format == "VK_FORMAT_R16_UNORM") {
        return vk::Format::eR16Unorm;
    }
    if (format == "VK_FORMAT_R16_SNORM") {
        return vk::Format::eR16Snorm;
    }
    if (format == "VK_FORMAT_R8G8_SINT") {
        return vk::Format::eR8G8Sint;
    }
    if (format == "VK_FORMAT_R8G8_UINT") {
        return vk::Format::eR8G8Uint;
    }
    if (format == "VK_FORMAT_R8G8_UNORM") {
        return vk::Format::eR8G8Unorm;
    }
    if (format == "VK_FORMAT_R8G8_SNORM") {
        return vk::Format::eR8G8Snorm;
    }
    if (format == "VK_FORMAT_R8G8B8_SINT") {
        return vk::Format::eR8G8B8Sint;
    }
    if (format == "VK_FORMAT_R8G8B8_UINT") {
        return vk::Format::eR8G8B8Uint;
    }
    if (format == "VK_FORMAT_R8G8B8_UNORM") {
        return vk::Format::eR8G8B8Unorm;
    }
    if (format == "VK_FORMAT_R8G8B8_SRGB") {
        return vk::Format::eR8G8B8Srgb;
    }
    if (format == "VK_FORMAT_R32_SINT") {
        return vk::Format::eR32Sint;
    }
    if (format == "VK_FORMAT_R16_SFLOAT") {
        return vk::Format::eR16Sfloat;
    }
    if (format == "VK_FORMAT_R32_SFLOAT") {
        return vk::Format::eR32Sfloat;
    }
    if (format == "VK_FORMAT_R32G32_UINT") {
        return vk::Format::eR32G32Uint;
    }
    if (format == "VK_FORMAT_R32G32_SINT") {
        return vk::Format::eR32G32Sint;
    }
    if (format == "VK_FORMAT_R32G32_SFLOAT") {
        return vk::Format::eR32G32Sfloat;
    }
    if (format == "VK_FORMAT_B8G8R8A8_UNORM") {
        return vk::Format::eB8G8R8A8Unorm;
    }
    if (format == "VK_FORMAT_B8G8R8A8_SRGB") {
        return vk::Format::eB8G8R8A8Srgb;
    }
    if (format == "VK_FORMAT_R8G8B8A8_UNORM") {
        return vk::Format::eR8G8B8A8Unorm;
    }
    if (format == "VK_FORMAT_R8G8B8A8_UINT") {
        return vk::Format::eR8G8B8A8Uint;
    }
    if (format == "VK_FORMAT_R8G8B8A8_SRGB") {
        return vk::Format::eR8G8B8A8Srgb;
    }
    if (format == "VK_FORMAT_R64_SINT") {
        return vk::Format::eR64Sint;
    }
    if (format == "VK_FORMAT_R8G8B8A8_SNORM") {
        return vk::Format::eR8G8B8A8Snorm;
    }
    if (format == "VK_FORMAT_R8G8B8_SNORM") {
        return vk::Format::eR8G8B8Snorm;
    }
    if (format == "VK_FORMAT_B8G8R8_UNORM") {
        return vk::Format::eB8G8R8Unorm;
    }
    if (format == "VK_FORMAT_B8G8R8_SRGB") {
        return vk::Format::eB8G8R8Srgb;
    }
    if (format == "VK_FORMAT_R8G8B8A8_SINT") {
        return vk::Format::eR8G8B8A8Sint;
    }
    if (format == "VK_FORMAT_R16G16B16A16_UNORM") {
        return vk::Format::eR16G16B16A16Unorm;
    }
    if (format == "VK_FORMAT_R16G16B16A16_UINT") {
        return vk::Format::eR16G16B16A16Uint;
    }
    if (format == "VK_FORMAT_R16G16B16A16_SNORM") {
        return vk::Format::eR16G16B16A16Snorm;
    }
    if (format == "VK_FORMAT_R16G16B16A16_SFLOAT") {
        return vk::Format::eR16G16B16A16Sfloat;
    }
    if (format == "VK_FORMAT_R16G16B16A16_SINT") {
        return vk::Format::eR16G16B16A16Sint;
    }
    if (format == "VK_FORMAT_R32G32B32A32_UINT") {
        return vk::Format::eR32G32B32A32Uint;
    }
    if (format == "VK_FORMAT_R32G32B32A32_SINT") {
        return vk::Format::eR32G32B32A32Sint;
    }
    if (format == "VK_FORMAT_R32G32B32A32_SFLOAT") {
        return vk::Format::eR32G32B32A32Sfloat;
    }
    if (format == "VK_FORMAT_R16G16_UINT") {
        return vk::Format::eR16G16Uint;
    }
    if (format == "VK_FORMAT_R16G16_SINT") {
        return vk::Format::eR16G16Sint;
    }
    if (format == "VK_FORMAT_R16G16_UNORM") {
        return vk::Format::eR16G16Unorm;
    }
    if (format == "VK_FORMAT_R16G16_SNORM") {
        return vk::Format::eR16G16Snorm;
    }
    if (format == "VK_FORMAT_R16G16_SFLOAT") {
        return vk::Format::eR16G16Sfloat;
    }
    if (format == "VK_FORMAT_B10G11R11_UFLOAT_PACK32") {
        return vk::Format::eB10G11R11UfloatPack32;
    }
    if (format == "VK_FORMAT_D32_SFLOAT") {
        return vk::Format::eD32Sfloat;
    }
    if (format == "VK_FORMAT_D24_UNORM_S8_UINT") {
        return vk::Format::eD24UnormS8Uint;
    }
    if (format == "VK_FORMAT_D32_SFLOAT_S8_UINT") {
        return vk::Format::eD32SfloatS8Uint;
    }
    if (format == "VK_FORMAT_R8_UNORM") {
        return vk::Format::eR8Unorm;
    }
    if (format == "VK_FORMAT_R32_UINT") {
        return vk::Format::eR32Uint;
    }
    if (format == "VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM") {
        return vk::Format::eR16SfloatFpencodingBfloat16ARM;
    }
    if (format == "VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM") {
        return vk::Format::eR8SfloatFpencodingFloat8E4M3ARM;
    }
    if (format == "VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM") {
        return vk::Format::eR8SfloatFpencodingFloat8E5M2ARM;
    }
    throw std::runtime_error("Unknown VkFormat: " + format);
}

vk::ImageAspectFlags getImageAspectMaskForVkFormat(vk::Format format) {
    if (format == vk::Format::eD32Sfloat) {
        return vk::ImageAspectFlagBits::eDepth;
    }

    return vk::ImageAspectFlagBits::eColor;
}

vgfutils::numpy::DType getDTypeFromVkFormat(vk::Format format) {
    if (numComponentsFromVkFormat(format) != 1) {
        throw std::runtime_error("More than 1 components from VkFormat: " +
                                 vgflib::FormatTypeToName(vgflib::ToFormatType(format)));
    }
    // Handle special case for BFLOAT16 and FLOAT8 to use raw byte encoding 'V' in numpy output
    if (format == vk::Format::eR16SfloatFpencodingBfloat16ARM) {
        return vgfutils::numpy::DType('V', 2);
    }
    if (format == vk::Format::eR8SfloatFpencodingFloat8E4M3ARM) {
        return vgfutils::numpy::DType('V', 1, '<');
    }
    if (format == vk::Format::eR8SfloatFpencodingFloat8E5M2ARM) {
        return vgfutils::numpy::DType('u', 1);
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

std::string lowercaseExtension(const std::string &path) {
    auto ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
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
        auto spirv =
            GlslCompiler::get().compile(content, shaderInfo.stage, shaderInfo.buildOpts, shaderInfo.includeDirs);
        if (!spirv.first.empty()) {
            throw std::runtime_error("Compilation error\n" + spirv.first);
        }
        return spirv.second;
    }
    case ShaderType::HLSL: {
#ifdef SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT
        std::ifstream shaderFile{shaderInfo.src};
        if (!shaderFile.is_open()) {
            throw std::runtime_error("Cannot open HLSL shader file.");
        }
        shaderFile.exceptions(std::ios::badbit);
        std::string content((std::istreambuf_iterator<char>(shaderFile)), (std::istreambuf_iterator<char>()));
        auto spirv = HlslCompiler::get().compile(content, shaderInfo.entry, shaderInfo.debugName, shaderInfo.buildOpts,
                                                 shaderInfo.includeDirs);
        if (!spirv.first.empty()) {
            throw std::runtime_error("Compilation error\n" + spirv.first);
        }
        return spirv.second;
#else
        throw std::runtime_error("HLSL shaders are not supported on this platform.");
#endif
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
