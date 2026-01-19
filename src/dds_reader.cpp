/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dds_reader.hpp"
#include "utils.hpp"

#include "vgf/vulkan_helpers.generated.hpp"

#include <fstream>

namespace mlsdk::scenariorunner {

namespace {

constexpr uint32_t MAGIC_WORD = 0x20534444;      // "DDS "
constexpr uint32_t DX10_MAGIC_WORD = 0x30315844; // "DX10"
constexpr uint32_t REQUIRED_FLAGS =
    0x1 | 0x2 | 0x4 | 0x1000; // Caps, Height, Width and PixelFormat flags are always required in DDS header
constexpr uint32_t REQUIRED_CAPS = 0x1000;         // Required by DDS spec
constexpr uint32_t DX10_2D_IMAGE_RESOURCE_DIM = 3; // Represents a 2d image
constexpr uint32_t DX10_CUBE_MAP_FLAG = 0x4;

void validateDDSHeader(const DDSHeaderInfo &header) {
    if (header.magicWord != MAGIC_WORD) {
        throw std::runtime_error("Invalid DDS magic word");
    }
    if (header.header.size != 124) {
        throw std::runtime_error("Invalid DDS header size (Must be 124)");
    }
    if ((header.header.flags & REQUIRED_FLAGS) != REQUIRED_FLAGS) {
        throw std::runtime_error("Required DDS header height/width flags not set");
    }
    if (header.header.mipMapCount > 1) {
        throw std::runtime_error("Mipmaps are not supported");
    }
    if (header.header.pixelFormat.size != 32) {
        throw std::runtime_error("Invalid DDS pixel format header size (Must be 32)");
    }
    if (!(header.header.caps & REQUIRED_CAPS)) {
        throw std::runtime_error("Required DDS header caps flag not set");
    }
    if (!header.isDx10) {
        throw std::runtime_error("Non-DX10 DDS files not supported");
    }
    if (header.header10.resourceDimension != DX10_2D_IMAGE_RESOURCE_DIM) {
        throw std::runtime_error("Only 2D DDS textures are supported");
    }
    if (header.header10.miscFlag == DX10_CUBE_MAP_FLAG) {
        throw std::runtime_error("Cube-map DDS textures are not supported");
    }
}

vk::Format ddsFormatToVkFormat(const DDSHeaderInfo &header) {
    if (!header.isDx10) {
        return vk::Format::eUndefined;
    }

    switch (header.header10.dxgiFormat) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return vk::Format::eR32G32B32A32Sfloat;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return vk::Format::eR16G16B16A16Sfloat;
    case DXGI_FORMAT_R16G16B16A16_SINT:
        return vk::Format::eR16G16B16A16Sint;
    case DXGI_FORMAT_R16G16_FLOAT:
        return vk::Format::eR16G16Sfloat;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return vk::Format::eB10G11R11UfloatPack32;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return vk::Format::eD32SfloatS8Uint;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return vk::Format::eR8G8B8A8Unorm;
    case DXGI_FORMAT_R8G8B8A8_SNORM:
        return vk::Format::eR8G8B8A8Snorm;
    case DXGI_FORMAT_R8G8B8_SNORM_CUSTOM:
        return vk::Format::eR8G8B8Snorm;
    case DXGI_FORMAT_R8G8B8A8_SINT:
        return vk::Format::eR8G8B8A8Sint;
    case DXGI_FORMAT_R8G8B8_SINT_CUSTOM:
        return vk::Format::eR8G8B8Sint;
    case DXGI_FORMAT_R8G8_UNORM:
        return vk::Format::eR8G8Unorm;
    case DXGI_FORMAT_R8G8_UINT:
        return vk::Format::eR8G8Uint;
    case DXGI_FORMAT_R8G8_SINT:
        return vk::Format::eR8G8Sint;
    case DXGI_FORMAT_R8_UNORM:
        return vk::Format::eR8Unorm;
    case DXGI_FORMAT_R8_SNORM:
        return vk::Format::eR8Snorm;
    case DXGI_FORMAT_R32_UINT:
        return vk::Format::eR32Uint;
    case DXGI_FORMAT_R32_FLOAT:
        return vk::Format::eR32Sfloat;
    case DXGI_FORMAT_R16_FLOAT:
        return vk::Format::eR16Sfloat;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return vk::Format::eB8G8R8A8Unorm;
#ifdef SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT
    // These are image formats that haven't been fully tested yet.
    case DXGI_FORMAT_R32G32B32A32_UINT:
        return vk::Format::eR32G32B32A32Uint;
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return vk::Format::eR32G32B32A32Sint;
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        return vk::Format::eR16G16B16A16Unorm;
    case DXGI_FORMAT_R16G16B16A16_UINT:
        return vk::Format::eR16G16B16A16Uint;
    case DXGI_FORMAT_R16G16B16A16_SNORM:
        return vk::Format::eR16G16B16A16Snorm;
    case DXGI_FORMAT_R16G16_UNORM:
        return vk::Format::eR16G16Unorm;
    case DXGI_FORMAT_R16G16_UINT:
        return vk::Format::eR16G16Uint;
    case DXGI_FORMAT_R16G16_SNORM:
        return vk::Format::eR16G16Snorm;
    case DXGI_FORMAT_R16G16_SINT:
        return vk::Format::eR16G16Sint;
    case DXGI_FORMAT_D32_FLOAT:
        return vk::Format::eD32Sfloat;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return vk::Format::eR8G8B8A8Srgb;
    case DXGI_FORMAT_R8G8B8A8_UINT:
        return vk::Format::eR8G8B8A8Uint;
    case DXGI_FORMAT_R8G8_SNORM:
        return vk::Format::eR8G8Snorm;
    case DXGI_FORMAT_R32_SINT:
        return vk::Format::eR32Sint;
    case DXGI_FORMAT_R16_UNORM:
        return vk::Format::eR16Unorm;
    case DXGI_FORMAT_R16_UINT:
        return vk::Format::eR16Uint;
    case DXGI_FORMAT_R16_SNORM:
        return vk::Format::eR16Snorm;
    case DXGI_FORMAT_R16_SINT:
        return vk::Format::eR16Sint;
    case DXGI_FORMAT_R8_UINT:
        return vk::Format::eR8Uint;
    case DXGI_FORMAT_R8_SINT:
        return vk::Format::eR8Sint;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return vk::Format::eD24UnormS8Uint;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return vk::Format::eB8G8R8A8Srgb;
    case DXGI_FORMAT_B8G8R8X8_UNORM:
        return vk::Format::eB8G8R8Unorm;
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return vk::Format::eB8G8R8Srgb;
    case DXGI_FORMAT_BC1_UNORM:
        return vk::Format::eBc1RgbaUnormBlock;
    case DXGI_FORMAT_BC1_UNORM_SRGB:
        return vk::Format::eBc1RgbaSrgbBlock;
    case DXGI_FORMAT_BC2_UNORM:
        return vk::Format::eBc2UnormBlock;
    case DXGI_FORMAT_BC2_UNORM_SRGB:
        return vk::Format::eBc2SrgbBlock;
    case DXGI_FORMAT_BC3_UNORM:
        return vk::Format::eBc3UnormBlock;
    case DXGI_FORMAT_BC3_UNORM_SRGB:
        return vk::Format::eBc3SrgbBlock;
    case DXGI_FORMAT_BC4_UNORM:
        return vk::Format::eBc4UnormBlock;
    case DXGI_FORMAT_BC4_SNORM:
        return vk::Format::eBc4SnormBlock;
    case DXGI_FORMAT_BC5_UNORM:
        return vk::Format::eBc5UnormBlock;
    case DXGI_FORMAT_BC5_SNORM:
        return vk::Format::eBc5SnormBlock;
    case DXGI_FORMAT_BC6H_UF16:
        return vk::Format::eBc6HUfloatBlock;
    case DXGI_FORMAT_BC6H_SF16:
        return vk::Format::eBc6HSfloatBlock;
    case DXGI_FORMAT_BC7_UNORM:
        return vk::Format::eBc7UnormBlock;
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return vk::Format::eBc7SrgbBlock;
#endif
    default:
        throw std::runtime_error("Unknown DXGI format: " + std::to_string(header.header10.dxgiFormat));
    }
}

DxgiFormat vkFormatToDDSFormat(vk::Format vkFormat) {
    switch (vkFormat) {
    case vk::Format::eR16G16B16A16Sint:
        return DXGI_FORMAT_R16G16B16A16_SINT;
    case vk::Format::eR16G16B16A16Sfloat:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case vk::Format::eR16G16Sfloat:
        return DXGI_FORMAT_R16G16_FLOAT;
    case vk::Format::eB10G11R11UfloatPack32:
        return DXGI_FORMAT_R11G11B10_FLOAT;
    case vk::Format::eD32SfloatS8Uint:
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case vk::Format::eR8G8B8A8Snorm:
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    case vk::Format::eR8G8B8Snorm:
        return DXGI_FORMAT_R8G8B8_SNORM_CUSTOM;
    case vk::Format::eR8G8B8A8Sint:
        return DXGI_FORMAT_R8G8B8A8_SINT;
    case vk::Format::eR8G8B8Sint:
        return DXGI_FORMAT_R8G8B8_SINT_CUSTOM;
    case vk::Format::eR8Snorm:
        return DXGI_FORMAT_R8_SNORM;
    case vk::Format::eR32Sfloat:
        return DXGI_FORMAT_R32_FLOAT;
    case vk::Format::eR8G8Unorm:
        return DXGI_FORMAT_R8G8_UNORM;
    case vk::Format::eR16Sfloat:
        return DXGI_FORMAT_R16_FLOAT;
    case vk::Format::eR8Unorm:
        return DXGI_FORMAT_R8_UNORM;
    case vk::Format::eR32Uint:
        return DXGI_FORMAT_R32_UINT;
    case vk::Format::eR32G32B32A32Sfloat:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case vk::Format::eB8G8R8A8Unorm:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case vk::Format::eR8G8B8A8Unorm:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case vk::Format::eR8G8Uint:
        return DXGI_FORMAT_R8G8_UINT;
    case vk::Format::eR8G8Sint:
        return DXGI_FORMAT_R8G8_SINT;
#ifdef SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT
    // These are image formats that haven't been fully tested yet.
    case vk::Format::eR32G32B32A32Uint:
        return DXGI_FORMAT_R32G32B32A32_UINT;
    case vk::Format::eR32G32B32A32Sint:
        return DXGI_FORMAT_R32G32B32A32_SINT;
    case vk::Format::eR16G16B16A16Unorm:
        return DXGI_FORMAT_R16G16B16A16_UNORM;
    case vk::Format::eR16G16B16A16Uint:
        return DXGI_FORMAT_R16G16B16A16_UINT;
    case vk::Format::eR16G16B16A16Snorm:
        return DXGI_FORMAT_R16G16B16A16_SNORM;
    case vk::Format::eR16G16Unorm:
        return DXGI_FORMAT_R16G16_UNORM;
    case vk::Format::eR16G16Uint:
        return DXGI_FORMAT_R16G16_UINT;
    case vk::Format::eR16G16Snorm:
        return DXGI_FORMAT_R16G16_SNORM;
    case vk::Format::eR16G16Sint:
        return DXGI_FORMAT_R16G16_SINT;
    case vk::Format::eD32Sfloat:
        return DXGI_FORMAT_D32_FLOAT;
    case vk::Format::eR8G8B8A8Srgb:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case vk::Format::eR8G8B8A8Uint:
        return DXGI_FORMAT_R8G8B8A8_UINT;
    case vk::Format::eR8G8Snorm:
        return DXGI_FORMAT_R8G8_SNORM;
    case vk::Format::eR32Sint:
        return DXGI_FORMAT_R32_SINT;
    case vk::Format::eR16Unorm:
        return DXGI_FORMAT_R16_UNORM;
    case vk::Format::eR16Uint:
        return DXGI_FORMAT_R16_UINT;
    case vk::Format::eR16Snorm:
        return DXGI_FORMAT_R16_SNORM;
    case vk::Format::eR16Sint:
        return DXGI_FORMAT_R16_SINT;
    case vk::Format::eR8Uint:
        return DXGI_FORMAT_R8_UINT;
    case vk::Format::eR8Sint:
        return DXGI_FORMAT_R8_SINT;
    case vk::Format::eD24UnormS8Uint:
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case vk::Format::eB8G8R8A8Srgb:
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case vk::Format::eB8G8R8Unorm:
        return DXGI_FORMAT_B8G8R8X8_UNORM;
    case vk::Format::eB8G8R8Srgb:
        return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
    case vk::Format::eBc1RgbaUnormBlock:
        return DXGI_FORMAT_BC1_UNORM;
    case vk::Format::eBc1RgbaSrgbBlock:
        return DXGI_FORMAT_BC1_UNORM_SRGB;
    case vk::Format::eBc2UnormBlock:
        return DXGI_FORMAT_BC2_UNORM;
    case vk::Format::eBc2SrgbBlock:
        return DXGI_FORMAT_BC2_UNORM_SRGB;
    case vk::Format::eBc3UnormBlock:
        return DXGI_FORMAT_BC3_UNORM;
    case vk::Format::eBc3SrgbBlock:
        return DXGI_FORMAT_BC3_UNORM_SRGB;
    case vk::Format::eBc4UnormBlock:
        return DXGI_FORMAT_BC4_UNORM;
    case vk::Format::eBc4SnormBlock:
        return DXGI_FORMAT_BC4_SNORM;
    case vk::Format::eBc5UnormBlock:
        return DXGI_FORMAT_BC5_UNORM;
    case vk::Format::eBc5SnormBlock:
        return DXGI_FORMAT_BC5_SNORM;
    case vk::Format::eBc6HUfloatBlock:
        return DXGI_FORMAT_BC6H_UF16;
    case vk::Format::eBc6HSfloatBlock:
        return DXGI_FORMAT_BC6H_SF16;
    case vk::Format::eBc7UnormBlock:
        return DXGI_FORMAT_BC7_UNORM;
    case vk::Format::eBc7SrgbBlock:
        return DXGI_FORMAT_BC7_UNORM_SRGB;
#endif
    default:
        throw std::runtime_error("Unknown VkFormat: " + vgflib::FormatTypeToName(vgflib::ToFormatType(vkFormat)));
    }
}

uint32_t calculatePitch(uint32_t width, uint32_t elementSize) {
    // Calculate pitch in recommended way
    return (width * elementSize * 8 + 7) / 8;
}

DDSHeaderInfo generateDDSHeader(const Image &image) {
    DDSHeaderInfo header =
        generateDefaultDDSHeader(static_cast<uint32_t>(image.shape()[2]), static_cast<uint32_t>(image.shape()[1]),
                                 elementSizeFromVkFormat(image.dataType()), vkFormatToDDSFormat(image.dataType()));
    validateDDSHeader(header);
    return header;
}

} // namespace

DDSHeaderInfo readDDSHeader(std::ifstream &file) {
    DDSHeaderInfo info;
    file.read(reinterpret_cast<char *>(&info.magicWord), sizeof(info.magicWord));
    file.read(reinterpret_cast<char *>(&info.header), sizeof(info.header));
    info.isDx10 = info.header.pixelFormat.fourCC == DX10_MAGIC_WORD;
    if (info.isDx10) {
        file.read(reinterpret_cast<char *>(&info.header10), sizeof(info.header10));
    }
    return info;
}

void loadDataFromDDS(const std::string &filename, std::vector<uint8_t> &data, vk::Format &initialFormat,
                     uint32_t expectedHeight, uint32_t expectedWidth) {
    std::ifstream file(filename, std::ifstream::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    if (!file.is_open()) {
        throw std::runtime_error("Error while opening DDS file: " + filename);
    }
    DDSHeaderInfo header = readDDSHeader(file);
    validateDDSHeader(header);
    if (expectedHeight && header.header.height != expectedHeight) {
        throw std::runtime_error("DDS image height does not match that in the scenario file: " +
                                 std::to_string(header.header.height) + " vs " + std::to_string(expectedHeight));
    }
    if (expectedWidth && header.header.width != expectedWidth) {
        throw std::runtime_error("DDS image width does not match that in the scenario file: " +
                                 std::to_string(header.header.width) + " vs " + std::to_string(expectedWidth));
    }

    auto dataPos = file.tellg();
    file.seekg(0, std::ios::end);
    auto fileSize = file.tellg() - dataPos;
    if (fileSize < 0) {
        throw std::runtime_error("Failed to get DDS file size: " + filename);
    }

    data.resize(static_cast<size_t>(fileSize));
    file.seekg(dataPos);
    file.read(reinterpret_cast<char *>(data.data()), fileSize);
    initialFormat = ddsFormatToVkFormat(header);
}

vk::Format getVkFormatFromDDS(const std::string &filename) {
    std::ifstream file(filename, std::ifstream::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error while opening DDS file: " + filename);
    }
    file.exceptions(std::ios::badbit | std::ios::failbit);

    DDSHeaderInfo header = readDDSHeader(file);
    validateDDSHeader(header);

    return ddsFormatToVkFormat(header);
}

void saveHeaderToDDS(const DDSHeaderInfo &header, std::ofstream &fstream) {
    fstream.write(reinterpret_cast<const char *>(&header.magicWord), sizeof(header.magicWord));
    fstream.write(reinterpret_cast<const char *>(&header.header), sizeof(header.header));
    if (header.isDx10) {
        fstream.write(reinterpret_cast<const char *>(&header.header10), sizeof(header.header10));
    }
}

void saveDataToDDS(const std::string &filename, const Image &image, const std::vector<char> &data) {
    std::ofstream fstream(filename, std::ofstream::binary);
    if (!fstream.is_open()) {
        throw std::runtime_error("Error creating DDS file: " + filename);
    }
    fstream.exceptions(std::ios::badbit | std::ios::failbit);

    DDSHeaderInfo header = generateDDSHeader(image);
    saveHeaderToDDS(header, fstream);
    fstream.write(data.data(), std::streamsize(data.size()));
    fstream.close();
}

DDSHeaderInfo generateDefaultDDSHeader(uint32_t height, uint32_t width, uint32_t elementSize, DxgiFormat format) {
    return {MAGIC_WORD,
            DDSHeader{/* .size = */ 124,
                      /* .flags = */ 0x100F,
                      /* .height = */ height,
                      /* .width = */ width,
                      /* .pitchOrLinearSize = */ calculatePitch(width, elementSize),
                      /* .depth = */ 1,
                      /* .mipMapCount = */ 1,
                      /* reserved */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                      /* .pixelFormat = */
                      DDSPixelFormat{/* .size = */ 32,
                                     /* .flags = */ 0x4,
                                     /* .fourCC = */ DX10_MAGIC_WORD,
                                     /* .rbgBitCount = */ 0,
                                     /* .rBitMask*/ 0,
                                     /* .gBitMask*/ 0,
                                     /* .bBitMask*/ 0,
                                     /* .aBitMask*/ 0},
                      /* .caps = */ 0x1000,
                      /* .caps2 = */ 0,
                      /* .caps3 = */ 0,
                      /* .caps4 = */ 0,
                      /* reserved */ 0},
            DDSHeaderDX10{/* .dxgiFormat = */ static_cast<uint32_t>(format),
                          /* .resourceDimension = */ 3,
                          /* .miscFlag = */ 0,
                          /* .arraySize = */ 1,
                          /* .miscFlags2 = */ 0},
            true};
}

} // namespace mlsdk::scenariorunner
