/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
    case DXGI_FORMAT_R8G8_SINT:
        return vk::Format::eR8G8Sint;
    case DXGI_FORMAT_R8G8_UNORM:
        return vk::Format::eR8G8Unorm;
    case DXGI_FORMAT_R8_SNORM:
        return vk::Format::eR8Snorm;
    case DXGI_FORMAT_R32_FLOAT:
        return vk::Format::eR32Sfloat;
    case DXGI_FORMAT_R16_FLOAT:
        return vk::Format::eR16Sfloat;
    case DXGI_FORMAT_R8_UNORM:
        return vk::Format::eR8Unorm;
    default:
        throw std::runtime_error("Unknown DXGI format: " + std::to_string(header.header10.dxgiFormat));
    }
}

DxgiFormat vkFormatToDDSFormat(vk::Format vkFormat) {
    switch (vkFormat) {
    case vk::Format::eR32G32B32A32Sfloat:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case vk::Format::eR16G16B16A16Sfloat:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case vk::Format::eR16G16Sfloat:
        return DXGI_FORMAT_R16G16_FLOAT;
    case vk::Format::eB10G11R11UfloatPack32:
        return DXGI_FORMAT_R11G11B10_FLOAT;
    case vk::Format::eD32SfloatS8Uint:
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case vk::Format::eR8G8B8A8Unorm:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case vk::Format::eR8G8B8A8Snorm:
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    case vk::Format::eR8G8B8Snorm:
        return DXGI_FORMAT_R8G8B8_SNORM_CUSTOM;
    case vk::Format::eR8G8B8A8Sint:
        return DXGI_FORMAT_R8G8B8A8_SINT;
    case vk::Format::eR8G8B8Sint:
        return DXGI_FORMAT_R8G8B8_SINT_CUSTOM;
    case vk::Format::eR8G8Sint:
        return DXGI_FORMAT_R8G8_SINT;
    case vk::Format::eR8G8Unorm:
        return DXGI_FORMAT_R8G8_UNORM;
    case vk::Format::eR8Snorm:
        return DXGI_FORMAT_R8_SNORM;
    case vk::Format::eR32Sfloat:
        return DXGI_FORMAT_R32_FLOAT;
    case vk::Format::eR16Sfloat:
        return DXGI_FORMAT_R16_FLOAT;
    case vk::Format::eR8Unorm:
        return DXGI_FORMAT_R8_UNORM;
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

void loadDataFromDDS(const std::string &filename, std::vector<uint8_t> &data, vk::Format &initialFormat) {
    std::ifstream file(filename, std::ifstream::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    if (!file.is_open()) {
        throw std::runtime_error("Error while opening DDS file: " + filename);
    }

    DDSHeaderInfo header = readDDSHeader(file);
    validateDDSHeader(header);

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
