/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <vector>

#include "image.hpp"
#include "vulkan/vulkan_core.h"

namespace mlsdk::scenariorunner {

enum DxgiFormat {
    DXGI_FORMAT_UNKNOWN = 0,
    DXGI_FORMAT_R32G32B32A32_FLOAT = 2,
    DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
    DXGI_FORMAT_R16G16B16A16_SINT = 14,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT = 20,
    DXGI_FORMAT_R11G11B10_FLOAT = 26,
    DXGI_FORMAT_R8G8B8A8_UNORM = 28,
    DXGI_FORMAT_R8G8B8A8_SNORM = 31,
    DXGI_FORMAT_R8G8B8A8_SINT = 32,
    DXGI_FORMAT_R16G16_FLOAT = 34,
    DXGI_FORMAT_R32_FLOAT = 41,
    DXGI_FORMAT_R32_UINT = 42,
    DXGI_FORMAT_R8G8_UNORM = 49,
    DXGI_FORMAT_R8G8_SINT = 50,
    DXGI_FORMAT_R16_FLOAT = 54,
    DXGI_FORMAT_R8_UNORM = 61,
    DXGI_FORMAT_R8_SNORM = 63,
    // These constants are not present in the DXGI standard.
    // todo: remove when non dxgi texture formats support is added.
    DXGI_FORMAT_R8G8B8_SNORM_CUSTOM = 133,
    DXGI_FORMAT_R8G8B8_SINT_CUSTOM = 134
};

/// \brief Structure that describes DDS pixel data layout
struct DDSPixelFormat {
    uint32_t size{32};
    uint32_t flags{};
    uint32_t fourCC{};
    uint32_t rgbBitCount{};
    uint32_t rBitMask{};
    uint32_t gBitMask{};
    uint32_t bBitMask{};
    uint32_t aBitMask{};
};

/// \brief Structure that describes main part of header from DDS file
///
/// \note Order of members and presence of "Reserved" data allow memcpy to work on file data
struct DDSHeader {
    uint32_t size{124};
    uint32_t flags{};
    uint32_t height{};
    uint32_t width{};
    uint32_t pitchOrLinearSize{};
    uint32_t depth{};
    uint32_t mipMapCount{};
    uint32_t reserved[11]{};
    DDSPixelFormat pixelFormat{};
    uint32_t caps{};
    uint32_t caps2{};
    uint32_t caps3{};
    uint32_t caps4{};
    uint32_t reserved2{};
};

/// \brief Structure that describes optional extension to DDS header format
struct DDSHeaderDX10 {
    uint32_t dxgiFormat{};
    uint32_t resourceDimension{};
    uint32_t miscFlag{};
    uint32_t arraySize{};
    uint32_t miscFlags2{};
};

/// \brief Structure that stores all non-pixel data from DDS file
struct DDSHeaderInfo {
    uint32_t magicWord{};
    DDSHeader header{};
    DDSHeaderDX10 header10{};
    bool isDx10{};
};

/// \brief Load data from a DDS file
///
/// \param filename DDS file to load
/// \param data pixel data from file
/// \param initialFormat vk::Format of DDS file
void loadDataFromDDS(const std::string &filename, std::vector<uint8_t> &data, vk::Format &initialFormat);

/// \brief Get vk::Format from a DDS file
///
/// \param filename DDS file to load
vk::Format getVkFormatFromDDS(const std::string &filename);

/// @brief Generate DDS header information from byte data of DDS file
///
/// @param file stream of dds file
DDSHeaderInfo readDDSHeader(std::ifstream &file);

/// \brief Write all header data to file stream
///
/// \param header Header information to write
/// \param fstream stream to write data to
void saveHeaderToDDS(const DDSHeaderInfo &header, std::ofstream &fstream);

/// \brief Create DDS file "filename" from image data
///
/// \param filename file to create
/// \param image image data to save to file
/// \param data vector of raw data to save
void saveDataToDDS(const std::string &filename, const Image &image, const std::vector<char> &data);

/// \brief Create a standard DDS (DX10) file header
///
/// \param height image height
/// \param width image width
/// \param elementSize image pixel datatype size (bytes)
/// \param format image pixel datatype
DDSHeaderInfo generateDefaultDDSHeader(uint32_t height, uint32_t width, uint32_t elementSize, DxgiFormat format);

} // namespace mlsdk::scenariorunner
