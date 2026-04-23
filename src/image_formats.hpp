/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace mlsdk::scenariorunner {

struct ImageLoadOptions {
    // Common options
    /// Expected height in pixels (0 to skip check).
    uint32_t expectedHeight{0};
    /// Expected width in pixels (0 to skip check).
    uint32_t expectedWidth{0};

    // PNG options
    /// Maximum allowed height in pixels (0 to skip check).
    uint32_t maxHeight{8192};
    /// Maximum allowed width in pixels (0 to skip check).
    uint32_t maxWidth{8192};
    /// Maximum decoded byte size (0 to skip check).
    uint64_t maxDecodedBytes{8192 * 8192 * 4};
};

struct ImageSaveOptions {
    /// Image shape expected as NHWC.
    std::vector<int64_t> shape;
    /// vk::Format of source image data.
    vk::Format dataType{vk::Format::eUndefined};
    /// Pixel data to save.
    const std::vector<char> &data;
};

struct ImageLoadResult {
    explicit ImageLoadResult(vk::Format vkFormat, uint32_t imageWidth, uint32_t imageHeight)
        : initialFormat(vkFormat), width(imageWidth), height(imageHeight) {}

    /// vk::Format of image file
    vk::Format initialFormat{vk::Format::eUndefined};
    /// Image width in pixels
    uint32_t width{0};
    /// Image height in pixels
    uint32_t height{0};
    /// Pixel data from file
    std::vector<uint8_t> data;
    /// Number of mip levels (only populated if relevant for data format)
    uint32_t mipLevels{1};
};

struct ImageFormatHandler {
    vk::Format (*getFormat)(const std::string &filename);
    ImageLoadResult (*loadData)(const std::string &filename, const ImageLoadOptions &options);
    void (*saveData)(const std::string &filename, const ImageSaveOptions &options);
};

/// \brief Find format handler by filename (extension is extracted and lower-cased).
const ImageFormatHandler *getImageFormatHandler(const std::string &filename);

/// \brief Get Vulkan format for a supported image file.
vk::Format getVkFormatForImage(const std::string &filename);

} // namespace mlsdk::scenariorunner
