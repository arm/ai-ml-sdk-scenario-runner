/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "image.hpp"

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

struct ImageSaveOptions {};

struct ImageLoadResult {
    explicit ImageLoadResult(vk::Format vkFormat) : initialFormat(vkFormat) {}

    /// vk::Format of image file
    vk::Format initialFormat{vk::Format::eUndefined};
    /// Pixel data from file
    std::vector<uint8_t> data;
    /// Number of mip levels (only populated if relevant for data format)
    uint32_t mipLevels{1};
};

struct ImageFormatHandler {
    std::set<std::string> extensions; // lower-case extensions including the dot
    std::function<vk::Format(const std::string &filename)> getFormat;
    std::function<ImageLoadResult(const std::string &filename, const ImageLoadOptions &options)> loadData;
    std::function<void(const std::string &filename, const Image &image, const std::vector<char> &data,
                       const ImageSaveOptions &options)>
        saveData;
};

/// \brief Find format handler by filename (extension is extracted and lower-cased).
const ImageFormatHandler *getImageFormatHandler(const std::string &filename);

/// \brief Get Vulkan format for a supported image file.
vk::Format getVkFormatForImage(const std::string &filename);

} // namespace mlsdk::scenariorunner
