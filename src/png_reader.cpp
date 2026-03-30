/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "png_reader.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mlsdk::scenariorunner {

namespace {

size_t checkedSize(int64_t width, int64_t height) {
    if (width <= 0 || height <= 0 || width > (std::numeric_limits<int>::max() / 4) ||
        height > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Dimensions exceed supported limits");
    }
    if (static_cast<uint64_t>(height) >
        (static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / 4) / static_cast<uint64_t>(width)) {
        throw std::runtime_error("Size exceeds addressable memory");
    }
    return static_cast<size_t>(static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4);
}

void validatePNG(const std::string &filename, const ImageLoadOptions &options) {
    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_info(filename.c_str(), &width, &height, &channels)) {
        throw std::runtime_error("Failed to inspect PNG: " + filename);
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("PNG has zero dimension for file: " + filename);
    }

    if ((options.expectedHeight != 0 && static_cast<uint32_t>(height) != options.expectedHeight) ||
        (options.expectedWidth != 0 && static_cast<uint32_t>(width) != options.expectedWidth)) {
        throw std::runtime_error("PNG dimensions do not match expected size for file: " + filename);
    }

    const auto uWidth = static_cast<uint32_t>(width);
    const auto uHeight = static_cast<uint32_t>(height);
    if ((options.maxWidth != 0 && uWidth > options.maxWidth) ||
        (options.maxHeight != 0 && uHeight > options.maxHeight)) {
        throw std::runtime_error("PNG dimensions exceed allowed limits for file: " + filename);
    }
    const uint64_t decodedBytes = static_cast<uint64_t>(uWidth) * uHeight * 4;
    if (options.maxDecodedBytes != 0 && decodedBytes > options.maxDecodedBytes) {
        throw std::runtime_error("PNG decoded size exceeds allowed limit for file: " + filename);
    }
}

} // namespace

vk::Format getVkFormatFromPNG(const std::string &filename) {
    validatePNG(filename, {});
    return vk::Format::eR8G8B8A8Unorm;
}

void loadDataFromPNG(const std::string &filename, std::vector<uint8_t> &data, vk::Format &initialFormat,
                     const ImageLoadOptions &options) {
    validatePNG(filename, options);

    int width = 0;
    int height = 0;
    int channels = 0;
    std::unique_ptr<stbi_uc, void (*)(void *)> decoded(stbi_load(filename.c_str(), &width, &height, &channels, 4),
                                                       stbi_image_free);
    if (!decoded) {
        throw std::runtime_error("Failed to decode PNG: " + filename);
    }

    const size_t expectedSize = checkedSize(width, height);
    data.assign(decoded.get(), decoded.get() + expectedSize);
    initialFormat = vk::Format::eR8G8B8A8Unorm;
}

void saveDataToPNG(const std::string &filename, const Image &image, const std::vector<char> &data,
                   const ImageSaveOptions &) {
    const auto &shape = image.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("Unexpected image shape for PNG export");
    }

    vk::Format format = image.dataType();
    if (format != vk::Format::eR8G8B8A8Unorm) {
        throw std::runtime_error("PNG export supports only VK_FORMAT_R8G8B8A8_UNORM");
    }

    const int64_t width64 = shape[1];
    const int64_t height64 = shape[2];
    const size_t expectedSize = checkedSize(width64, height64);
    if (data.size() != expectedSize) {
        throw std::runtime_error("Unexpected PNG data size for export");
    }

    const int width = static_cast<int>(width64);
    const int height = static_cast<int>(height64);
    const auto *raw = reinterpret_cast<const unsigned char *>(data.data());
    if (stbi_write_png(filename.c_str(), width, height, 4, raw, width * 4) == 0) {
        throw std::runtime_error("Failed to write PNG: " + filename);
    }
}

} // namespace mlsdk::scenariorunner
