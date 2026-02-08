/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include "image.hpp"
#include "image_formats.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include "vulkan/vulkan_core.h"

namespace mlsdk::scenariorunner {

/// \brief Load data from a PNG file
///
/// \param filename PNG file to load
/// \param data pixel data from file
/// \param initialFormat vk::Format of PNG file
/// \param options load options
void loadDataFromPNG(const std::string &filename, std::vector<uint8_t> &data, vk::Format &initialFormat,
                     const ImageLoadOptions &options);

/// \brief Get vk::Format from a PNG file
///
/// \param filename PNG file to load
vk::Format getVkFormatFromPNG(const std::string &filename);

/// \brief Create PNG file from image data (expects RGBA8 data)
///
/// \param filename file to create
/// \param image image data to save to file
/// \param data vector of raw data to save
/// \param options save options
void saveDataToPNG(const std::string &filename, const Image &image, const std::vector<char> &data,
                   const ImageSaveOptions &options);

} // namespace mlsdk::scenariorunner
