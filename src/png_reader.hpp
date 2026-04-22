/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "image_formats.hpp"
#include "vulkan/vulkan_core.h"

namespace mlsdk::scenariorunner {

/// \brief Load data from a PNG file
///
/// \param filename PNG file to load
/// \param options load options
/// \return struct containing loaded data and metadata about it
ImageLoadResult loadDataFromPNG(const std::string &filename, const ImageLoadOptions &options);

/// \brief Get vk::Format from a PNG file
///
/// \param filename PNG file to load
vk::Format getVkFormatFromPNG(const std::string &filename);

/// \brief Create PNG file from image data (expects RGBA8 data)
///
/// \param filename file to create
/// \param options save options
void saveDataToPNG(const std::string &filename, const ImageSaveOptions &options);

} // namespace mlsdk::scenariorunner
