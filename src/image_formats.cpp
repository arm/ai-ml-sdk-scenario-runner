/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_formats.hpp"

#include "dds_reader.hpp"
#include "png_reader.hpp"
#include "utils.hpp"

#include <array>
#include <stdexcept>

namespace mlsdk::scenariorunner {

namespace {

struct HandlerMapping {
    std::string_view extension; // lower-case extension including the dot
    ImageFormatHandler handler;
};

constexpr std::array kHandlerMappings = {
    HandlerMapping{".dds",
                   {
                       getVkFormatFromDDS,
                       loadDataFromDDS,
                       saveDataToDDS,
                   }},
    HandlerMapping{".png",
                   {
                       getVkFormatFromPNG,
                       loadDataFromPNG,
                       saveDataToPNG,
                   }},
};

} // namespace

const ImageFormatHandler *getImageFormatHandler(const std::string &filename) {
    const auto extension = lowercaseExtension(filename);
    if (extension.empty()) {
        return nullptr;
    }
    for (const auto &mapping : kHandlerMappings) {
        if (mapping.extension == extension) {
            return &mapping.handler;
        }
    }
    return nullptr;
}

vk::Format getVkFormatForImage(const std::string &filename) {
    const auto *handler = getImageFormatHandler(filename);
    if (!handler) {
        throw std::runtime_error("Unsupported image format for file: " + filename);
    }
    return handler->getFormat(filename);
}

} // namespace mlsdk::scenariorunner
