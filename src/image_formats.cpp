/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_formats.hpp"

#include "dds_reader.hpp"
#include "utils.hpp"

#include <array>
#include <stdexcept>

namespace mlsdk::scenariorunner {

namespace {

const std::array<ImageFormatHandler, 1> kImageFormatHandlers = {
    ImageFormatHandler{
        {".dds"},
        getVkFormatFromDDS,
        loadDataFromDDS,
        saveDataToDDS,
    },
};

} // namespace

const ImageFormatHandler *getImageFormatHandler(const std::string &filename) {
    const auto extension = lowercaseExtension(filename);
    if (extension.empty()) {
        return nullptr;
    }
    for (const auto &handler : kImageFormatHandlers) {
        if (handler.extensions.count(extension) > 0) {
            return &handler;
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
