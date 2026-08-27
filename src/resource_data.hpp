/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <vector>

#include "types.hpp"

namespace mlsdk::scenariorunner {

struct BufferDataView {
    const void *data{nullptr};
    size_t size{0};
};

struct BufferData {
    std::vector<char> data;
};

struct TensorDataView {
    const void *data{nullptr};
    size_t size{0};
    std::vector<int64_t> shape;
    std::optional<vk::Format> format{std::nullopt}; // optional: validate against tensor format
};

struct TensorData {
    std::vector<char> data;
    std::vector<int64_t> shape;
    std::optional<vk::Format> format{std::nullopt};
};

struct ImageDataView {
    const void *data{nullptr};
    size_t size{0};
    std::vector<int64_t> shape;
    std::optional<vk::Format> format{std::nullopt};
    uint32_t mipLevels{1};
};

struct ImageData {
    std::vector<std::byte> data;
    std::vector<int64_t> shape;
    std::optional<vk::Format> format{std::nullopt};
    uint32_t mipLevels{1};
};

} // namespace mlsdk::scenariorunner
