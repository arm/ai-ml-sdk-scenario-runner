/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <vector>

#include "types.hpp"

namespace mlsdk::scenariorunner {

/// @brief Non-owning bytes used to upload a buffer.
struct BufferDataView {
    /// @brief Address of the first byte, or nullptr when size is zero.
    const void *data{nullptr};
    /// @brief Number of bytes available at data.
    size_t size{0};
};

/// @brief Owning buffer contents returned by Scenario::download().
struct BufferData {
    /// @brief Downloaded bytes.
    std::vector<std::byte> data;
};

/// @brief Non-owning bytes and optional metadata used to upload a tensor.
struct TensorDataView {
    /// @brief Address of the first byte, or nullptr when size is zero.
    const void *data{nullptr};
    /// @brief Number of bytes available at data.
    size_t size{0};
    /// @brief Tensor shape to validate.
    ///
    /// An empty shape is accepted only for a rank-converted scalar tensor.
    std::vector<int64_t> shape;
    /// @brief Tensor format to validate, or std::nullopt to omit format validation.
    std::optional<vk::Format> format{std::nullopt};
};

/// @brief Owning tensor contents and metadata returned by Scenario::download().
struct TensorData {
    /// @brief Downloaded bytes in tightly packed tensor order.
    std::vector<std::byte> data;
    /// @brief Logical tensor shape.
    std::vector<int64_t> shape;
    /// @brief Tensor element format when available.
    std::optional<vk::Format> format{std::nullopt};
};

/// @brief Non-owning bytes and optional metadata used to upload an image.
struct ImageDataView {
    /// @brief Address of the first byte, or nullptr when size is zero.
    const void *data{nullptr};
    /// @brief Number of bytes available at data.
    size_t size{0};
    /// @brief Image shape to validate; it must match the resource shape.
    std::vector<int64_t> shape;
    /// @brief Image format to validate, or std::nullopt to omit format validation.
    std::optional<vk::Format> format{std::nullopt};
    /// @brief Number of mip levels contained in data.
    uint32_t mipLevels{1};
};

/// @brief Owning image contents and metadata returned by Scenario::download().
struct ImageData {
    /// @brief Downloaded base-mip bytes in tightly packed image order.
    std::vector<std::byte> data;
    /// @brief Logical image shape.
    std::vector<int64_t> shape;
    /// @brief Image texel format when available.
    std::optional<vk::Format> format{std::nullopt};
    /// @brief Number of mip levels represented in data.
    uint32_t mipLevels{1};
};

} // namespace mlsdk::scenariorunner
