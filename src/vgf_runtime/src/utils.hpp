/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vgf_runtime::detail {
namespace utils {

[[noreturn]] inline void throwNotImplemented(const char *api) {
    throw std::runtime_error(std::string(api) + " is not implemented yet");
}

inline vk::DeviceSize elementCount(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
        return 0;
    }
    vk::DeviceSize elements = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0) {
            return 0;
        }
        elements *= static_cast<vk::DeviceSize>(dimension);
    }
    return elements;
}

} // namespace utils

namespace vulkan_helpers {

inline uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t memoryTypeBits,
                               vk::MemoryPropertyFlags requiredFlags) {
    const auto memoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool supportsType = (memoryTypeBits & (uint32_t{1} << i)) != 0;
        const bool hasFlags = (memoryProperties.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags;
        if (supportsType && hasFlags) {
            return i;
        }
    }
    throw std::runtime_error("Cannot find a compatible memory type");
}

} // namespace vulkan_helpers
} // namespace vgf_runtime::detail
