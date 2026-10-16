/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_build_data.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace mlsdk::scenariorunner::detail {

enum class DeviceCapability : uint32_t {
    DataGraph = 0x01,
    Tensors = 0x02,
    DataGraphShaderModules = 0x05,               // Includes DataGraph.
    DataGraphBinaries = 0x09,                    // Includes DataGraph.
    DataGraphOpticalFlow = 0x11,                 // Includes DataGraph.
    DataGraphNeuralAcceleratorStatistics = 0x21, // Includes DataGraph.
    DynamicRendering = 0x40,
};

constexpr DeviceCapability operator|(DeviceCapability lhs, DeviceCapability rhs) noexcept {
    return static_cast<DeviceCapability>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr DeviceCapability &operator|=(DeviceCapability &lhs, DeviceCapability rhs) noexcept {
    lhs = lhs | rhs;
    return lhs;
}

struct DeviceRequirements {
    vk::QueueFlags queueFlags{vk::QueueFlagBits::eCompute};
    DeviceCapability capabilities{DeviceCapability::DataGraph};
};

struct DeviceCandidate {
    vk::PhysicalDeviceType type{vk::PhysicalDeviceType::eOther};
    DeviceCapability capabilities{};
    std::vector<vk::QueueFamilyProperties> queueFamilies;
    std::vector<uint32_t> opticalFlowQueueFamilyIndices;
};

struct DeviceSelection {
    size_t deviceIndex{};
    uint32_t queueFamilyIndex{};
};

DeviceRequirements getDeviceRequirements(const ScenarioBuildData &buildData, const ScenarioOptions &scenarioOptions);
bool hasCapabilities(DeviceCapability available, DeviceCapability required) noexcept;
bool supportsQueueRequirements(vk::QueueFlags availableFlags, vk::QueueFlags workloadFlags);
DeviceSelection selectDevice(const std::vector<DeviceCandidate> &candidates, const DeviceRequirements &requirements);

} // namespace mlsdk::scenariorunner::detail
