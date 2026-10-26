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
    DataGraph = 1U << 0,
    DataGraphShaderModules = 1U << 1,
    DataGraphBinaries = 1U << 2,
    DataGraphOpticalFlow = 1U << 3,
    DataGraphNeuralAcceleratorStatistics = 1U << 4,
    DataGraphImageLayouts = 1U << 5,
    DynamicRendering = 1U << 6,
    ShaderTensorAccess = 1U << 7,
    Tensors = 1U << 8,
};

using DeviceCapabilities = vk::Flags<DeviceCapability>;

constexpr DeviceCapabilities operator|(DeviceCapability lhs, DeviceCapability rhs) noexcept {
    return DeviceCapabilities(lhs) | rhs;
}

struct DeviceRequirements {
    vk::QueueFlags queueFlags{vk::QueueFlagBits::eCompute};
    DeviceCapabilities capabilities;
};

struct DeviceCandidate {
    vk::PhysicalDeviceType type{vk::PhysicalDeviceType::eOther};
    DeviceCapabilities capabilities;
    std::vector<vk::QueueFamilyProperties> queueFamilies;
    std::vector<uint32_t> opticalFlowQueueFamilyIndices;
};

struct DeviceSelection {
    size_t deviceIndex{};
    uint32_t queueFamilyIndex{};
};

DeviceRequirements getDeviceRequirements(const ScenarioBuildData &buildData, const ScenarioOptions &scenarioOptions);
// Expand workload dependencies without inferring any additional device support.
DeviceRequirements expandDeviceRequirements(DeviceRequirements requirements) noexcept;
bool hasCapabilities(DeviceCapabilities available, DeviceCapabilities required) noexcept;
bool supportsQueueRequirements(vk::QueueFlags availableFlags, vk::QueueFlags workloadFlags);
DeviceSelection selectDevice(const std::vector<DeviceCandidate> &candidates, const DeviceRequirements &requirements);

} // namespace mlsdk::scenariorunner::detail
