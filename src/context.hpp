/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "vulkan/vulkan_raii.hpp"

namespace mlsdk::scenariorunner {

struct ScenarioOptions;

/// \brief Vulkan extensions structure
struct OptionalExtensions {
    bool custom_border_color = false;
    bool mark_boundary = false;
    bool maintenance5 = false;
    bool deferred_operation = false;
    bool replicated_composites = false;
};

/// \brief Context that contains device related information
///
/// Acts as a mechanism to keep commonly used objects in a single place.
/// Example objects are the Vulkan® instance/device etc.
class Context {
  public:
    /// \brief Constructor
    explicit Context(const ScenarioOptions &scenarioOptions);

    /// \brief Logical device accessor
    /// \return Reference to the Vulkan logical device
    const vk::raii::Device &device() const;

    /// \brief Physical device accessor
    /// \return Reference to Vulkan physical device
    const vk::raii::PhysicalDevice &physicalDevice() const;

    /// \brief struct of optional extensions
    OptionalExtensions _optionals;

    /// \brief Index to a compute family queue accessor
    /// \return Index to a compute family queue
    uint32_t computeFamilyQueueIdx() const;

    /// \brief Are GPU debug markers enabled?
    /// \return Whether GPU debug markers are enabled or not
    inline bool gpuDebugMarkersEnabled() const { return _gpuDebugMarkersEnabled; }

    /// @brief Does graph session memory need to be dumped?
    /// @return Whether graph session memory needs to be dumped
    inline bool sessionMemoryDumpEnabled() const { return _sessionMemoryDumpEnabled; }

  private:
    bool _gpuDebugMarkersEnabled;
    bool _sessionMemoryDumpEnabled;
    vk::raii::Context _ctx{};
    vk::raii::Instance _instance{nullptr};
    vk::raii::PhysicalDevice _physicalDev{nullptr};
    vk::raii::Device _dev{nullptr};
    uint32_t _computeQueueIdx{0};
};

} // namespace mlsdk::scenariorunner
