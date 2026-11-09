/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mlworkloadlib/session.hpp"

#include <cstdint>

namespace mlsdk::scenariorunner {

class ResourceMemoryManager;

/// Alias-peer requirements for one shared backing allocation, collected across dispatches.
/// Callers select the group and placements; overlapping placements are allowed.
/// Supply Vulkan allocation requirements, not logical resource byte sizes.
class VgfMemoryPlan {
  public:
    /// Peer offsets are relative to the region base supplied to BindingSet.
    /// A dedicated-allocation preference is optional; a requirement rejects sharing.
    void addAliasPeer(const workloadlib::AliasPeerMemoryRequirements &requirements, vk::DeviceSize regionBase,
                      vk::DeviceSize peerOffset);

    const vk::MemoryRequirements &requirements() const { return _requirements; }

    /// Merge with external requirements already collected by SR resource setup,
    /// before the manager allocates memory.
    void apply(ResourceMemoryManager &backing) const;

    /// Borrow the allocated region to use with the same caller-selected placements.
    workloadlib::BoundMemoryInfo backingRegion(const ResourceMemoryManager &backing, vk::DeviceSize regionBase) const;

  private:
    /// Resources requiring dedicated allocations must be allocated separately;
    /// they cannot be included in this shared backing plan. Supply that requirement
    /// explicitly because vk::MemoryRequirements does not contain it.
    void addResource(const vk::MemoryRequirements &memory, vk::DeviceSize offset, bool requiresDedicatedAllocation);

    // Shared allocation requirements: start with zero extent, no alignment
    // restriction and all memory types, then grow size/alignment and intersect types.
    vk::MemoryRequirements _requirements{0, 1, ~uint32_t{0}};
};

} // namespace mlsdk::scenariorunner
