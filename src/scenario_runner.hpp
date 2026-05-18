/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "guid.hpp"
#include "vulkan_memory_manager.hpp"

#include <memory>
#include <set>
#include <unordered_map>
#include <utility>

namespace mlsdk::scenariorunner {

/// All resource types
enum class ResourceIdType {
    Unknown,
    /// These may alias
    Buffer,
    Image,
    Tensor,
    /// Other resources
    RawData,
    Shader,
    VgfView,
    BufferBarrier,
    ImageBarrier,
    MemoryBarrier,
    TensorBarrier,
};

using GroupResourceEntry = std::pair<Guid, ResourceIdType>;
using GroupResources = std::unordered_map<Guid, std::set<GroupResourceEntry>>;

/// Managing memory groups, all aliasing resources belong to a shared memory manager.
class IGroupManager {
  public:
    virtual ~IGroupManager() = default;

    virtual void addResourceToGroup(const Guid &group, const Guid &resource, ResourceIdType resourceIdType) = 0;
    virtual size_t getAliasCount(const Guid &resource) const = 0;
    virtual bool isAliased(const Guid &resource) const = 0;
    virtual bool hasAliasOfType(const Guid &resource, ResourceIdType resourceIdType) const = 0;
    virtual std::shared_ptr<ResourceMemoryManager> getMemoryManager(const Guid &resource) = 0;
    virtual const GroupResources &getGroups() const = 0;
};

} // namespace mlsdk::scenariorunner
