/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "guid.hpp"
#include "vulkan_memory_manager.hpp"

#include <set>
#include <string>
#include <unordered_map>

namespace mlsdk::scenariorunner {

enum class ResourceIdType {
    Unknown,
    Buffer,
    Image,
    Tensor,
    RawData,
};

class GroupManager {
  public:
    /// Create or add resource to group
    void addResourceToGroup(const Guid &group, const Guid &resource, ResourceIdType resourceIdType);

    /// Return size of group that resource belongs to
    size_t getAliasCount(const Guid &resource) const;

    bool isAliased(const Guid &resource) const;

    ///  Returns true if resource is aliased with any resource of type
    bool hasAliasOfType(const Guid &resource, ResourceIdType resourceIdType) const;

    const std::unordered_map<Guid, std::set<std::pair<Guid, ResourceIdType>>> &getGroupResources() const {
        return _groupResources;
    }

    // Get memory manager, shared if resource is aliased.
    std::shared_ptr<ResourceMemoryManager> getMemoryManager(const Guid &resource);

  private:
    std::unordered_map<Guid, Guid> _resourceToGroup;
    std::unordered_map<Guid, std::set<std::pair<Guid, ResourceIdType>>> _groupResources;
    std::unordered_map<Guid, std::shared_ptr<ResourceMemoryManager>> _groupMemoryManagers{};
};

} // namespace mlsdk::scenariorunner
