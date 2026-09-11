/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_runner/resource_id.hpp"

#include "vulkan_memory_manager.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace mlsdk::scenariorunner {

using GroupResources = std::unordered_map<MemoryGroupId, std::vector<MemoryResourceId>>;

class GroupManager {
  public:
    MemoryGroupId createMemoryGroup();

    /// Add resource to group
    void addResourceToGroup(MemoryGroupId group, MemoryResourceId resource);

    /// Complete group registration and create the shared memory managers.
    void finalize();

    /// Return size of group that resource belongs to
    size_t getAliasCount(MemoryResourceId resource) const;

    bool isAliased(MemoryResourceId resource) const;

    // Get memory manager, shared if resource is aliased.
    std::shared_ptr<ResourceMemoryManager> getMemoryManager(MemoryResourceId resource);
    const GroupResources &getGroups() const;

    std::optional<MemoryGroupId> getGroupForResource(MemoryResourceId resource) const;
    std::vector<MemoryResourceId> getResourcesInGroup(MemoryGroupId group) const;

  private:
    bool _finalized{false};
    size_t _nextGroupId{};
    std::unordered_map<MemoryResourceId, MemoryGroupId> _resourceToGroup;
    GroupResources _groupResources;
    std::unordered_map<MemoryGroupId, std::shared_ptr<ResourceMemoryManager>> _groupMemoryManagers;
};

} // namespace mlsdk::scenariorunner
