/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_runner.hpp"

#include <unordered_map>

namespace mlsdk::scenariorunner {

class GroupManager : public IGroupManager {
  public:
    MemoryGroupId createMemoryGroup() override;

    /// Add resource to group
    void addResourceToGroup(MemoryGroupId group, MemoryResourceId resource) override;

    /// Complete group registration and create the shared memory managers.
    void finalize() override;

    /// Return size of group that resource belongs to
    size_t getAliasCount(MemoryResourceId resource) const override;

    bool isAliased(MemoryResourceId resource) const override;

    // Get memory manager, shared if resource is aliased.
    std::shared_ptr<ResourceMemoryManager> getMemoryManager(MemoryResourceId resource) override;
    const GroupResources &getGroups() const override;

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
