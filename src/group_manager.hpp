/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_runner.hpp"

#include <set>
#include <unordered_map>

namespace mlsdk::scenariorunner {

class GroupManager : public IGroupManager {
  public:
    /// Create or add resource to group
    void addResourceToGroup(const Guid &group, const Guid &resource, ResourceIdType resourceIdType) override;

    /// Return size of group that resource belongs to
    size_t getAliasCount(const Guid &resource) const override;

    bool isAliased(const Guid &resource) const override;

    ///  Returns true if resource is aliased with any resource of type
    bool hasAliasOfType(const Guid &resource, ResourceIdType resourceIdType) const override;

    // Get memory manager, shared if resource is aliased.
    std::shared_ptr<ResourceMemoryManager> getMemoryManager(const Guid &resource) override;
    const GroupResources &getGroups() const override;

    std::optional<Guid> getGroupForResource(const Guid &resource) const;
    std::vector<GroupResourceEntry> getResourcesInGroup(const Guid &group) const;

  private:
    std::unordered_map<Guid, Guid> _resourceToGroup;
    GroupResources _groupResources;
    std::unordered_map<Guid, std::shared_ptr<ResourceMemoryManager>> _groupMemoryManagers;
};

} // namespace mlsdk::scenariorunner
