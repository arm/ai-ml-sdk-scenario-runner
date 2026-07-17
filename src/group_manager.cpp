/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_manager.hpp"
#include "logging.hpp"

#include <stdexcept>

namespace mlsdk::scenariorunner {

void GroupManager::addResourceToGroup(const Guid &group, const Guid &resource, ResourceIdType resourceIdType) {
    const auto [resourceIt, inserted] = _resourceToGroup.emplace(resource, group);
    if (!inserted && resourceIt->second != group) {
        throw std::runtime_error("Resource already belongs to a different group");
    }
    logging::debug("addResourceToGroup count of resources: " + std::to_string(_resourceToGroup.size()) +
                   " added type: " + std::to_string(static_cast<int>(resourceIdType)));
    _groupResources[group].insert({resource, resourceIdType});
}

size_t GroupManager::getAliasCount(const Guid &resource) const {
    const auto it = _resourceToGroup.find(resource);
    if (it != _resourceToGroup.end()) {
        return _groupResources.at(it->second).size();
    }
    return 0;
}

bool GroupManager::isAliased(const Guid &resource) const { return getAliasCount(resource) > 1; }

bool GroupManager::hasAliasOfType(const Guid &resource, ResourceIdType resourceIdType) const {
    if (const auto group = getGroupForResource(resource); group.has_value()) {
        // Group found, look for requested type
        for (const auto &[aliasResource, aliasType] : _groupResources.at(*group)) {
            if (aliasResource != resource && aliasType == resourceIdType) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<ResourceMemoryManager> GroupManager::getMemoryManager(const Guid &resource) {
    if (const auto group = getGroupForResource(resource); group.has_value()) {
        // Create one memory manager per group on first access.
        const auto [manager, inserted] =
            _groupMemoryManagers.emplace(*group, std::make_shared<ResourceMemoryManager>());
        if (inserted && isAliased(resource)) {
            manager->second->markShared();
        }
        return manager->second;
    }
    // Not a group, create new one
    return std::make_shared<ResourceMemoryManager>();
}

std::optional<Guid> GroupManager::getGroupForResource(const Guid &resource) const {
    const auto it = _resourceToGroup.find(resource);
    if (it != _resourceToGroup.end()) {
        return it->second;
    }
    return std::nullopt;
}

const GroupResources &GroupManager::getGroups() const { return _groupResources; }

std::vector<GroupResourceEntry> GroupManager::getResourcesInGroup(const Guid &group) const {
    const auto it = _groupResources.find(group);
    if (it == _groupResources.end()) {
        return {};
    }
    return {it->second.begin(), it->second.end()};
}

} // namespace mlsdk::scenariorunner
