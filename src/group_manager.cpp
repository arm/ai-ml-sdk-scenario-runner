/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_manager.hpp"
#include "logging.hpp"

namespace mlsdk::scenariorunner {

void GroupManager::addResourceToGroup(const Guid &group, const Guid &resource, ResourceIdType resourceIdType) {
    _resourceToGroup.emplace(resource, group);
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

bool GroupManager::isAliased(const Guid &resource) const { return getAliasCount(resource) > 0; }

bool GroupManager::isAliasedTo(const Guid &resource, ResourceIdType resourceIdType) const {
    const auto it = _resourceToGroup.find(resource);
    if (it != _resourceToGroup.end()) {
        // Group found, look for requested type
        for ([[maybe_unused]] const auto &[_, type] : _groupResources.at(it->second)) {
            if (type == resourceIdType) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<ResourceMemoryManager> GroupManager::getMemoryManager(const Guid &resource) {
    const auto it = _resourceToGroup.find(resource);
    if (it != _resourceToGroup.end()) {
        const auto &group = it->second;
        // Only inserts if not existing
        const auto result = _groupMemoryManagers.emplace(group, std::make_shared<ResourceMemoryManager>());
        return result.first->second;
    }
    // Not a group, create new one
    return std::make_shared<ResourceMemoryManager>();
}

} // namespace mlsdk::scenariorunner
