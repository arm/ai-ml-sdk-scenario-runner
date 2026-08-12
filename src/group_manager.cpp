/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_manager.hpp"
#include "logging.hpp"

#include <stdexcept>

namespace mlsdk::scenariorunner {

MemoryGroupId GroupManager::createMemoryGroup() {
    if (_finalized) {
        throw std::runtime_error("Cannot create memory group after finalization");
    }
    const MemoryGroupId group{_nextGroupId++};
    _groupResources.emplace(group, std::vector<MemoryResourceId>{});
    return group;
}

void GroupManager::addResourceToGroup(MemoryGroupId group, MemoryResourceId resource) {
    if (_finalized) {
        throw std::runtime_error("Cannot add resource to memory group after finalization");
    }
    const auto groupIt = _groupResources.find(group);
    if (groupIt == _groupResources.end()) {
        throw std::runtime_error("Memory group does not exist");
    }
    const auto [resourceIt, inserted] = _resourceToGroup.emplace(resource, group);
    if (!inserted && resourceIt->second != group) {
        throw std::runtime_error("Resource already belongs to a different group");
    }
    if (!inserted) {
        return;
    }
    logging::debug("addResourceToGroup count of resources: " + std::to_string(_resourceToGroup.size()) +
                   " added type: " + std::to_string(resource.index()));
    groupIt->second.push_back(resource);
}

void GroupManager::finalize() {
    if (_finalized) {
        throw std::runtime_error("Memory groups are already finalized");
    }
    for (const auto &[group, resources] : _groupResources) {
        auto manager = std::make_shared<ResourceMemoryManager>();
        if (resources.size() > 1) {
            manager->markShared();
        }
        _groupMemoryManagers.emplace(group, std::move(manager));
    }
    _finalized = true;
}

size_t GroupManager::getAliasCount(MemoryResourceId resource) const {
    const auto it = _resourceToGroup.find(resource);
    if (it != _resourceToGroup.end()) {
        return _groupResources.at(it->second).size();
    }
    return 0;
}

bool GroupManager::isAliased(MemoryResourceId resource) const { return getAliasCount(resource) > 1; }

std::shared_ptr<ResourceMemoryManager> GroupManager::getMemoryManager(MemoryResourceId resource) {
    if (!_finalized) {
        throw std::runtime_error("Memory groups must be finalized before accessing memory managers");
    }
    if (const auto group = getGroupForResource(resource); group.has_value()) {
        return _groupMemoryManagers.at(*group);
    }
    // Not a group, create new one
    return std::make_shared<ResourceMemoryManager>();
}

std::optional<MemoryGroupId> GroupManager::getGroupForResource(MemoryResourceId resource) const {
    const auto it = _resourceToGroup.find(resource);
    if (it != _resourceToGroup.end()) {
        return it->second;
    }
    return std::nullopt;
}

const GroupResources &GroupManager::getGroups() const { return _groupResources; }

std::vector<MemoryResourceId> GroupManager::getResourcesInGroup(MemoryGroupId group) const {
    const auto it = _groupResources.find(group);
    if (it == _groupResources.end()) {
        return {};
    }
    return it->second;
}

} // namespace mlsdk::scenariorunner
