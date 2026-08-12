/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resource_id.hpp"
#include "vulkan_memory_manager.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace mlsdk::scenariorunner {

using GroupResources = std::unordered_map<MemoryGroupId, std::vector<MemoryResourceId>>;

/// Managing memory groups, all aliasing resources belong to a shared memory manager.
class IGroupManager {
  public:
    virtual ~IGroupManager() = default;

    virtual MemoryGroupId createMemoryGroup() = 0;
    virtual void addResourceToGroup(MemoryGroupId group, MemoryResourceId resource) = 0;
    virtual void finalize() = 0;
    virtual size_t getAliasCount(MemoryResourceId resource) const = 0;
    virtual bool isAliased(MemoryResourceId resource) const = 0;
    virtual std::shared_ptr<ResourceMemoryManager> getMemoryManager(MemoryResourceId resource) = 0;
    virtual const GroupResources &getGroups() const = 0;
};

} // namespace mlsdk::scenariorunner
