/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf_memory_plan.hpp"

#include "utils.hpp"
#include "vulkan_memory_manager.hpp"

#include <algorithm>
#include <stdexcept>

namespace mlsdk::scenariorunner {

void VgfMemoryPlan::addResource(const vk::MemoryRequirements &memory, vk::DeviceSize offset,
                                bool requiresDedicatedAllocation) {
    if (requiresDedicatedAllocation) {
        throw std::invalid_argument("VGF shared backing cannot satisfy a required dedicated allocation");
    }
    if (memory.alignment == 0 || (memory.alignment & (memory.alignment - 1)) != 0 || offset % memory.alignment != 0) {
        throw std::invalid_argument("VGF shared memory placement is misaligned");
    }
    const auto types = _requirements.memoryTypeBits & memory.memoryTypeBits;
    if (types == 0) {
        throw std::invalid_argument("VGF shared memory requirements have no compatible memory type");
    }
    const auto end = checkedAdd(offset, memory.size);
    _requirements.size = std::max(_requirements.size, end);
    _requirements.alignment = std::max(_requirements.alignment, memory.alignment);
    _requirements.memoryTypeBits = types;
}

void VgfMemoryPlan::addAliasPeer(const workloadlib::AliasPeerMemoryRequirements &requirements,
                                 vk::DeviceSize regionBase, vk::DeviceSize peerOffset) {
    addResource(requirements.memory, checkedAdd(regionBase, peerOffset), requirements.requiresDedicatedAllocation);
}

void VgfMemoryPlan::apply(ResourceMemoryManager &backing) const {
    if (_requirements.size == 0) {
        throw std::logic_error("VGF shared memory plan is empty");
    }
    backing.mergeMemoryRequirements(_requirements);
}

workloadlib::BoundMemoryInfo VgfMemoryPlan::backingRegion(const ResourceMemoryManager &backing,
                                                          vk::DeviceSize regionBase) const {
    if (!backing.isInitalized() || backing.getMemSize() < _requirements.size || regionBase >= backing.getMemSize()) {
        throw std::logic_error("VGF backing region requires a sufficiently large allocated memory manager");
    }
    return {*backing.getDeviceMemory(), regionBase, backing.getMemSize() - regionBase};
}

} // namespace mlsdk::scenariorunner
