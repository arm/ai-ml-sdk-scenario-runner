/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vgf_memory_plan.hpp"
#include "vulkan_memory_manager.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {
using namespace mlsdk::scenariorunner;
namespace wl = mlsdk::workloadlib;

TEST(VgfMemoryPlan, CombinesPeerRequirementsAcrossDispatches) {
    VgfMemoryPlan plan;
    plan.addAliasPeer({{64, 16, 0b1100}}, 128, 0);
    plan.addAliasPeer({{256, 64, 0b1010}}, 128, 64);
    plan.addAliasPeer({{32, 128, 0b1000}}, 256, 0);
    // Aliases can deliberately overlap; sizes are extents, not a sum.
    plan.addAliasPeer({{16, 16, 0b1111}}, 128, 0);
    EXPECT_EQ(plan.requirements().size, 448u);
    EXPECT_EQ(plan.requirements().alignment, 128u);
    EXPECT_EQ(plan.requirements().memoryTypeBits, 0b1000u);
}

TEST(VgfMemoryPlan, ChecksEffectiveOffsetsAndPreservesPlanOnFailure) {
    VgfMemoryPlan plan;
    plan.addAliasPeer({{16, 8, 1}}, 3, 5);
    const auto before = plan.requirements();
    EXPECT_THROW(plan.addAliasPeer({{16, 8, 1}}, 3, 4), std::invalid_argument);
    EXPECT_THROW(plan.addAliasPeer({{16, 0, 1}}, 0, 0), std::invalid_argument);
    EXPECT_THROW(plan.addAliasPeer({{16, 3, 1}}, 0, 0), std::invalid_argument);
    EXPECT_THROW(plan.addAliasPeer({{16, 8, 2}}, 0, 0), std::invalid_argument);
    EXPECT_THROW(plan.addAliasPeer({{16, 8, 0}}, 0, 0), std::invalid_argument);
    const auto maximum = std::numeric_limits<vk::DeviceSize>::max();
    EXPECT_THROW(plan.addAliasPeer({{1, 1, 1}}, maximum, 1), std::overflow_error);
    EXPECT_THROW(plan.addAliasPeer({{16, 1, 1}}, maximum - 8, 0), std::overflow_error);
    EXPECT_EQ(plan.requirements(), before);
}

TEST(VgfMemoryPlan, RejectsRequiredDedicatedAllocationsButAllowsPreferences) {
    VgfMemoryPlan plan;
    wl::AliasPeerMemoryRequirements peer{{64, 16, 1}, true, false};
    plan.addAliasPeer(peer, 0, 0);
    peer.requiresDedicatedAllocation = true;
    EXPECT_THROW(plan.addAliasPeer(peer, 0, 0), std::invalid_argument);
    EXPECT_EQ(plan.requirements().size, 64u);
}

TEST(VgfMemoryPlan, MergesPeerRequirementsWithExistingBackingBeforeAllocation) {
    VgfMemoryPlan plan;
    plan.addAliasPeer({{400, 16, 0b1100}}, 0, 0);
    ResourceMemoryManager backing;
    backing.updateMemSize(512);
    backing.updateMemType(0b1110);
    plan.apply(backing);
    EXPECT_EQ(backing.getMemSize(), 512u);
    EXPECT_EQ(backing.getMemType(), 0b1100u);
    EXPECT_FALSE(backing.isInitalized());
    EXPECT_THROW(plan.backingRegion(backing, 0), std::logic_error);

    // Applying another dispatch can enlarge the same group before allocation.
    VgfMemoryPlan later;
    later.addAliasPeer({{128, 64, 0b1000}}, 512, 64);
    later.apply(backing);
    EXPECT_EQ(backing.getMemSize(), 704u);
    EXPECT_EQ(backing.getMemType(), 0b1000u);
}

TEST(VgfMemoryPlan, RejectsEmptyOrIncompatiblePlansWithoutChangingBacking) {
    ResourceMemoryManager backing;
    backing.updateMemSize(64);
    backing.updateMemType(1);
    EXPECT_THROW(VgfMemoryPlan{}.apply(backing), std::logic_error);
    VgfMemoryPlan plan;
    plan.addAliasPeer({{128, 16, 2}}, 0, 0);
    EXPECT_THROW(plan.apply(backing), std::invalid_argument);
    EXPECT_EQ(backing.getMemSize(), 64u);
    EXPECT_EQ(backing.getMemType(), 1u);
}

} // namespace
