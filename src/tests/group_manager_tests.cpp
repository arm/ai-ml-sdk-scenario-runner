/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_manager.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace mlsdk::scenariorunner;

TEST(GroupManager, GroupHandling) {
    const TensorId tensor0{0};
    const ImageId image0{0};
    const ImageId image1{1};
    GroupManager gm;
    const auto group0 = gm.createMemoryGroup();
    gm.addResourceToGroup(group0, tensor0);
    ASSERT_EQ(gm.getAliasCount(tensor0), 1U);
    ASSERT_FALSE(gm.isAliased(tensor0));
    ASSERT_EQ(gm.getAliasCount(image1), 0U);
    ASSERT_FALSE(gm.isAliased(image1));

    gm.addResourceToGroup(group0, image0);
    ASSERT_EQ(gm.getAliasCount(tensor0), 2U);
    ASSERT_TRUE(gm.isAliased(tensor0));
    ASSERT_EQ(gm.getAliasCount(image0), 2U);
    ASSERT_TRUE(gm.isAliased(image0));
    ASSERT_THROW(static_cast<void>(gm.getMemoryManager(tensor0)), std::runtime_error);
    gm.finalize();
    auto mmTensor = gm.getMemoryManager(tensor0);
    auto mmImage = gm.getMemoryManager(image0);
    ASSERT_EQ(mmTensor, mmImage);
    ASSERT_TRUE(mmTensor->isShared());
}

TEST(GroupManager, DuplicateResourceRegistrationToDifferentGroupThrows) {
    const TensorId tensor0{0};
    GroupManager gm;
    const auto group0 = gm.createMemoryGroup();
    const auto group1 = gm.createMemoryGroup();

    gm.addResourceToGroup(group0, tensor0);

    ASSERT_THROW(gm.addResourceToGroup(group1, tensor0), std::runtime_error);
}

TEST(GroupManager, GroupQueries) {
    const TensorId tensor0{0};
    const ImageId image0{0};
    GroupManager gm;
    const auto group0 = gm.createMemoryGroup();

    gm.addResourceToGroup(group0, tensor0);
    gm.addResourceToGroup(group0, image0);

    const auto group = gm.getGroupForResource(tensor0);
    ASSERT_TRUE(group.has_value());
    ASSERT_EQ(*group, group0);

    const auto groups = gm.getGroups();
    ASSERT_EQ(groups.size(), 1U);
    ASSERT_NE(groups.find(group0), groups.end());

    const auto resources = gm.getResourcesInGroup(group0);
    ASSERT_EQ(resources.size(), 2U);
}

TEST(GroupManager, FinalizationPreventsFurtherRegistration) {
    GroupManager gm;
    const auto group = gm.createMemoryGroup();
    gm.addResourceToGroup(group, BufferId{0});

    gm.finalize();

    ASSERT_THROW(static_cast<void>(gm.createMemoryGroup()), std::runtime_error);
    ASSERT_THROW(gm.addResourceToGroup(group, BufferId{1}), std::runtime_error);
    ASSERT_THROW(gm.finalize(), std::runtime_error);
}
