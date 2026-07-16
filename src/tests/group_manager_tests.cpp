/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_manager.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace mlsdk::scenariorunner;

TEST(GroupManager, GroupHandling) {
    const Guid group0("group0");
    const Guid tensor0("tensor0");
    const Guid image0("image0");
    GroupManager gm;
    gm.addResourceToGroup(group0, tensor0, ResourceIdType::Tensor);
    ASSERT_EQ(gm.getAliasCount(tensor0), 1U);
    ASSERT_FALSE(gm.isAliased(tensor0));
    ASSERT_EQ(gm.getAliasCount(image0), 0U);
    ASSERT_FALSE(gm.isAliased(image0));
    ASSERT_FALSE(gm.hasAliasOfType(tensor0, ResourceIdType::Image));
    ASSERT_FALSE(gm.hasAliasOfType(image0, ResourceIdType::Tensor));
    ASSERT_FALSE(gm.hasAliasOfType(group0, ResourceIdType::Tensor));

    gm.addResourceToGroup(group0, image0, ResourceIdType::Image);
    ASSERT_EQ(gm.getAliasCount(tensor0), 2U);
    ASSERT_TRUE(gm.isAliased(tensor0));
    ASSERT_EQ(gm.getAliasCount(image0), 2U);
    ASSERT_TRUE(gm.isAliased(image0));
    ASSERT_TRUE(gm.hasAliasOfType(tensor0, ResourceIdType::Image));
    ASSERT_TRUE(gm.hasAliasOfType(image0, ResourceIdType::Tensor));

    auto mmTensor = gm.getMemoryManager(tensor0);
    auto mmImage = gm.getMemoryManager(image0);
    ASSERT_EQ(mmTensor, mmImage);
    ASSERT_TRUE(mmTensor->isShared());
}

TEST(GroupManager, DuplicateResourceRegistrationToDifferentGroupThrows) {
    const Guid group0("group0");
    const Guid group1("group1");
    const Guid tensor0("tensor0");
    GroupManager gm;

    gm.addResourceToGroup(group0, tensor0, ResourceIdType::Tensor);

    ASSERT_THROW(gm.addResourceToGroup(group1, tensor0, ResourceIdType::Tensor), std::runtime_error);
}

TEST(GroupManager, GroupQueries) {
    const Guid group0("group0");
    const Guid tensor0("tensor0");
    const Guid image0("image0");
    GroupManager gm;

    gm.addResourceToGroup(group0, tensor0, ResourceIdType::Tensor);
    gm.addResourceToGroup(group0, image0, ResourceIdType::Image);

    const auto group = gm.getGroupForResource(tensor0);
    ASSERT_TRUE(group.has_value());
    ASSERT_EQ(*group, group0);

    const auto groups = gm.getGroups();
    ASSERT_EQ(groups.size(), 1U);
    ASSERT_NE(groups.find(group0), groups.end());

    const auto resources = gm.getResourcesInGroup(group0);
    ASSERT_EQ(resources.size(), 2U);
}
