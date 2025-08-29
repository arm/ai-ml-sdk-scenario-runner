/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_manager.hpp"

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

TEST(GroupManager, GroupHandling) {
    const Guid group0("group0");
    const Guid tensor0("tensor0");
    const Guid image0("image0");
    GroupManager gm;
    gm.addResourceToGroup(group0, tensor0, ResourceIdType::Tensor);
    ASSERT_EQ(gm.getAliasCount(tensor0), 1U);
    ASSERT_TRUE(gm.isAliased(tensor0));
    ASSERT_EQ(gm.getAliasCount(image0), 0U);
    ASSERT_FALSE(gm.isAliased(image0));
    ASSERT_FALSE(gm.isAliasedTo(tensor0, ResourceIdType::Image));
    ASSERT_FALSE(gm.isAliasedTo(image0, ResourceIdType::Tensor));
    ASSERT_FALSE(gm.isAliasedTo(group0, ResourceIdType::Tensor));

    gm.addResourceToGroup(group0, image0, ResourceIdType::Image);
    ASSERT_EQ(gm.getAliasCount(tensor0), 2U);
    ASSERT_TRUE(gm.isAliased(tensor0));
    ASSERT_EQ(gm.getAliasCount(image0), 2U);
    ASSERT_TRUE(gm.isAliased(image0));
    ASSERT_TRUE(gm.isAliasedTo(tensor0, ResourceIdType::Image));
    ASSERT_TRUE(gm.isAliasedTo(image0, ResourceIdType::Tensor));

    auto mmTensor = gm.getMemoryManager(tensor0);
    auto mmImage = gm.getMemoryManager(image0);
    ASSERT_EQ(mmTensor, mmImage);
}
