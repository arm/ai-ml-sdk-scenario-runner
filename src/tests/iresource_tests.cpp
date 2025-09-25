/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "iresource.hpp"

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

TEST(DataManagerResourceViewer, HasResources) {
    DataManager dm;

    {
        const Guid tensor("tensor");
        DataManagerResourceViewer viewer(dm, tensor);
        ASSERT_FALSE(viewer.hasBuffer());
        ASSERT_FALSE(viewer.hasImage());
        ASSERT_FALSE(viewer.hasTensor());
        ASSERT_THROW(viewer.getBuffer(), std::runtime_error);
        ASSERT_THROW(viewer.getImage(), std::runtime_error);
        ASSERT_THROW(viewer.getTensor(), std::runtime_error);
    }

    {
        const Guid buffer("buffer");
        dm.createBuffer(buffer, BufferInfo{});
        DataManagerResourceViewer viewer(dm, buffer);
        ASSERT_TRUE(viewer.hasBuffer());
        ASSERT_FALSE(viewer.hasImage());
        ASSERT_FALSE(viewer.hasTensor());
        ASSERT_NO_THROW(viewer.getBuffer());
        ASSERT_THROW(viewer.getImage(), std::runtime_error);
        ASSERT_THROW(viewer.getTensor(), std::runtime_error);
    }
}
