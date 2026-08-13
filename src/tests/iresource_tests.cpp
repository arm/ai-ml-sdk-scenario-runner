/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "iresource.hpp"

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

TEST(DataManager, MissingTypedResourceThrowsRuntimeError) {
    DataManager dataManager;
    const auto &constDataManager = dataManager;

    EXPECT_THROW(dataManager.getBufferMut(BufferId{0}), std::runtime_error);
    EXPECT_THROW(dataManager.getImageMut(ImageId{0}), std::runtime_error);
    EXPECT_THROW(dataManager.getTensorMut(TensorId{0}), std::runtime_error);
    EXPECT_THROW(constDataManager.getBuffer(BufferId{0}), std::runtime_error);
    EXPECT_THROW(constDataManager.getImage(ImageId{0}), std::runtime_error);
    EXPECT_THROW(constDataManager.getTensor(TensorId{0}), std::runtime_error);
}

TEST(DataManagerResourceViewer, HasResources) {
    DataManager dm;

    {
        const TensorId tensor{0};
        DataManagerResourceViewer viewer(dm, tensor);
        ASSERT_FALSE(viewer.hasBuffer());
        ASSERT_FALSE(viewer.hasImage());
        ASSERT_FALSE(viewer.hasTensor());
        ASSERT_THROW(viewer.getBuffer(), std::runtime_error);
        ASSERT_THROW(viewer.getImage(), std::runtime_error);
        ASSERT_THROW(viewer.getTensor(), std::runtime_error);
    }

    {
        const BufferId buffer{0};
        dm.createBuffer(buffer, BufferInfo{});
        DataManagerResourceViewer viewer(dm, buffer);
        ASSERT_TRUE(viewer.hasBuffer());
        ASSERT_FALSE(viewer.hasImage());
        ASSERT_FALSE(viewer.hasTensor());
        ASSERT_NO_THROW(viewer.getBuffer());
        ASSERT_THROW(viewer.getImage(), std::runtime_error);
        ASSERT_THROW(viewer.getTensor(), std::runtime_error);
    }

    {
        const ImageId image{0};
        dm.createImage(image, ImageInfo{});
        DataManagerResourceViewer viewer(dm, image);
        ASSERT_FALSE(viewer.hasBuffer());
        ASSERT_TRUE(viewer.hasImage());
        ASSERT_FALSE(viewer.hasTensor());
        ASSERT_THROW(viewer.getBuffer(), std::runtime_error);
        ASSERT_NO_THROW(viewer.getImage());
        ASSERT_THROW(viewer.getTensor(), std::runtime_error);
    }
}
