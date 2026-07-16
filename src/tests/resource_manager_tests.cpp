/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resource_manager.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace mlsdk::scenariorunner;

static_assert(!std::is_same_v<BufferId, TensorId>);
static_assert(!std::is_convertible_v<BufferId, TensorId>);

TEST(ResourceManager, AssignsIdsIndependentlyPerType) {
    ResourceManager resources;

    BufferInfo bufferA{};
    bufferA.debugName = "buffer_a";
    ImageInfo image{};
    image.debugName = "image";
    BufferInfo bufferB{};
    bufferB.debugName = "buffer_b";

    const auto bufferAId = resources.addBuffer(std::move(bufferA));
    const auto imageId = resources.addImage(std::move(image));
    const auto bufferBId = resources.addBuffer(std::move(bufferB));

    EXPECT_EQ(bufferAId.value(), 0U);
    EXPECT_EQ(imageId.value(), 0U);
    EXPECT_EQ(bufferBId.value(), 1U);
}

TEST(ResourceManager, PreservesResourceInfo) {
    ResourceManager resources;

    ShaderInfo shader{};
    shader.debugName = "shader";
    shader.entry = "main";
    shader.pushConstantsSize = 16;

    const auto shaderId = resources.addShader(std::move(shader));

    EXPECT_EQ(resources.get(shaderId).debugName, "shader");
    EXPECT_EQ(resources.get(shaderId).entry, "main");
    EXPECT_EQ(resources.get(shaderId).pushConstantsSize, 16U);
}

TEST(ResourceManager, RejectsOutOfRangeIds) {
    const ResourceManager resources;

    EXPECT_THROW(static_cast<void>(resources.get(BufferId{0})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(resources.get(TensorId{0})), std::out_of_range);
}
