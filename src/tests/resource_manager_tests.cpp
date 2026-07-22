/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resource_manager.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

using namespace mlsdk::scenariorunner;

static_assert(!std::is_same_v<BufferId, TensorId>);
static_assert(!std::is_convertible_v<BufferId, TensorId>);

TEST(ResourceId, SupportsUnorderedContainers) {
    std::unordered_map<BufferId, std::string> buffers;
    buffers.emplace(BufferId{3}, "buffer");

    EXPECT_EQ(buffers.at(BufferId{3}), "buffer");
}

TEST(ResourceManager, AssignsIdsIndependentlyPerType) {
    ResourceManager resources;

    BufferInfo bufferA{};
    bufferA.debugName = "buffer_a";
    ImageInfo image{};
    image.debugName = "image";
    BufferInfo bufferB{};
    bufferB.debugName = "buffer_b";
    RawDataInfo rawData{"raw_data", "data.npy"};
    DataGraphInfo dataGraph{"data_graph", "graph.vgf", 0, {}};
    GraphConstantInfo graphConstant{"constant", vk::Format::eR32Sfloat, {1}};

    const auto bufferAId = resources.addBuffer(std::move(bufferA));
    const auto imageId = resources.addImage(std::move(image));
    const auto bufferBId = resources.addBuffer(std::move(bufferB));
    const auto rawDataId = resources.addRawData(std::move(rawData));
    const auto dataGraphId = resources.addDataGraph(std::move(dataGraph));
    const auto graphConstantId = resources.addGraphConstant(std::move(graphConstant));

    EXPECT_EQ(bufferAId.value(), 0U);
    EXPECT_EQ(imageId.value(), 0U);
    EXPECT_EQ(bufferBId.value(), 1U);
    EXPECT_EQ(rawDataId.value(), 0U);
    EXPECT_EQ(dataGraphId.value(), 0U);
    EXPECT_EQ(graphConstantId.value(), 0U);
}

TEST(ResourceManager, PreservesResourceInfo) {
    ResourceManager resources;

    ShaderInfo shader{};
    shader.debugName = "shader";
    shader.entry = "main";
    shader.pushConstantsSize = 16;

    const auto shaderId = resources.addShader(shader);

    EXPECT_EQ(shader.debugName, "shader");
    EXPECT_EQ(resources.get(shaderId).debugName, "shader");
    EXPECT_EQ(resources.get(shaderId).entry, "main");
    EXPECT_EQ(resources.get(shaderId).pushConstantsSize, 16U);

    const auto rawDataId = resources.addRawData({"raw_data", "data.npy"});
    const auto dataGraphId = resources.addDataGraph({"data_graph", "graph.vgf", 0, {}});
    GraphConstantInfo graphConstant{"constant", vk::Format::eR32Sint, {2}};
    graphConstant.data = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto graphConstantId = resources.addGraphConstant(std::move(graphConstant));

    EXPECT_EQ(resources.get(rawDataId).debugName, "raw_data");
    EXPECT_EQ(resources.get(rawDataId).src, "data.npy");
    EXPECT_EQ(resources.get(dataGraphId).debugName, "data_graph");
    EXPECT_EQ(resources.get(dataGraphId).src, "graph.vgf");
    EXPECT_EQ(resources.get(graphConstantId).debugName, "constant");
    EXPECT_EQ(resources.get(graphConstantId).format, vk::Format::eR32Sint);
    EXPECT_EQ(resources.get(graphConstantId).dims, std::vector<int64_t>({2}));
    EXPECT_EQ(resources.get(graphConstantId).data.size(), 8U);
}

TEST(ResourceManager, RejectsOutOfRangeIds) {
    const ResourceManager resources;

    EXPECT_THROW(static_cast<void>(resources.get(BufferId{0})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(resources.get(TensorId{0})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(resources.get(RawDataId{0})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(resources.get(DataGraphId{0})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(resources.get(GraphConstantResourceId{0})), std::out_of_range);
}
