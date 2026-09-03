/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "context.hpp"
#include "data_manager.hpp"
#include "resource_data.hpp"
#include "scenario_options.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
using namespace mlsdk::scenariorunner;

Tensor &prepareTensor(Context &ctx, DataManager &dm, TensorId id, const std::vector<int64_t> &shape, vk::Format format,
                      uint64_t memoryOffset = 0) {
    TensorInfo info;
    info.debugName = "test_tensor";
    info.shape = shape;
    info.format = format;
    info.tiling = Tiling::Linear;
    info.memoryOffset = memoryOffset;
    dm.createTensor(id, info);
    auto &tensor = dm.getTensorMut(id);
    tensor.setup(ctx);
    tensor.allocateMemory(ctx);
    return tensor;
}

size_t bytesFor(vk::Format format, const std::vector<int64_t> &shape) {
    return static_cast<size_t>(elementSizeFromVkFormat(format) * totalElementsFromShape(shape));
}

std::vector<std::byte> sequence(size_t size, uint8_t first = 0) {
    std::vector<std::byte> result(size);
    std::generate(result.begin(), result.end(), [&first] { return std::byte{first++}; });
    return result;
}

TEST(TensorInMemoryTransfer, UploadThrowsOnShapeMismatch) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const std::vector<int64_t> shape{2, 2};
    const vk::Format fmt = vk::Format::eR8Uint;

    auto &tensor = prepareTensor(ctx, dm, TensorId{0}, shape, fmt);
    std::vector<std::byte> payload(bytesFor(fmt, shape), std::byte{0x3C});

    const std::vector<int64_t> wrongShape{2, 3};
    TensorDataView view{payload.data(), payload.size(), {}, std::nullopt};
    view.shape = wrongShape;
    view.format = fmt;
    EXPECT_THROW(tensor.upload(ctx, view), std::runtime_error);
}

TEST(TensorInMemoryTransfer, UploadThrowsOnIncompatibleFormatWhenProvided) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const std::vector<int64_t> shape{2, 2};
    const vk::Format fmt = vk::Format::eR8Uint;

    auto &tensor = prepareTensor(ctx, dm, TensorId{0}, shape, fmt);
    std::vector<std::byte> payload(bytesFor(fmt, shape), std::byte{0x7F});

    TensorDataView view{payload.data(), payload.size(), {}, std::nullopt};
    view.shape = shape;
    view.format = vk::Format::eR16Uint;
    EXPECT_THROW(tensor.upload(ctx, view), std::runtime_error);
}

TEST(TensorInMemoryTransfer, UploadThrowsOnSizeMismatch) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const std::vector<int64_t> shape{2, 2};
    const vk::Format fmt = vk::Format::eR8Uint;

    auto &tensor = prepareTensor(ctx, dm, TensorId{0}, shape, fmt);
    std::vector<std::byte> small(3, std::byte{0x11}); // too small by 1 byte

    TensorDataView view{small.data(), small.size(), {}, std::nullopt};
    view.shape = shape;
    EXPECT_THROW(tensor.upload(ctx, view), std::runtime_error);
}

TEST(TensorInMemoryTransfer, UploadSucceedsAndPersistsCopy_FormatOptional) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const std::vector<int64_t> shape{3, 2};
    const vk::Format fmt = vk::Format::eR8Uint; // 6 bytes

    auto &tensor = prepareTensor(ctx, dm, TensorId{0}, shape, fmt);
    auto payload = sequence(bytesFor(fmt, shape), 1);
    TensorDataView view{payload.data(), payload.size(), {}, std::nullopt};
    view.shape = shape;
    ASSERT_NO_THROW(tensor.upload(ctx, view));

    // Mutate source to ensure copy semantics
    std::fill(payload.begin(), payload.end(), std::byte{0xAA});

    const auto tensorData = tensor.download(ctx);
    ASSERT_EQ(tensorData.data.size(), payload.size());
    EXPECT_EQ(tensorData.shape, shape);
    ASSERT_TRUE(tensorData.format.has_value());
    EXPECT_EQ(tensorData.format.value(), fmt);
    for (size_t i = 0; i < tensorData.data.size(); ++i) {
        EXPECT_EQ(std::to_integer<unsigned char>(tensorData.data[i]), static_cast<unsigned char>(i + 1))
            << "mismatch at index " << i;
    }
}

TEST(TensorInMemoryTransfer, UploadAcceptsRankConvertedEmptyShape) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const std::vector<int64_t> emptyShape{};    // will be converted to [1]
    const vk::Format fmt = vk::Format::eR8Uint; // 1 byte

    auto &tensor = prepareTensor(ctx, dm, TensorId{0}, emptyShape, fmt);

    std::vector<std::byte> payload(1, std::byte{0x5A});
    TensorDataView view{payload.data(), payload.size(), {}, std::nullopt};
    view.shape = emptyShape;
    ASSERT_NO_THROW(tensor.upload(ctx, view));

    const auto tensorData = tensor.download(ctx);
    // Rank-converted tensors return empty shape in download metadata
    EXPECT_TRUE(tensorData.shape.empty());
    ASSERT_TRUE(tensorData.format.has_value());
    EXPECT_EQ(tensorData.format.value(), fmt);
    ASSERT_EQ(tensorData.data.size(), payload.size());
    EXPECT_EQ(tensorData.data[0], payload[0]);
}

TEST(TensorInMemoryTransfer, DownloadReturnsUploadedData) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const std::vector<int64_t> shape{2, 3, 1};
    const vk::Format fmt = vk::Format::eR8Uint; // 6 bytes

    auto &tensor = prepareTensor(ctx, dm, TensorId{0}, shape, fmt);
    auto payload = sequence(bytesFor(fmt, shape));
    TensorDataView view{payload.data(), payload.size(), {}, std::nullopt};
    view.shape = shape;
    view.format = fmt;
    tensor.upload(ctx, view);

    const auto tensorData = tensor.download(ctx);
    ASSERT_EQ(tensorData.data.size(), payload.size());
    EXPECT_EQ(tensorData.shape, shape);
    ASSERT_TRUE(tensorData.format.has_value());
    EXPECT_EQ(tensorData.format.value(), fmt);
    EXPECT_EQ(tensorData.data, payload);
}

TEST(TensorInMemoryTransfer, UploadAndDownloadRespectMemoryOffset) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const std::vector<int64_t> shape{2, 3, 1};
    const vk::Format format = vk::Format::eR8Uint;
    constexpr uint64_t memoryOffset = 4096;

    auto &tensor = prepareTensor(ctx, dm, TensorId{0}, shape, format, memoryOffset);
    auto payload = sequence(bytesFor(format, shape), 1);

    TensorDataView view{payload.data(), payload.size(), shape, format};
    ASSERT_NO_THROW(tensor.upload(ctx, view));

    const auto tensorData = tensor.download(ctx);
    EXPECT_EQ(tensorData.data, payload);
}

TEST(TensorInMemoryTransfer, UploadAndDownloadRespectImageSubresourceOffset) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const TensorId id{0};
    const std::vector<int64_t> shape{2, 3, 1};
    const vk::Format format = vk::Format::eR8Uint;
    constexpr vk::DeviceSize subresourceOffset = 4096;

    auto memoryManager = std::make_shared<ResourceMemoryManager>();
    // Image::setup records a linear image's VkSubresourceLayout::offset this way.
    memoryManager->updateSubResourceOffset(subresourceOffset);

    TensorInfo info;
    info.debugName = "test_tensor_image_subresource_offset";
    info.shape = shape;
    info.format = format;
    info.tiling = Tiling::Linear;
    dm.createTensor(id, info);

    auto &tensor = dm.getTensorMut(id);
    tensor.setup(ctx, memoryManager);
    tensor.allocateMemory(ctx);

    auto payload = sequence(bytesFor(format, shape), 1);
    ASSERT_NO_THROW(tensor.upload(ctx, {payload.data(), payload.size(), shape, format}));

    const auto tensorData = tensor.download(ctx);
    EXPECT_EQ(tensorData.data, payload);
}
