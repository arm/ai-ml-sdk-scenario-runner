/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "resource_data.hpp"
#include "scenario.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <numeric>
#include <vector>

#include <gtest/gtest.h>
using namespace mlsdk::scenariorunner;

Tensor &prepareTensor(Context &ctx, DataManager &dm, const Guid &guid, const std::vector<int64_t> &shape,
                      vk::Format format) {
    TensorInfo info;
    info.debugName = "test_tensor";
    info.shape = shape;
    info.format = format;
    info.tiling = Tiling::Linear;
    dm.createTensor(guid, info);
    auto &tensor = dm.getTensorMut(guid);
    tensor.setup(ctx);
    tensor.allocateMemory(ctx);
    return tensor;
}

size_t bytesFor(vk::Format format, const std::vector<int64_t> &shape) {
    return static_cast<size_t>(elementSizeFromVkFormat(format) * totalElementsFromShape(shape));
}

TEST(TensorInMemoryTransfer, UploadThrowsOnShapeMismatch) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const Guid guid("tensor_shape_mismatch");
    const std::vector<int64_t> shape{2, 2};
    const vk::Format fmt = vk::Format::eR8Uint;

    auto &tensor = prepareTensor(ctx, dm, guid, shape, fmt);
    std::vector<char> payload(bytesFor(fmt, shape), 0x3C);

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
    const Guid guid("tensor_format_mismatch");
    const std::vector<int64_t> shape{2, 2};
    const vk::Format fmt = vk::Format::eR8Uint;

    auto &tensor = prepareTensor(ctx, dm, guid, shape, fmt);
    std::vector<char> payload(bytesFor(fmt, shape), 0x7F);

    TensorDataView view{payload.data(), payload.size(), {}, std::nullopt};
    view.shape = shape;
    view.format = vk::Format::eR16Uint;
    EXPECT_THROW(tensor.upload(ctx, view), std::runtime_error);
}

TEST(TensorInMemoryTransfer, UploadThrowsOnSizeMismatch) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const Guid guid("tensor_size_mismatch");
    const std::vector<int64_t> shape{2, 2};
    const vk::Format fmt = vk::Format::eR8Uint;

    auto &tensor = prepareTensor(ctx, dm, guid, shape, fmt);
    std::vector<char> small(3, 0x11); // too small by 1 byte

    TensorDataView view{small.data(), small.size(), {}, std::nullopt};
    view.shape = shape;
    EXPECT_THROW(tensor.upload(ctx, view), std::runtime_error);
}

TEST(TensorInMemoryTransfer, UploadSucceedsAndPersistsCopy_FormatOptional) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const Guid guid("tensor_upload_ok");
    const std::vector<int64_t> shape{3, 2};
    const vk::Format fmt = vk::Format::eR8Uint; // 6 bytes

    auto &tensor = prepareTensor(ctx, dm, guid, shape, fmt);
    std::vector<char> payload(bytesFor(fmt, shape));
    std::iota(payload.begin(), payload.end(), static_cast<char>(1));
    TensorDataView view{payload.data(), payload.size(), {}, std::nullopt};
    view.shape = shape;
    ASSERT_NO_THROW(tensor.upload(ctx, view));

    // Mutate source to ensure copy semantics
    std::fill(payload.begin(), payload.end(), static_cast<char>(0xAA));

    const auto tensorData = tensor.download(ctx);
    ASSERT_EQ(tensorData.data.size(), payload.size());
    EXPECT_EQ(tensorData.shape, shape);
    ASSERT_TRUE(tensorData.format.has_value());
    EXPECT_EQ(tensorData.format.value(), fmt);
    for (size_t i = 0; i < tensorData.data.size(); ++i) {
        EXPECT_EQ(static_cast<unsigned char>(tensorData.data[i]), static_cast<unsigned char>(i + 1))
            << "mismatch at index " << i;
    }
}

TEST(TensorInMemoryTransfer, UploadAcceptsRankConvertedEmptyShape) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;
    const Guid guid("tensor_rank_converted");
    const std::vector<int64_t> emptyShape{};    // will be converted to [1]
    const vk::Format fmt = vk::Format::eR8Uint; // 1 byte

    auto &tensor = prepareTensor(ctx, dm, guid, emptyShape, fmt);

    std::vector<char> payload(1, static_cast<char>(0x5A));
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
    const Guid guid("tensor_download_ok");
    const std::vector<int64_t> shape{2, 3, 1};
    const vk::Format fmt = vk::Format::eR8Uint; // 6 bytes

    auto &tensor = prepareTensor(ctx, dm, guid, shape, fmt);
    std::vector<char> payload(bytesFor(fmt, shape));
    std::iota(payload.begin(), payload.end(), static_cast<char>(0));
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
