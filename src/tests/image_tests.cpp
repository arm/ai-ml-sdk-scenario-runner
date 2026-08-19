/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "image.hpp"
#include "resource_data.hpp"
#include "scenario.hpp"

#include <numeric>
#include <vector>

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

namespace {
Image &prepareImage(Context &ctx, DataManager &dataManager, ImageId id, const std::vector<int64_t> &shape,
                    vk::Format format, uint32_t mipLevels = 1) {
    ImageInfo info{};
    info.debugName = "test_image";
    info.shape = shape;
    info.format = format;
    info.targetFormat = format;
    info.isInput = true;
    info.isSampled = true;
    info.mips = mipLevels;
    info.tiling = mipLevels > 1 ? Tiling::Optimal : Tiling::Linear;

    dataManager.createImage(id, info);
    auto &image = dataManager.getImageMut(id);
    image.setup(ctx);
    image.allocateMemory(ctx);
    return image;
}
} // namespace

TEST(ImageInMemoryTransfer, UploadValidatesMetadataAndSize) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dataManager;
    const ImageId imageId{0};
    const std::vector<int64_t> shape{1, 3, 2, 1};
    const vk::Format format = vk::Format::eR8Uint;

    auto &image = prepareImage(ctx, dataManager, imageId, shape, format);
    std::vector<char> payload(image.baseDataSize(), 0x3c);

    EXPECT_THROW(image.upload(ctx, {payload.data(), payload.size(), {1, 2, 2, 1}, format}), std::runtime_error);
    EXPECT_THROW(image.upload(ctx, {payload.data(), payload.size(), shape, vk::Format::eR16Uint}), std::runtime_error);
    EXPECT_THROW(image.upload(ctx, {payload.data(), payload.size() - 1, shape, format}), std::runtime_error);
    EXPECT_THROW(image.upload(ctx, {payload.data(), payload.size(), shape, format, 0}), std::runtime_error);
    EXPECT_THROW(image.upload(ctx, {payload.data(), payload.size(), shape, format, 2}), std::runtime_error);
}

TEST(ImageInMemoryTransfer, CanUploadAndDownloadRepeatedly) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dataManager;
    const ImageId imageId{0};
    const std::vector<int64_t> shape{1, 3, 2, 1};
    const vk::Format format = vk::Format::eR8Uint;

    auto &image = prepareImage(ctx, dataManager, imageId, shape, format);
    std::vector<char> firstPayload(image.baseDataSize());
    std::iota(firstPayload.begin(), firstPayload.end(), static_cast<char>(1));

    image.transitionLayout(ctx, vk::ImageLayout::eShaderReadOnlyOptimal);
    ASSERT_NO_THROW(image.upload(ctx, {firstPayload.data(), firstPayload.size(), shape, format}));
    EXPECT_EQ(image.getImageLayout(), vk::ImageLayout::eShaderReadOnlyOptimal);

    image.transitionLayout(ctx, vk::ImageLayout::eShaderReadOnlyOptimal);
    const auto firstDownload = image.download(ctx);
    EXPECT_EQ(firstDownload.data, firstPayload);
    EXPECT_EQ(firstDownload.shape, shape);
    EXPECT_EQ(firstDownload.mipLevels, 1);
    EXPECT_EQ(image.getImageLayout(), vk::ImageLayout::eShaderReadOnlyOptimal);
    ASSERT_TRUE(firstDownload.format.has_value());
    EXPECT_EQ(firstDownload.format.value(), format);

    std::vector<char> secondPayload(image.baseDataSize());
    std::iota(secondPayload.rbegin(), secondPayload.rend(), static_cast<char>(10));

    ASSERT_NO_THROW(image.upload(ctx, {secondPayload.data(), secondPayload.size(), shape, std::nullopt}));
    const auto secondDownload = image.download(ctx);
    EXPECT_EQ(secondDownload.data, secondPayload);
    EXPECT_EQ(secondDownload.shape, shape);
    EXPECT_EQ(secondDownload.mipLevels, 1);
    ASSERT_TRUE(secondDownload.format.has_value());
    EXPECT_EQ(secondDownload.format.value(), format);
}

TEST(ImageInMemoryTransfer, CanUploadCompleteMipChainAndDownloadBaseMip) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dataManager;
    const ImageId imageId{0};
    const std::vector<int64_t> shape{1, 4, 4, 1};
    const vk::Format format = vk::Format::eR8Uint;
    constexpr uint32_t mipLevels = 3;

    auto &image = prepareImage(ctx, dataManager, imageId, shape, format, mipLevels);
    std::vector<char> payload(image.mipChainDataSize(mipLevels));
    std::iota(payload.begin(), payload.end(), static_cast<char>(1));

    ASSERT_NO_THROW(image.upload(ctx, {payload.data(), payload.size(), shape, format, mipLevels}));
    const auto download = image.download(ctx);

    payload.resize(image.baseDataSize());
    EXPECT_EQ(download.data, payload);
    EXPECT_EQ(download.shape, shape);
    EXPECT_EQ(download.mipLevels, 1);
    ASSERT_TRUE(download.format.has_value());
    EXPECT_EQ(download.format.value(), format);
}
