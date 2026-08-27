/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "image.hpp"
#include "resource_data.hpp"
#include "scenario_desc.hpp"
#include "scenario_json_factory.hpp"
#include "scenario_options.hpp"
#include "utils.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

namespace {
size_t packedImageDataSize(const std::vector<int64_t> &shape, vk::Format format, uint32_t mipLevels = 1) {
    auto width = static_cast<size_t>(shape[1]);
    auto height = static_cast<size_t>(shape[2]);
    const auto depth = static_cast<size_t>(shape[3]);
    const auto elementSize = static_cast<size_t>(elementSizeFromVkFormat(format));
    size_t size = 0;
    for (uint32_t mip = 0; mip < mipLevels; ++mip) {
        size += width * height * depth * elementSize;
        width = std::max(width / 2, size_t{1});
        height = std::max(height / 2, size_t{1});
    }
    return size;
}

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

TEST(IScenario, ScenarioSupportsImageTransfers) {
    const std::string scenarioJson = R"(
        {
            "commands": [],
            "resources": [
                {
                    "image": {
                        "uid": "inImage",
                        "dims": [1, 2, 2, 1],
                        "format": "VK_FORMAT_R8_UINT",
                        "shader_access": "readwrite",
                        "mips": 1
                    }
                }
            ]
        }
    )";
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    auto api = ScenarioJsonFactory::make(ScenarioOptions{}, spec);
    const auto imageId = api->getImageId("inImage");
    const std::vector<std::byte> payload{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};

    api->upload(imageId, {payload.data(), payload.size(), {1, 2, 2, 1}, vk::Format::eR8Uint});
    api->run();

    EXPECT_EQ(api->download(imageId).data, payload);
}

TEST(ImageInMemoryTransfer, UploadValidatesMetadataAndSize) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dataManager;
    const ImageId imageId{0};
    const std::vector<int64_t> shape{1, 3, 2, 1};
    const vk::Format format = vk::Format::eR8Uint;

    auto &image = prepareImage(ctx, dataManager, imageId, shape, format);
    std::vector<std::byte> payload(packedImageDataSize(shape, format), std::byte{0x3c});

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
    std::vector<std::byte> firstPayload(packedImageDataSize(shape, format));
    uint8_t value = 1;
    std::generate(firstPayload.begin(), firstPayload.end(), [&value] { return std::byte{value++}; });

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

    std::vector<std::byte> secondPayload(packedImageDataSize(shape, format));
    uint8_t secondValue = 10;
    std::generate(secondPayload.rbegin(), secondPayload.rend(), [&secondValue] { return std::byte{secondValue++}; });

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
    std::vector<std::byte> payload(packedImageDataSize(shape, format, mipLevels));
    uint8_t value = 1;
    std::generate(payload.begin(), payload.end(), [&value] { return std::byte{value++}; });

    ASSERT_NO_THROW(image.upload(ctx, {payload.data(), payload.size(), shape, format, mipLevels}));
    const auto download = image.download(ctx);

    payload.resize(packedImageDataSize(shape, format));
    EXPECT_EQ(download.data, payload);
    EXPECT_EQ(download.shape, shape);
    EXPECT_EQ(download.mipLevels, 1);
    ASSERT_TRUE(download.format.has_value());
    EXPECT_EQ(download.format.value(), format);
}
