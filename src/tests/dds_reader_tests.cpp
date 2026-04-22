/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dds_reader.hpp"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

namespace {

std::filesystem::path makeTempDDSPath(const std::string &name) { return std::filesystem::temp_directory_path() / name; }

uint32_t mipChainSizeBytes(uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t elementSize) {
    uint32_t size = 0;
    for (uint32_t mip = 0; mip < mipLevels; ++mip) {
        size += width * height * elementSize;
        width = std::max(width / 2u, 1u);
        height = std::max(height / 2u, 1u);
    }
    return size;
}

void writeTestDDS(const std::filesystem::path &filePath, uint32_t width, uint32_t height, uint32_t mipLevels,
                  uint32_t elementSize) {
    DDSHeaderInfo header = generateDefaultDDSHeader(height, width, elementSize, DXGI_FORMAT_R8G8B8A8_UNORM);
    header.header.mipMapCount = mipLevels;

    std::ofstream file(filePath, std::ofstream::binary);
    ASSERT_TRUE(file.is_open());
    saveHeaderToDDS(header, file);

    const auto dataSize = mipChainSizeBytes(width, height, mipLevels, elementSize);
    std::vector<char> pixelData(dataSize, 0);
    file.write(pixelData.data(), static_cast<std::streamsize>(pixelData.size()));
}

} // namespace

TEST(DdsReader, loadDataFromDDS) {
    const auto filePath = makeTempDDSPath("scenario_runner_dds_test_mips.dds");
    const uint32_t width = 4;
    const uint32_t height = 4;
    const uint32_t mipLevels = 3;
    const uint32_t elementSize = 4;

    writeTestDDS(filePath, width, height, mipLevels, elementSize);

    const auto result = loadDataFromDDS(filePath.string(), {});

    EXPECT_EQ(result.initialFormat, vk::Format::eR8G8B8A8Unorm);
    EXPECT_EQ(result.width, width);
    EXPECT_EQ(result.height, height);
    EXPECT_EQ(result.mipLevels, mipLevels);

    std::error_code ignored;
    std::filesystem::remove(filePath, ignored);
}

TEST(DdsReader, ThrowsOnDimensionMismatch) {
    const auto filePath = makeTempDDSPath("scenario_runner_dds_test_dimension_mismatch.dds");
    const uint32_t width = 4;
    const uint32_t height = 4;
    const uint32_t mipLevels = 1;
    const uint32_t elementSize = 4;

    writeTestDDS(filePath, width, height, mipLevels, elementSize);

    ImageLoadOptions options{};
    options.expectedHeight = height + 1;
    options.expectedWidth = width;

    EXPECT_THROW((void)loadDataFromDDS(filePath.string(), options), std::runtime_error);

    std::error_code ignored;
    std::filesystem::remove(filePath, ignored);
}

TEST(DdsReader, SaveDataToDDS) {
    const auto filePath = makeTempDDSPath("scenario_runner_dds_writer_test.dds");
    const uint32_t width = 4;
    const uint32_t height = 4;
    const uint32_t mipLevels = 1;
    const uint32_t elementSize = 4;

    const auto data = std::vector<char>(static_cast<size_t>(width) * static_cast<size_t>(height) * elementSize, 0);
    ImageSaveOptions options{{1, height, width, 4}, vk::Format::eR8G8B8A8Unorm, data};

    saveDataToDDS(filePath.string(), options);

    const auto result = loadDataFromDDS(filePath.string(), {});

    EXPECT_EQ(result.initialFormat, vk::Format::eR8G8B8A8Unorm);
    EXPECT_EQ(result.width, width);
    EXPECT_EQ(result.height, height);
    EXPECT_EQ(result.mipLevels, mipLevels);

    std::error_code ignored;
    std::filesystem::remove(filePath, ignored);
}
