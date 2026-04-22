/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "png_reader.hpp"

#include "stb_image_write.h"

#include <cstdint>
#include <filesystem>
#include <vector>

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

namespace {

std::filesystem::path makeTempPNGPath(const std::string &name) { return std::filesystem::temp_directory_path() / name; }

void writeTestPNG(const std::filesystem::path &filePath, uint32_t width, uint32_t height) {
    const int channels = 4;
    std::vector<uint8_t> pixelData(static_cast<size_t>(width) * static_cast<size_t>(height) * channels, 0x7f);
    const int stride = static_cast<int>(width) * channels;
    const int writeRes = stbi_write_png(filePath.string().c_str(), static_cast<int>(width), static_cast<int>(height),
                                        channels, pixelData.data(), stride);
    ASSERT_NE(writeRes, 0);
}

} // namespace

TEST(PngReader, LoadDataFromPNG) {
    const auto filePath = makeTempPNGPath("scenario_runner_png_reader_test.png");
    const uint32_t width = 4;
    const uint32_t height = 4;

    writeTestPNG(filePath, width, height);

    const auto result = loadDataFromPNG(filePath.string(), {});

    EXPECT_EQ(result.initialFormat, vk::Format::eR8G8B8A8Unorm);
    EXPECT_EQ(result.width, width);
    EXPECT_EQ(result.height, height);
    EXPECT_EQ(result.data.size(), static_cast<size_t>(width) * static_cast<size_t>(height) * 4);

    std::error_code ignored;
    std::filesystem::remove(filePath, ignored);
}

TEST(PngReader, ThrowsOnDimensionMismatch) {
    const auto filePath = makeTempPNGPath("scenario_runner_png_dimension_mismatch_test.png");
    const uint32_t width = 4;
    const uint32_t height = 4;

    writeTestPNG(filePath, width, height);

    ImageLoadOptions options{};
    options.expectedHeight = height + 1;
    options.expectedWidth = width;

    EXPECT_THROW((void)loadDataFromPNG(filePath.string(), options), std::runtime_error);

    std::error_code ignored;
    std::filesystem::remove(filePath, ignored);
}

TEST(PngReader, SaveDataToPNG) {
    const auto filePath = makeTempPNGPath("scenario_runner_png_writer_test.png");
    const uint32_t width = 4;
    const uint32_t height = 4;

    const auto data = std::vector<char>(static_cast<size_t>(width) * static_cast<size_t>(height) * 4, 0x7f);
    ImageSaveOptions options{{1, height, width, 4}, vk::Format::eR8G8B8A8Unorm, data};

    saveDataToPNG(filePath.string(), options);

    const auto result = loadDataFromPNG(filePath.string(), {});

    EXPECT_EQ(result.initialFormat, vk::Format::eR8G8B8A8Unorm);
    EXPECT_EQ(result.width, width);
    EXPECT_EQ(result.height, height);
    EXPECT_EQ(result.data.size(), static_cast<size_t>(width) * static_cast<size_t>(height) * 4);

    std::error_code ignored;
    std::filesystem::remove(filePath, ignored);
}
