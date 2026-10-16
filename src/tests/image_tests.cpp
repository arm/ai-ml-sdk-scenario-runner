/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/resource_data.hpp"
#include "scenario_runner/scenario_json_factory.hpp"
#include "scenario_runner/scenario_options.hpp"

#include "data_manager.hpp"
#include "dds_reader.hpp"
#include "device_selection.hpp"
#include "image.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "vgf-utils/temp_folder.hpp"

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
                    vk::Format format, uint32_t mipLevels = 1,
                    std::shared_ptr<ResourceMemoryManager> memoryManager = std::make_shared<ResourceMemoryManager>()) {
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
    image.setup(ctx, std::move(memoryManager));
    image.allocateMemory(ctx);
    return image;
}

// The public download operation intentionally exposes only the base level.
// Read back every subresource here so mip initialization/generation cannot pass
// merely because the base level was copied correctly.
std::vector<std::byte> downloadMipChain(const Context &ctx, Image &image, ResourceMemoryManager &memoryManager) {
    const auto originalLayout = image.getImageLayout();
    image.transitionLayout(ctx, vk::ImageLayout::eGeneral);
    const auto size = packedImageDataSize(image.shape(), image.dataType(), image.getInfo().mips);
    std::memset(memoryManager.mapStagingBufferMemory(0, size), 0xcd, size);
    memoryManager.unmapStagingBufferMemory();

    auto pool = ctx.device().createCommandPool({{}, ctx.familyQueueIdx()});
    auto commands = ctx.device().allocateCommandBuffers({*pool, vk::CommandBufferLevel::ePrimary, 1});
    auto &command = commands.front();
    command.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    std::vector<vk::BufferImageCopy> regions;
    auto width = static_cast<uint32_t>(image.shape()[1]);
    auto height = static_cast<uint32_t>(image.shape()[2]);
    vk::DeviceSize offset = 0;
    for (uint32_t mip = 0; mip < image.getInfo().mips; ++mip) {
        vk::BufferImageCopy region{};
        region.bufferOffset = offset;
        region.imageSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, 1};
        region.imageExtent = vk::Extent3D{width, height, 1};
        regions.push_back(region);
        offset += static_cast<vk::DeviceSize>(width) * height * elementSizeFromVkFormat(image.dataType());
        width = std::max(width / 2, 1U);
        height = std::max(height / 2, 1U);
    }
    command.copyImageToBuffer(image.image(), vk::ImageLayout::eGeneral, *memoryManager.getStagingBuffer(), regions);
    const vk::MemoryBarrier2 hostRead{vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                      vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead};
    command.pipelineBarrier2(vk::DependencyInfo({}, hostRead));
    command.end();
    auto fence = ctx.device().createFence({});
    ctx.device().getQueue(ctx.familyQueueIdx(), 0).submit(vk::SubmitInfo({}, {}, *command), *fence);
    if (ctx.device().waitForFences({*fence}, true, WAIT_FOR_FENCE_TIMEOUT) != vk::Result::eSuccess) {
        throw std::runtime_error("Timed out reading image mip chain");
    }
    std::vector<std::byte> data(size);
    std::memcpy(data.data(), memoryManager.mapStagingBufferMemory(0, size), size);
    memoryManager.unmapStagingBufferMemory();
    image.transitionLayout(ctx, originalLayout);
    return data;
}

constexpr auto packedFloatFormat = vk::Format::eB10G11R11UfloatPack32;
constexpr auto imageTransferFeatures = vk::FormatFeatureFlagBits::eSampledImage |
                                       vk::FormatFeatureFlagBits::eTransferSrc |
                                       vk::FormatFeatureFlagBits::eTransferDst;
constexpr auto mipGenerationFeatures = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst |
                                       vk::FormatFeatureFlagBits::eSampledImageFilterLinear;

class ImageAutomaticTiling : public testing::TestWithParam<bool> {};
} // namespace

TEST(Scenario, ScenarioSupportsImageTransfers) {
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
    TempFolder tempFolder("scenario_image_transfer_test");
    const auto scenarioPath = tempFolder.relative("scenario.json");
    std::ofstream(scenarioPath) << scenarioJson;
    auto api = ScenarioJsonFactory::make(scenarioPath);
    const auto imageId = api->getImageId("inImage");
    const std::vector<std::byte> payload{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};

    api->upload(imageId, {payload.data(), payload.size(), {1, 2, 2, 1}, vk::Format::eR8Uint});
    api->run();

    EXPECT_EQ(api->download(imageId).data, payload);
}

TEST(ImageInMemoryTransfer, UploadValidatesMetadataAndSize) {
    ScenarioOptions opts{};
    Context ctx{opts, detail::DeviceRequirements{}};
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
    Context ctx{opts, detail::DeviceRequirements{}};
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
    Context ctx{opts, detail::DeviceRequirements{}};
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

TEST(Scenario, InitializesEveryPackedFloatMipWithoutBlitting) {
    ScenarioOptions opts{};
    Context ctx{opts, detail::DeviceRequirements{}};
    const auto features = ctx.physicalDevice().getFormatProperties(packedFloatFormat).optimalTilingFeatures;
    if ((features & imageTransferFeatures) != imageTransferFeatures) {
        GTEST_SKIP() << "Packed float sampled image transfers unavailable";
    }

    // Inspect every texel rather than just the base mip exposed by download().
    const std::string shaderSource = R"(
        #version 450
        layout(local_size_x = 1) in;
        layout(set = 0, binding = 0) uniform sampler2D inputImage;
        layout(set = 0, binding = 1) writeonly buffer Results { uint nonzero[]; } results;
        void main() {
            int mip = int(gl_GlobalInvocationID.x);
            ivec2 size = textureSize(inputImage, mip);
            uint count = 0;
            for (int y = 0; y < size.y; ++y) {
                for (int x = 0; x < size.x; ++x) {
                    if (any(notEqual(texelFetch(inputImage, ivec2(x, y), mip).rgb, vec3(0.0)))) {
                        ++count;
                    }
                }
            }
            results.nonzero[mip] = count;
        }
    )";
    TempFolder tempFolder("scenario_empty_mip_chain");
    std::ofstream(tempFolder.relative("inspect.comp")) << shaderSource;
    const auto scenarioPath = tempFolder.relative("scenario.json");
    // JSON initialization is intentional: plain ScenarioBuilder images do not
    // receive the source-free zero initialization used by file-based scenarios.
    std::ofstream(scenarioPath) << R"({
        "resources": [
            {"image": {"uid": "empty_packed_float", "dims": [1, 32, 32, 1],
                       "format": "VK_FORMAT_B10G11R11_UFLOAT_PACK32", "mips": 6,
                       "shader_access": "readonly"}},
            {"buffer": {"uid": "mip_nonzero_counts", "size": 24, "shader_access": "writeonly"}},
            {"shader": {"uid": "inspect", "src": "inspect.comp", "type": "GLSL"}}
        ],
        "commands": [{"dispatch_compute": {
            "shader_ref": "inspect", "rangeND": [6, 1, 1],
            "bindings": [{"id": 0, "set": 0, "resource_ref": "empty_packed_float"},
                         {"id": 1, "set": 0, "resource_ref": "mip_nonzero_counts"}]
        }}]
    })";
    auto scenario = ScenarioJsonFactory::make(scenarioPath);
    const auto resultId = scenario->getBufferId("mip_nonzero_counts");
    // A sentinel ensures a missing dispatch or inaccessible mip cannot look like
    // successful initialization simply because the result buffer starts at zero.
    const std::vector<std::byte> sentinel(6 * sizeof(uint32_t), std::byte{0xff});
    scenario->upload(resultId, {sentinel.data(), sentinel.size()});
    scenario->run();
    EXPECT_EQ(scenario->download(resultId).data, std::vector<std::byte>(sentinel.size(), std::byte{0}));
}

TEST(ImageInMemoryTransfer, CompleteMipChainDoesNotRequireBlitSupport) {
    ScenarioOptions opts{};
    Context ctx{opts, detail::DeviceRequirements{}};
    const auto features = ctx.physicalDevice().getFormatProperties(packedFloatFormat).optimalTilingFeatures;
    if ((features & imageTransferFeatures) != imageTransferFeatures) {
        GTEST_SKIP() << "Packed float sampled image transfers unavailable";
    }
    if ((features & mipGenerationFeatures) == mipGenerationFeatures) {
        GTEST_SKIP() << "Device supports packed float mip generation; cannot exercise unsupported generation";
    }

    DataManager dataManager;
    auto memoryManager = std::make_shared<ResourceMemoryManager>();
    const std::vector<int64_t> shape{1, 32, 32, 1};
    auto &image = prepareImage(ctx, dataManager, ImageId{0}, shape, packedFloatFormat, 6, memoryManager);
    std::vector<std::byte> payload(packedImageDataSize(shape, packedFloatFormat, 6));
    // Different bytes in successive mips detect missing copies and wrong offsets.
    uint8_t value = 0;
    std::generate(payload.begin(), payload.end(), [&value] { return std::byte{value++}; });
    ASSERT_NO_THROW(image.upload(ctx, {payload.data(), payload.size(), shape, packedFloatFormat, 6}));
    EXPECT_EQ(downloadMipChain(ctx, image, *memoryManager), payload);

    const auto previousLayout = image.getImageLayout();
    const auto baseSize = packedImageDataSize(shape, packedFloatFormat);
    try {
        image.upload(ctx, {payload.data(), baseSize, shape, packedFloatFormat});
        FAIL() << "Incomplete chain must reject unsupported mip generation";
    } catch (const std::runtime_error &error) {
        const std::string diagnostic = error.what();
        EXPECT_NE(diagnostic.find("test_image"), std::string::npos) << diagnostic;
        EXPECT_NE(diagnostic.find("B10G11R11"), std::string::npos) << diagnostic;
        EXPECT_NE(diagnostic.find("missing"), std::string::npos) << diagnostic;
        if (!(features & vk::FormatFeatureFlagBits::eBlitDst)) {
            EXPECT_NE(diagnostic.find("BlitDst"), std::string::npos) << diagnostic;
        }
    }
    EXPECT_EQ(image.getImageLayout(), previousLayout);
    EXPECT_EQ(downloadMipChain(ctx, image, *memoryManager), payload);
}

TEST(Scenario, SelectsMipGenerationRequirementsFromSourceMipCount) {
    ScenarioOptions opts{};
    Context ctx{opts, detail::DeviceRequirements{}};
    const auto props = ctx.physicalDevice().getFormatProperties(packedFloatFormat);
    if ((props.optimalTilingFeatures & imageTransferFeatures) != imageTransferFeatures) {
        GTEST_SKIP() << "Packed float sampled image transfers unavailable";
    }
    if ((props.optimalTilingFeatures & mipGenerationFeatures) == mipGenerationFeatures ||
        (props.linearTilingFeatures & mipGenerationFeatures) == mipGenerationFeatures) {
        GTEST_SKIP() << "Device can generate packed float mips; cannot exercise unsupported source generation";
    }

    TempFolder tempFolder("scenario_source_mip_chain");
    const auto sourcePath = tempFolder.relative("input.dds");
    const auto scenarioPath = tempFolder.relative("scenario.json");
    std::ofstream(scenarioPath) << R"({
        "resources": [{"image": {
            "uid": "source_packed_float", "dims": [1, 32, 32, 1],
            "format": "VK_FORMAT_B10G11R11_UFLOAT_PACK32", "mips": 6,
            "src": "input.dds", "shader_access": "readonly"
        }}],
        "commands": []
    })";
    const std::vector<int64_t> shape{1, 32, 32, 1};
    const std::vector<std::byte> source(packedImageDataSize(shape, packedFloatFormat, 6), std::byte{0x3c});
    auto header = generateDefaultDDSHeader(32, 32, 4, DXGI_FORMAT_R11G11B10_FLOAT);
    header.header.mipMapCount = 6;
    {
        std::ofstream file(sourcePath, std::ios::binary);
        saveHeaderToDDS(header, file);
        file.write(reinterpret_cast<const char *>(source.data()), static_cast<std::streamsize>(source.size()));
    }
    // A complete source must not add blit requirements during automatic tiling
    // selection or ask upload() to generate any levels.
    auto scenario = ScenarioJsonFactory::make(scenarioPath);
    EXPECT_EQ(scenario->download(scenario->getImageId("source_packed_float")).data,
              std::vector<std::byte>(packedImageDataSize(shape, packedFloatFormat), std::byte{0x3c}));
    scenario.reset();

    // Replacing only the DDS with a base-level source must inform image setup
    // that initialization will generate mips. Rejection here (rather than the
    // later upload guard) protects the source-to-tiling-selection wiring.
    const std::vector<std::byte> base(
        source.begin(), source.begin() + static_cast<std::ptrdiff_t>(packedImageDataSize(shape, packedFloatFormat)));
    saveDataToDDS(sourcePath.string(), ImageSaveOptions{shape, packedFloatFormat, base});
    try {
        auto incomplete = ScenarioJsonFactory::make(scenarioPath);
        FAIL() << "Incomplete source must reject unsupported mip generation during setup";
    } catch (const std::runtime_error &error) {
        const std::string diagnostic = error.what();
        EXPECT_NE(diagnostic.find("source_packed_float"), std::string::npos) << diagnostic;
        EXPECT_NE(diagnostic.find("No supported tiling"), std::string::npos) << diagnostic;
        EXPECT_NE(diagnostic.find("missing"), std::string::npos) << diagnostic;
    }
}

TEST_P(ImageAutomaticTiling, GeneratesMissingMipsWithAutomaticTiling) {
    ScenarioOptions opts{};
    Context ctx{opts, detail::DeviceRequirements{}};
    constexpr auto format = vk::Format::eR8G8B8A8Unorm;
    const auto props = ctx.physicalDevice().getFormatProperties(format);
    const auto required = imageTransferFeatures | mipGenerationFeatures;
    if ((props.optimalTilingFeatures & required) != required) {
        GTEST_SKIP() << "RGBA8 sampled image mip generation unavailable";
    }

    ImageInfo info{};
    info.debugName = "generated_mips";
    info.shape = {1, 8, 8, 1};
    info.format = format;
    info.targetFormat = format;
    info.isInput = true;
    info.isSampled = true;
    info.mips = 4;
    Image image{info};
    auto memoryManager = std::make_shared<ResourceMemoryManager>();
    image.setup(ctx, memoryManager, GetParam());
    image.allocateMemory(ctx);
    const auto selectedFeatures =
        image.tiling() == vk::ImageTiling::eLinear ? props.linearTilingFeatures : props.optimalTilingFeatures;
    EXPECT_EQ(selectedFeatures & required, required);

    // A complete initial upload must preserve generation support for later uploads,
    // even when setup did not know that mip generation would be needed.
    const std::vector<std::byte> complete(image.dataSize(), std::byte{0x33});
    ASSERT_NO_THROW(image.upload(ctx, {complete.data(), complete.size(), info.shape, format, info.mips}));

    // A constant normalized value survives linear filtering exactly at every mip.
    const std::vector<std::byte> base(packedImageDataSize(info.shape, format), std::byte{0x77});
    ASSERT_NO_THROW(image.upload(ctx, {base.data(), base.size(), info.shape, format}));
    EXPECT_EQ(downloadMipChain(ctx, image, *memoryManager),
              std::vector<std::byte>(packedImageDataSize(info.shape, format, info.mips), std::byte{0x77}));

    // When several levels are supplied, generation must start from the last
    // supplied level while leaving earlier levels intact.
    std::vector<std::byte> partial(packedImageDataSize(info.shape, format, 2), std::byte{0x55});
    std::copy(base.begin(), base.end(), partial.begin());
    ASSERT_NO_THROW(image.upload(ctx, {partial.data(), partial.size(), info.shape, format, 2}));
    auto expected = std::vector<std::byte>(packedImageDataSize(info.shape, format, info.mips), std::byte{0x55});
    std::copy(base.begin(), base.end(), expected.begin());
    EXPECT_EQ(downloadMipChain(ctx, image, *memoryManager), expected);
}

INSTANTIATE_TEST_SUITE_P(InitialOrLaterMipGeneration, ImageAutomaticTiling, testing::Bool());

TEST(Scenario, GeneratesMissingMipsAfterZeroOrCompleteInitialization) {
    ScenarioOptions opts{};
    Context ctx{opts, detail::DeviceRequirements{}};
    constexpr auto format = vk::Format::eR8G8B8A8Unorm;
    const auto features = ctx.physicalDevice().getFormatProperties(format).optimalTilingFeatures;
    const auto required = imageTransferFeatures | mipGenerationFeatures;
    if ((features & required) != required) {
        GTEST_SKIP() << "RGBA8 sampled image mip generation unavailable";
    }

    TempFolder tempFolder("scenario_later_mip_generation");
    const std::vector<int64_t> shape{1, 8, 8, 1};
    const std::vector<std::byte> complete(packedImageDataSize(shape, format, 4), std::byte{0x33});
    auto header = generateDefaultDDSHeader(8, 8, 4, DXGI_FORMAT_R8G8B8A8_UNORM);
    header.header.mipMapCount = 4;
    {
        std::ofstream file(tempFolder.relative("input.dds"), std::ios::binary);
        saveHeaderToDDS(header, file);
        file.write(reinterpret_cast<const char *>(complete.data()), static_cast<std::streamsize>(complete.size()));
    }

    for (const auto hasSource : {false, true}) {
        SCOPED_TRACE(hasSource ? "complete initial mip chain" : "zero initialization");
        const auto scenarioPath = tempFolder.relative("scenario.json");
        std::ofstream(scenarioPath) << R"({
            "resources": [{"image": {
                "uid": "image", "dims": [1, 8, 8, 1],
                "format": "VK_FORMAT_R8G8B8A8_UNORM", "mips": 4,
                "shader_access": "readonly"
        )" << (hasSource ? R"(, "src": "input.dds")" : "")
                                    << R"(
            }}], "commands": []
        })";
        auto scenario = ScenarioJsonFactory::make(scenarioPath);
        const auto imageId = scenario->getImageId("image");
        const std::vector<std::byte> base(packedImageDataSize(shape, format), std::byte{0x77});
        ASSERT_NO_THROW(scenario->upload(imageId, {base.data(), base.size(), shape, format}));
        EXPECT_EQ(scenario->download(imageId).data, base);
    }
}

TEST(ImageInMemoryTransfer, RejectsMipGenerationWithoutLinearFiltering) {
    ScenarioOptions opts{};
    Context ctx{opts, detail::DeviceRequirements{}};
    constexpr auto format = vk::Format::eR8G8B8A8Uint;
    const auto features = ctx.physicalDevice().getFormatProperties(format).optimalTilingFeatures;
    const auto required =
        imageTransferFeatures | vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst;
    if ((features & required) != required || (features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        GTEST_SKIP() << "Device does not expose a blit-capable integer format without linear filtering";
    }
    DataManager dataManager;
    auto &image = prepareImage(ctx, dataManager, ImageId{0}, {1, 4, 4, 1}, format, 3);
    const std::vector<std::byte> base(packedImageDataSize(image.shape(), format), std::byte{0x11});
    const auto previousLayout = image.getImageLayout();
    try {
        image.upload(ctx, {base.data(), base.size(), image.shape(), format});
        FAIL() << "Linear-filter mip generation must reject integer formats before recording blits";
    } catch (const std::runtime_error &error) {
        const std::string diagnostic = error.what();
        EXPECT_NE(diagnostic.find("test_image"), std::string::npos) << diagnostic;
        EXPECT_NE(diagnostic.find("SampledImageFilterLinear"), std::string::npos) << diagnostic;
    }
    EXPECT_EQ(image.getImageLayout(), previousLayout);
}
