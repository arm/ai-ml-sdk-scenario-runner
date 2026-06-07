/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vgf_view.hpp"

#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace mlsdk::scenariorunner;
namespace vgflib = mlsdk::vgflib;

namespace {

class CapturingResourceCreator final : public IResourceCreator {
  public:
    void createBuffer(Guid guid, BufferInfo info) override { buffers.push_back({guid, info}); }
    void createTensor(Guid guid, TensorInfo info) override { tensors.push_back({guid, info}); }
    void createImage(Guid guid, ImageInfo info) override { images.push_back({guid, info}); }

    std::vector<std::pair<Guid, BufferInfo>> buffers;
    std::vector<std::pair<Guid, TensorInfo>> tensors;
    std::vector<std::pair<Guid, ImageInfo>> images;
};

std::filesystem::path writeVgfWithIntermediateBuffer(vk::Format format, const std::vector<int64_t> &shape) {
    auto encoder = vgflib::CreateEncoder(123);

    const auto module = encoder->AddModule(vgflib::ModuleType::COMPUTE, "buffer_size", "main");
    const auto buffer = encoder->AddIntermediateResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                                                         vgflib::ToFormatType(format), shape, {});
    const auto binding = encoder->AddBindingSlot(0, buffer);
    const auto descriptorSet = encoder->AddDescriptorSetInfo(std::vector<vgflib::BindingSlotRef>{binding});

    encoder->AddModelSequenceInputsOutputs({}, {}, {}, {});
    encoder->AddSegmentInfo(module, "buffer_size_segment", std::vector<vgflib::DescriptorSetInfoRef>{descriptorSet}, {},
                            std::vector<vgflib::BindingSlotRef>{binding}, {}, {1, 1, 1});
    encoder->Finish();

    const auto vgfPath = std::filesystem::temp_directory_path() / "scenario_runner_vgf_view_buffer_size.vgf";
    std::ofstream output(vgfPath, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open temporary VGF file for writing");
    }
    if (!encoder->WriteTo(output)) {
        throw std::runtime_error("Failed to write temporary VGF file");
    }
    return vgfPath;
}

std::filesystem::path writeVgfWithSampledIntermediateImage(uint32_t addressModeU, uint32_t addressModeV) {
    auto encoder = vgflib::CreateEncoder(123);

    const std::vector<int64_t> shape{16, 16, 1, 1};
    const auto module = encoder->AddModule(vgflib::ModuleType::COMPUTE, "sampled_image", "main");
    const auto image =
        encoder->AddIntermediateResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
                                         vgflib::ToFormatType(VK_FORMAT_R8G8B8A8_UNORM), shape, {});
    encoder->AddSamplerConfig(image, VK_FILTER_LINEAR, VK_FILTER_NEAREST, addressModeU, addressModeV,
                              VK_BORDER_COLOR_INT_OPAQUE_WHITE);
    const auto binding = encoder->AddBindingSlot(0, image);
    const auto descriptorSet = encoder->AddDescriptorSetInfo(std::vector<vgflib::BindingSlotRef>{binding});

    encoder->AddModelSequenceInputsOutputs({}, {}, {}, {});
    encoder->AddSegmentInfo(module, "sampled_image_segment", std::vector<vgflib::DescriptorSetInfoRef>{descriptorSet},
                            {}, std::vector<vgflib::BindingSlotRef>{binding}, {}, {1, 1, 1});
    encoder->Finish();

    const auto suffix = std::to_string(addressModeU) + "_" + std::to_string(addressModeV);
    const auto vgfPath =
        std::filesystem::temp_directory_path() / ("scenario_runner_vgf_view_sampled_image_" + suffix + ".vgf");
    std::ofstream output(vgfPath, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open temporary VGF file for writing");
    }
    if (!encoder->WriteTo(output)) {
        throw std::runtime_error("Failed to write temporary VGF file");
    }
    return vgfPath;
}

} // namespace

TEST(VgfView, IntermediateBufferSizeUsesVgfFormatElementSize) {
    constexpr uint32_t elementCount = 256;
    const auto vgfPath = writeVgfWithIntermediateBuffer(vk::Format::eR16Uint, {elementCount});

    try {
        auto view = VgfView::createVgfView(vgfPath.string());
        CapturingResourceCreator creator;

        view.createIntermediateResources(creator);

        ASSERT_EQ(creator.buffers.size(), 1);
        EXPECT_EQ(creator.buffers.front().second.size, elementCount * sizeof(uint16_t));
        EXPECT_TRUE(creator.tensors.empty());
        EXPECT_TRUE(creator.images.empty());
    } catch (...) {
        std::filesystem::remove(vgfPath);
        throw;
    }
    std::filesystem::remove(vgfPath);
}

TEST(VgfView, IntermediateSampledImageUsesVgfSamplerConfig) {
    const auto vgfPath = writeVgfWithSampledIntermediateImage(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);

    try {
        auto view = VgfView::createVgfView(vgfPath.string());
        CapturingResourceCreator creator;

        view.createIntermediateResources(creator);

        ASSERT_EQ(creator.images.size(), 1);
        const auto &samplerSettings = creator.images.front().second.samplerSettings;
        EXPECT_EQ(samplerSettings.minFilter, FilterMode::Linear);
        EXPECT_EQ(samplerSettings.magFilter, FilterMode::Nearest);
        EXPECT_EQ(samplerSettings.borderAddressMode, AddressMode::ClampBorder);
        EXPECT_EQ(samplerSettings.borderColor, BorderColor::IntOpaqueWhite);
        EXPECT_EQ(samplerSettings.mipFilter, FilterMode::Nearest);
        EXPECT_TRUE(creator.buffers.empty());
        EXPECT_TRUE(creator.tensors.empty());
    } catch (...) {
        std::filesystem::remove(vgfPath);
        throw;
    }
    std::filesystem::remove(vgfPath);
}

TEST(VgfView, IntermediateSampledImageRejectsDistinctVgfAddressModes) {
    const auto vgfPath =
        writeVgfWithSampledIntermediateImage(VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    try {
        auto view = VgfView::createVgfView(vgfPath.string());
        CapturingResourceCreator creator;

        try {
            view.createIntermediateResources(creator);
            FAIL() << "Expected createIntermediateResources to reject distinct sampler U/V address modes";
        } catch (const std::runtime_error &error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("Distinct VGF sampler U/V address modes are not yet supported"), std::string::npos);
        }
    } catch (...) {
        std::filesystem::remove(vgfPath);
        throw;
    }
    std::filesystem::remove(vgfPath);
}
