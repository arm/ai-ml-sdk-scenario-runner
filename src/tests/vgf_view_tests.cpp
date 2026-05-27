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
