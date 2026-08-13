/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vgf_view.hpp"

#include "data_manager.hpp"

#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include "vgf-utils/temp_folder.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace mlsdk::scenariorunner;
namespace vgflib = mlsdk::vgflib;

namespace {

class CapturingResourceCreator final : public IResourceCreator {
  public:
    BufferId createBuffer(BufferInfo &&info) override {
        const BufferId id{buffers.size()};
        buffers.push_back(std::move(info));
        return id;
    }

    TensorId createTensor(TensorInfo &&info) override {
        const TensorId id{tensors.size()};
        tensors.push_back(std::move(info));
        return id;
    }

    ImageId createImage(ImageInfo &&info) override {
        const ImageId id{images.size()};
        images.push_back(std::move(info));
        return id;
    }

    std::vector<BufferInfo> buffers;
    std::vector<TensorInfo> tensors;
    std::vector<ImageInfo> images;
};

std::filesystem::path writeVgfWithTensorInterfaceResources(TempFolder &tempFolder) {
    auto encoder = vgflib::CreateEncoder(123);

    const auto module = encoder->AddModule(vgflib::ModuleType::COMPUTE, "tensor_interface", "main");

    const std::vector<int64_t> inputShape{1, 16, 16, 4};
    const std::vector<int64_t> outputShape{1, 8, 8, 4};

    const auto inputTensor = encoder->AddInputResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_TENSOR_ARM),
                                                       vgflib::ToFormatType(VK_FORMAT_R8_SINT), inputShape, {});
    const auto outputTensor = encoder->AddOutputResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_TENSOR_ARM),
                                                         vgflib::ToFormatType(VK_FORMAT_R8_SINT), outputShape, {});

    const auto inputBinding = encoder->AddBindingSlot(0, inputTensor);
    const auto outputBinding = encoder->AddBindingSlot(1, outputTensor);
    const auto descriptorSet =
        encoder->AddDescriptorSetInfo(std::vector<vgflib::BindingSlotRef>{inputBinding, outputBinding});

    encoder->AddModelSequenceInputsOutputs({inputBinding}, {"inputTensor"}, {outputBinding}, {"outputTensor"});
    encoder->AddSegmentInfo(module, "tensor_interface_segment",
                            std::vector<vgflib::DescriptorSetInfoRef>{descriptorSet}, {inputBinding}, {outputBinding},
                            std::vector<vgflib::GraphConstantBindingRef>{}, {1, 1, 1});
    encoder->Finish();

    const std::string vgfPath = tempFolder.relative("scenario_runner_vgf_view_tensor_interface.vgf").string();
    std::ofstream output(vgfPath, std::ios::binary);
    encoder->WriteTo(output);
    return vgfPath;
}

DataManager makeDataManagerWithTensor(const std::string &uid, std::vector<int64_t> shape, vk::Format format) {
    DataManager dataManager;
    TensorInfo info{};
    info.debugName = uid;
    info.shape = std::move(shape);
    info.format = format;
    dataManager.createTensor(TensorId{0}, info);
    return dataManager;
}

std::string mismatchMessageFor(const VgfView &view, const DataManager &dataManager, const TypedBinding &binding) {
    try {
        view.resolveBindings(0, dataManager, std::vector<TypedBinding>{binding}, {});
        return {};
    } catch (const std::runtime_error &error) {
        return error.what();
    }
}

std::filesystem::path writeVgfWithIntermediateBuffer(TempFolder &tempFolder, vk::Format format,
                                                     const std::vector<int64_t> &shape) {
    auto encoder = vgflib::CreateEncoder(123);

    const auto module = encoder->AddModule(vgflib::ModuleType::COMPUTE, "buffer_size", "main");
    const auto buffer = encoder->AddIntermediateResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                                                         vgflib::ToFormatType(format), shape, {});
    const auto binding = encoder->AddBindingSlot(0, buffer);
    const auto descriptorSet = encoder->AddDescriptorSetInfo(std::vector<vgflib::BindingSlotRef>{binding});

    encoder->AddModelSequenceInputsOutputs({}, {}, {}, {});
    encoder->AddSegmentInfo(module, "buffer_size_segment", std::vector<vgflib::DescriptorSetInfoRef>{descriptorSet}, {},
                            std::vector<vgflib::BindingSlotRef>{binding},
                            std::vector<vgflib::GraphConstantBindingRef>{}, {1, 1, 1});
    encoder->Finish();

    const std::string vgfPath = tempFolder.relative("scenario_runner_vgf_view_buffer_size.vgf").string();
    std::ofstream output(vgfPath, std::ios::binary);
    encoder->WriteTo(output);
    return vgfPath;
}

std::filesystem::path writeVgfWithInputAndIntermediateBuffer(TempFolder &tempFolder) {
    auto encoder = vgflib::CreateEncoder(123);
    constexpr uint32_t aliasGroupId = 17;
    const std::vector<int64_t> shape{1};

    const auto module = encoder->AddModule(vgflib::ModuleType::COMPUTE, "resource_indexes", "main");
    const auto input = encoder->AddInputResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                                                 vgflib::ToFormatType(VK_FORMAT_R8_UINT), shape, {});
    const auto intermediate =
        encoder->AddIntermediateResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                                         vgflib::ToFormatType(VK_FORMAT_R8_UINT), shape, {});
    encoder->SetAliasGroup(input, aliasGroupId);
    encoder->SetAliasGroup(intermediate, aliasGroupId);
    const auto inputBinding = encoder->AddBindingSlot(0, input);
    const auto intermediateBinding = encoder->AddBindingSlot(1, intermediate);
    const auto descriptorSet =
        encoder->AddDescriptorSetInfo(std::vector<vgflib::BindingSlotRef>{inputBinding, intermediateBinding});

    encoder->AddModelSequenceInputsOutputs({inputBinding}, {"input"}, {}, {});
    encoder->AddSegmentInfo(module, "resource_indexes_segment",
                            std::vector<vgflib::DescriptorSetInfoRef>{descriptorSet}, {inputBinding},
                            {intermediateBinding}, std::vector<vgflib::GraphConstantBindingRef>{}, {1, 1, 1});
    encoder->Finish();

    const std::string vgfPath = tempFolder.relative("scenario_runner_vgf_view_resource_indexes.vgf").string();
    std::ofstream output(vgfPath, std::ios::binary);
    encoder->WriteTo(output);
    return vgfPath;
}

std::filesystem::path writeVgfWithSampledIntermediateImage(TempFolder &tempFolder, uint32_t addressModeU,
                                                           uint32_t addressModeV,
                                                           const std::vector<int64_t> &shape = {1, 16, 8, 4}) {
    auto encoder = vgflib::CreateEncoder(123);

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
                            {}, std::vector<vgflib::BindingSlotRef>{binding},
                            std::vector<vgflib::GraphConstantBindingRef>{}, {1, 1, 1});
    encoder->Finish();

    const auto suffix = std::to_string(addressModeU) + "_" + std::to_string(addressModeV);
    const auto vgfPath = tempFolder.relative("scenario_runner_vgf_view_sampled_image_" + suffix + ".vgf").string();
    std::ofstream output(vgfPath, std::ios::binary);
    encoder->WriteTo(output);
    return vgfPath;
}

} // namespace

TEST(VgfView, IntermediateBufferSizeUsesVgfFormatElementSize) {
    constexpr uint32_t elementCount = 256;
    TempFolder tempFolder("vgf_view");
    const auto vgfPath = writeVgfWithIntermediateBuffer(tempFolder, vk::Format::eR16Uint, {elementCount});

    auto view = VgfView::createVgfView(vgfPath.string());
    CapturingResourceCreator creator;

    const auto result = view.createIntermediateResources(creator);

    ASSERT_EQ(creator.buffers.size(), 1);
    EXPECT_EQ(creator.buffers.front().size, elementCount * sizeof(uint16_t));
    EXPECT_TRUE(creator.tensors.empty());
    EXPECT_TRUE(creator.images.empty());
    EXPECT_TRUE(result.memoryGroups.empty());
}

TEST(VgfView, IntermediateResourceAliasesUseCreatedIds) {
    TempFolder tempFolder("vgf_view");
    const auto vgfPath = writeVgfWithInputAndIntermediateBuffer(tempFolder);

    auto view = VgfView::createVgfView(vgfPath.string());
    CapturingResourceCreator creator;

    const auto result = view.createIntermediateResources(creator);

    ASSERT_EQ(creator.buffers.size(), 1);
    ASSERT_EQ(result.memoryGroups.size(), 1);
    ASSERT_EQ(result.memoryGroups.at(17).size(), 1);
    EXPECT_EQ(std::get<BufferId>(result.memoryGroups.at(17).front()), BufferId{0});
    EXPECT_EQ(view.getModelResourceAliasGroup(0), 17);
    EXPECT_THROW(view.getModelResourceAliasGroup(99), std::runtime_error);
}

TEST(VgfView, IntermediateSampledImageUsesVgfSamplerConfig) {
    TempFolder tempFolder("vgf_view");
    const auto vgfPath = writeVgfWithSampledIntermediateImage(tempFolder, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);

    auto view = VgfView::createVgfView(vgfPath.string());
    CapturingResourceCreator creator;

    const auto result = view.createIntermediateResources(creator);

    ASSERT_EQ(creator.images.size(), 1);
    EXPECT_EQ(creator.images.front().shape, (std::vector<int64_t>{1, 8, 16, 1}));
    const auto &samplerSettings = creator.images.front().samplerSettings;
    EXPECT_EQ(samplerSettings.minFilter, FilterMode::Linear);
    EXPECT_EQ(samplerSettings.magFilter, FilterMode::Nearest);
    EXPECT_EQ(samplerSettings.addressModeU, AddressMode::ClampBorder);
    EXPECT_EQ(samplerSettings.addressModeV, AddressMode::ClampBorder);
    EXPECT_EQ(samplerSettings.addressModeW, AddressMode::ClampEdge);
    EXPECT_EQ(samplerSettings.borderColor, BorderColor::IntOpaqueWhite);
    EXPECT_EQ(samplerSettings.mipFilter, FilterMode::Nearest);
    EXPECT_TRUE(creator.buffers.empty());
    EXPECT_TRUE(creator.tensors.empty());
    EXPECT_TRUE(result.memoryGroups.empty());
}

TEST(VgfView, IntermediateSampledImageAcceptsDistinctVgfAddressModes) {
    TempFolder tempFolder("vgf_view");
    const auto vgfPath = writeVgfWithSampledIntermediateImage(tempFolder, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    auto view = VgfView::createVgfView(vgfPath.string());
    CapturingResourceCreator creator;

    const auto result = view.createIntermediateResources(creator);

    ASSERT_EQ(creator.images.size(), 1);
    const auto &samplerSettings = creator.images.front().samplerSettings;
    EXPECT_EQ(samplerSettings.addressModeU, AddressMode::Repeat);
    EXPECT_EQ(samplerSettings.addressModeV, AddressMode::ClampEdge);
    EXPECT_EQ(samplerSettings.addressModeW, AddressMode::ClampEdge);
    EXPECT_TRUE(result.memoryGroups.empty());
}

TEST(VgfView, IntermediateSampledImageRejectsInvalidVgfShapes) {
    TempFolder tempFolder("vgf_view");
    const std::vector<std::pair<std::vector<int64_t>, std::string>> invalidShapes{
        {{16, 8}, "VGF image resources must have shape [H, W, C] or [1, H, W, C]"},
        {{2, 16, 8, 4}, "VGF image resources must have a batch size of 1"},
        {{1, 16, 8, 3}, "VGF image channel count 3 does not match format component count 4"},
    };

    for (const auto &[shape, expectedMessage] : invalidShapes) {
        const auto vgfPath = writeVgfWithSampledIntermediateImage(tempFolder, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, shape);

        auto view = VgfView::createVgfView(vgfPath.string());
        CapturingResourceCreator creator;

        try {
            view.createIntermediateResources(creator);
            FAIL() << "Expected invalid VGF image shape to be rejected";
        } catch (const std::runtime_error &error) {
            EXPECT_EQ(error.what(), expectedMessage);
        }
    }
}

TEST(VgfView, ResolveBindingsReportsTensorShapeMismatchWithJsonLine) {
    TempFolder tempFolder("vgf_view");
    const auto vgfPath = writeVgfWithTensorInterfaceResources(tempFolder);

    auto view = VgfView::createVgfView(vgfPath.string());
    const auto dataManager = makeDataManagerWithTensor("inputTensor", {1, 16, 16, 3}, vk::Format::eR8Sint);
    const TypedBinding binding{0, 0, TensorId{0}, std::nullopt, vk::DescriptorType::eTensorARM};

    const auto message = mismatchMessageFor(view, dataManager, binding);
    ASSERT_FALSE(message.empty());
    EXPECT_NE(message.find("VGF '" + vgfPath.string()), std::string::npos);
    EXPECT_NE(message.find("input idx 0"), std::string::npos);
    EXPECT_NE(message.find("has shape [1, 16, 16, 4]"), std::string::npos);
    EXPECT_NE(message.find("scenario tensor 'inputTensor'"), std::string::npos);
    EXPECT_NE(message.find("has shape [1, 16, 16, 3]"), std::string::npos);
}

TEST(VgfView, ResolveBindingsReportsTensorFormatMismatchWithJsonLine) {
    TempFolder tempFolder("vgf_view");
    const auto vgfPath = writeVgfWithTensorInterfaceResources(tempFolder);

    auto view = VgfView::createVgfView(vgfPath.string());
    const auto dataManager = makeDataManagerWithTensor("outputTensor", {1, 8, 8, 4}, vk::Format::eR16Uint);
    const TypedBinding binding{0, 1, TensorId{0}, std::nullopt, vk::DescriptorType::eTensorARM};

    const auto message = mismatchMessageFor(view, dataManager, binding);
    ASSERT_FALSE(message.empty());
    EXPECT_NE(message.find("VGF '" + vgfPath.string()), std::string::npos);
    EXPECT_NE(message.find("output idx 0"), std::string::npos);
    EXPECT_NE(message.find("has format R8Sint"), std::string::npos);
    EXPECT_NE(message.find("scenario tensor 'outputTensor'"), std::string::npos);
    EXPECT_NE(message.find("has format R16Uint"), std::string::npos);
}
