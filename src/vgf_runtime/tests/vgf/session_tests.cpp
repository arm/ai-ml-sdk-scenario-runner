/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils.hpp"
#include <vgf_runtime/session.hpp>

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace mlsdk::vgf_runtime;
using namespace vgf_runtime::test;

class VgfRuntimeFullTest : public RuntimeSessionExecutionTest {};

const std::vector<mlsdk::vgflib::GraphConstantBindingRef> noGraphConstants;

std::string makeTwoSegmentMaxpoolVgf() {
    const auto &firstCode = assembleMaxpool16x16To8x8Spirv("maxpool_16x16_to_8x8", {0, 0, 0, 1});
    const auto &secondCode = assembleMaxpool8x8To4x4Spirv("maxpool_8x8_to_4x4", {0, 0, 0, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto firstModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_16x16_to_8x8", "main", firstCode);
        const auto secondModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_8x8_to_4x4", "main", secondCode);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto firstOutput =
            encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto firstOutputBinding = encoder.AddBindingSlot(1, firstOutput);
        const auto firstDescriptorSet = encoder.AddDescriptorSetInfo({firstInputBinding, firstOutputBinding});
        encoder.AddSegmentInfo(firstModule, "first_graph_segment", {firstDescriptorSet}, {firstInputBinding},
                               {firstOutputBinding}, noGraphConstants);

        const auto secondOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16}, {});
        const auto secondInputBinding = encoder.AddBindingSlot(0, firstOutput);
        const auto secondOutputBinding = encoder.AddBindingSlot(1, secondOutput);
        const auto secondDescriptorSet = encoder.AddDescriptorSetInfo({secondInputBinding, secondOutputBinding});
        encoder.AddSegmentInfo(secondModule, "second_graph_segment", {secondDescriptorSet}, {secondInputBinding},
                               {secondOutputBinding}, noGraphConstants);
    });
}

std::string makeOutputBufferAliasedToIntermediateTensorVgf() {
    const auto &maxpoolCode = assembleMaxpool8x8To4x4Spirv("maxpool_8x8_to_4x4", {0, 0, 0, 1});
    const auto &addCode = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 17;
        const auto maxpoolModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool_8x8_to_4x4_mixed_alias", "main", maxpoolCode);
        const auto addModule =
            encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers_mixed_alias", "main", addCode);

        const auto tensorInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto intermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT,
                                                                  {1, 4, 4, 16}, {}, aliasGroup);
        const auto aliasedOutputBuffer =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {64}, {4}, aliasGroup);
        const auto zeroInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto finalOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});

        const auto tensorInputBinding = encoder.AddBindingSlot(0, tensorInput);
        const auto intermediateBinding = encoder.AddBindingSlot(1, intermediate);
        const auto tensorDescriptorSet = encoder.AddDescriptorSetInfo({tensorInputBinding, intermediateBinding}, 0);
        encoder.AddSegmentInfo(maxpoolModule, "write_intermediate_tensor_alias", {tensorDescriptorSet},
                               {tensorInputBinding}, {intermediateBinding}, noGraphConstants);

        const auto aliasedOutputBufferBinding = encoder.AddBindingSlot(0, aliasedOutputBuffer);
        const auto zeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto finalOutputBinding = encoder.AddBindingSlot(2, finalOutput);
        const auto bufferInputSet = encoder.AddDescriptorSetInfo({aliasedOutputBufferBinding, zeroInputBinding}, 0);
        const auto bufferOutputSet = encoder.AddDescriptorSetInfo({finalOutputBinding}, 1);
        encoder.AddSegmentInfo(addModule, "read_bound_output_buffer_alias", {bufferInputSet, bufferOutputSet},
                               {aliasedOutputBufferBinding, zeroInputBinding}, {finalOutputBinding}, noGraphConstants,
                               {10, 1, 1});
    });
}

std::string makeAddInt32BuffersVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto outputBinding = encoder.AddBindingSlot(2, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "add_int32_buffers_segment", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding}, {outputBinding}, noGraphConstants, {10, 1, 1});
    });
}

std::string makeAddInt32BuffersWithExternalImageVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module =
            encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers_with_image", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto imageInput = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                         VK_FORMAT_R8G8B8A8_SNORM, {1, 2, 2, 4}, {});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto imageInputBinding = encoder.AddBindingSlot(3, imageInput);
        const auto outputBinding = encoder.AddBindingSlot(2, output);
        const auto inputSet =
            encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding, imageInputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "add_int32_buffers_with_external_image_segment", {inputSet, outputSet},
                               {firstInputBinding, secondInputBinding, imageInputBinding}, {outputBinding},
                               noGraphConstants, {10, 1, 1});
    });
}

std::string makeDisjointAliasedIntermediateBuffersVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 3;
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto thirdInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto fourthInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto zeroInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto firstOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto firstIntermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                       VK_FORMAT_R32_SINT, {10}, {4}, aliasGroup);
        const auto secondIntermediate =
            encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        encoder.SetAliasGroup(secondIntermediate, aliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto firstIntermediateOutputBinding = encoder.AddBindingSlot(2, firstIntermediate);
        const auto firstInputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto firstOutputSet = encoder.AddDescriptorSetInfo({firstIntermediateOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "write_first_alias", {firstInputSet, firstOutputSet},
                               {firstInputBinding, secondInputBinding}, {firstIntermediateOutputBinding},
                               noGraphConstants, {10, 1, 1});

        const auto firstIntermediateInputBinding = encoder.AddBindingSlot(0, firstIntermediate);
        const auto zeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto firstOutputBinding = encoder.AddBindingSlot(2, firstOutput);
        const auto secondInputSet = encoder.AddDescriptorSetInfo({firstIntermediateInputBinding, zeroInputBinding}, 0);
        const auto secondOutputSet = encoder.AddDescriptorSetInfo({firstOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_first_alias", {secondInputSet, secondOutputSet},
                               {firstIntermediateInputBinding, zeroInputBinding}, {firstOutputBinding},
                               noGraphConstants, {10, 1, 1});

        const auto thirdInputBinding = encoder.AddBindingSlot(0, thirdInput);
        const auto fourthInputBinding = encoder.AddBindingSlot(1, fourthInput);
        const auto secondIntermediateOutputBinding = encoder.AddBindingSlot(2, secondIntermediate);
        const auto thirdInputSet = encoder.AddDescriptorSetInfo({thirdInputBinding, fourthInputBinding}, 0);
        const auto thirdOutputSet = encoder.AddDescriptorSetInfo({secondIntermediateOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "write_second_alias", {thirdInputSet, thirdOutputSet},
                               {thirdInputBinding, fourthInputBinding}, {secondIntermediateOutputBinding},
                               noGraphConstants, {10, 1, 1});

        const auto secondIntermediateInputBinding = encoder.AddBindingSlot(0, secondIntermediate);
        const auto repeatedZeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto secondOutputBinding = encoder.AddBindingSlot(2, secondOutput);
        const auto fourthInputSet =
            encoder.AddDescriptorSetInfo({secondIntermediateInputBinding, repeatedZeroInputBinding}, 0);
        const auto fourthOutputSet = encoder.AddDescriptorSetInfo({secondOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_second_alias", {fourthInputSet, fourthOutputSet},
                               {secondIntermediateInputBinding, repeatedZeroInputBinding}, {secondOutputBinding},
                               noGraphConstants, {10, 1, 1});
    });
}

std::string makeIndependentAliasGroupsVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t firstAliasGroup = 5;
        constexpr uint32_t secondAliasGroup = 7;
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto thirdInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto fourthInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto output = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto firstIntermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                       VK_FORMAT_R32_SINT, {10}, {4}, firstAliasGroup);
        const auto secondIntermediate = encoder.AddIntermediateResource(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4}, secondAliasGroup);
        const auto firstAliasedOutput = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT,
                                                                  {10}, {4}, firstAliasGroup);
        const auto secondAliasedOutput = encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                   VK_FORMAT_R32_SINT, {10}, {4}, secondAliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto firstIntermediateBinding = encoder.AddBindingSlot(2, firstIntermediate);
        const auto firstInputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto firstOutputSet = encoder.AddDescriptorSetInfo({firstIntermediateBinding}, 1);
        encoder.AddSegmentInfo(module, "write_first_alias_group", {firstInputSet, firstOutputSet},
                               {firstInputBinding, secondInputBinding}, {firstIntermediateBinding}, noGraphConstants,
                               {10, 1, 1});

        const auto thirdInputBinding = encoder.AddBindingSlot(0, thirdInput);
        const auto fourthInputBinding = encoder.AddBindingSlot(1, fourthInput);
        const auto secondIntermediateBinding = encoder.AddBindingSlot(2, secondIntermediate);
        const auto secondInputSet = encoder.AddDescriptorSetInfo({thirdInputBinding, fourthInputBinding}, 0);
        const auto secondOutputSet = encoder.AddDescriptorSetInfo({secondIntermediateBinding}, 1);
        encoder.AddSegmentInfo(module, "write_second_alias_group", {secondInputSet, secondOutputSet},
                               {thirdInputBinding, fourthInputBinding}, {secondIntermediateBinding}, noGraphConstants,
                               {10, 1, 1});

        const auto firstAliasedOutputBinding = encoder.AddBindingSlot(0, firstAliasedOutput);
        const auto secondAliasedOutputBinding = encoder.AddBindingSlot(1, secondAliasedOutput);
        const auto outputBinding = encoder.AddBindingSlot(2, output);
        const auto thirdInputSet =
            encoder.AddDescriptorSetInfo({firstAliasedOutputBinding, secondAliasedOutputBinding}, 0);
        const auto thirdOutputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_both_alias_groups", {thirdInputSet, thirdOutputSet},
                               {firstAliasedOutputBinding, secondAliasedOutputBinding}, {outputBinding},
                               noGraphConstants, {10, 1, 1});
    });
}

struct Image {
    Image(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device)
        : image(device, vk::ImageCreateInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Snorm, {2, 2, 1}, 1, 1,
                                            vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                                            vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive)) {
        const auto memoryRequirements = image.getMemoryRequirements();
        memorySize = memoryRequirements.size;
        const auto memoryType = vgf_runtime::detail::vulkan_helpers::findMemoryType(
            physicalDevice, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        memory = vk::raii::DeviceMemory(device, {memorySize, memoryType});
        image.bindMemory(*memory, 0);
    }

    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::Image image{nullptr};
    vk::DeviceSize memorySize = 0;
};

std::string makeOutputAliasedToIntermediateBufferVgf() {
    const auto &code = assembleAddInt32BuffersSpirv();
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        constexpr uint32_t aliasGroup = 11;
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "add_int32_buffers", "main", code);

        const auto firstInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto secondInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto zeroInput =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto finalOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4});
        const auto intermediate = encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT,
                                                                  {10}, {4}, aliasGroup);
        const auto aliasedOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SINT, {10}, {4}, aliasGroup);

        const auto firstInputBinding = encoder.AddBindingSlot(0, firstInput);
        const auto secondInputBinding = encoder.AddBindingSlot(1, secondInput);
        const auto intermediateBinding = encoder.AddBindingSlot(2, intermediate);
        const auto firstInputSet = encoder.AddDescriptorSetInfo({firstInputBinding, secondInputBinding}, 0);
        const auto firstOutputSet = encoder.AddDescriptorSetInfo({intermediateBinding}, 1);
        encoder.AddSegmentInfo(module, "write_intermediate_alias", {firstInputSet, firstOutputSet},
                               {firstInputBinding, secondInputBinding}, {intermediateBinding}, noGraphConstants,
                               {10, 1, 1});

        const auto aliasedOutputInputBinding = encoder.AddBindingSlot(0, aliasedOutput);
        const auto zeroInputBinding = encoder.AddBindingSlot(1, zeroInput);
        const auto finalOutputBinding = encoder.AddBindingSlot(2, finalOutput);
        const auto secondInputSet = encoder.AddDescriptorSetInfo({aliasedOutputInputBinding, zeroInputBinding}, 0);
        const auto secondOutputSet = encoder.AddDescriptorSetInfo({finalOutputBinding}, 1);
        encoder.AddSegmentInfo(module, "read_output_alias", {secondInputSet, secondOutputSet},
                               {aliasedOutputInputBinding, zeroInputBinding}, {finalOutputBinding}, noGraphConstants,
                               {10, 1, 1});
    });
}

} // namespace

TEST_F(VgfRuntimeFullTest, RunComputeShaderSegment) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeAddInt32BuffersVgf();
    const VGF vgf(bytes.data(), bytes.size());

    Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, 2, 3, 4, 5, -6, -7, 8, 9, 10};
    const std::vector<int32_t> secondInput = {10, 9, 8, 7, 6, 5, 4, -3, -2, -1};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto bindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(bindings.size(), 3);
    session.bindBuffer(firstInputBuffer.buffer, bindings[0]);
    session.bindBuffer(secondInputBuffer.buffer, bindings[1]);
    session.bindBuffer(outputBuffer.buffer, bindings[2]);

    session.configure();
    session.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfRuntimeFullTest, RunComputeShaderSegmentWithExternalImageBinding) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeAddInt32BuffersWithExternalImageVgf();
    const VGF vgf(bytes.data(), bytes.size());

    Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    Buffer outputBuffer(physicalDevice, device, bufferSize);
    Image imageInput(physicalDevice, device);

    const std::vector<int32_t> firstInput = {1, 2, 3, 4, 5, -6, -7, 8, 9, 10};
    const std::vector<int32_t> secondInput = {10, 9, 8, 7, 6, 5, 4, -3, -2, -1};
    std::vector<int32_t> expected(elements);
    std::transform(firstInput.begin(), firstInput.end(), secondInput.begin(), expected.begin(), std::plus<>());

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    outputBuffer.write(std::vector<int32_t>(elements, 0));

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto bindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(bindings.size(), 4);
    session.bindBuffer(firstInputBuffer.buffer, bindings[0]);
    session.bindBuffer(secondInputBuffer.buffer, bindings[1]);
    session.bindImage(imageInput.image, bindings[2], vk::ImageLayout::eUndefined);
    session.bindBuffer(outputBuffer.buffer, bindings[3]);

    session.configure();
    session.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);

    session.run();

    EXPECT_EQ(outputBuffer.read(elements), expected);
}

TEST_F(VgfRuntimeFullTest, RunDisjointAliasedIntermediateBuffers) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    // VGF layout:
    //   in0 + in1 -> [write_first_alias]  -> tmp0 (alias group 3) -> [read_first_alias]  + zero -> out0
    //   in2 + in3 -> [write_second_alias] -> tmp1 (alias group 3) -> [read_second_alias] + zero -> out1
    const auto bytes = makeDisjointAliasedIntermediateBuffersVgf();
    const VGF vgf(bytes.data(), bytes.size());
    ASSERT_EQ(vgf.getNumSegments(), 4);

    const auto firstAlias = vgf.getResource(7);
    const auto secondAlias = vgf.getResource(8);
    ASSERT_TRUE(firstAlias.aliasGroupId.has_value());
    ASSERT_TRUE(secondAlias.aliasGroupId.has_value());
    EXPECT_EQ(*firstAlias.aliasGroupId, *secondAlias.aliasGroupId);
    EXPECT_EQ(firstAlias.category, mlsdk::vgflib::ResourceCategory::INTERMEDIATE);
    EXPECT_EQ(secondAlias.category, mlsdk::vgflib::ResourceCategory::INTERMEDIATE);

    Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    Buffer thirdInputBuffer(physicalDevice, device, bufferSize);
    Buffer fourthInputBuffer(physicalDevice, device, bufferSize);
    Buffer zeroInputBuffer(physicalDevice, device, bufferSize);
    Buffer firstOutputBuffer(physicalDevice, device, bufferSize);
    Buffer secondOutputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    const std::vector<int32_t> secondInput = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    const std::vector<int32_t> thirdInput = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    const std::vector<int32_t> fourthInput = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    const std::vector<int32_t> zeros(elements, 0);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    thirdInputBuffer.write(thirdInput);
    fourthInputBuffer.write(fourthInput);
    zeroInputBuffer.write(zeros);
    firstOutputBuffer.write(zeros);
    secondOutputBuffer.write(zeros);

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto firstSegmentBindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(firstSegmentBindings.size(), 3);
    session.bindBuffer(firstInputBuffer.buffer, firstSegmentBindings[0]);
    session.bindBuffer(secondInputBuffer.buffer, firstSegmentBindings[1]);

    const auto secondSegmentBindings = vgf.getDescriptorBindings(1);
    ASSERT_EQ(secondSegmentBindings.size(), 3);
    session.bindBuffer(zeroInputBuffer.buffer, secondSegmentBindings[1]);
    session.bindBuffer(firstOutputBuffer.buffer, secondSegmentBindings[2]);

    const auto thirdSegmentBindings = vgf.getDescriptorBindings(2);
    ASSERT_EQ(thirdSegmentBindings.size(), 3);
    session.bindBuffer(thirdInputBuffer.buffer, thirdSegmentBindings[0]);
    session.bindBuffer(fourthInputBuffer.buffer, thirdSegmentBindings[1]);

    const auto fourthSegmentBindings = vgf.getDescriptorBindings(3);
    ASSERT_EQ(fourthSegmentBindings.size(), 3);
    session.bindBuffer(secondOutputBuffer.buffer, fourthSegmentBindings[2]);

    session.configure();
    session.run();

    EXPECT_EQ(firstOutputBuffer.read(elements), addVectors(firstInput, secondInput));
    EXPECT_EQ(secondOutputBuffer.read(elements), addVectors(thirdInput, fourthInput));
}

TEST_F(VgfRuntimeFullTest, RunIndependentIntermediateAliasGroups) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    // VGF layout:
    //   in0 + in1 -> [write_first_alias_group]  -> tmp0 (alias group 5)
    //                                                 == bound out0 (alias group 5)
    //   in2 + in3 -> [write_second_alias_group] -> tmp1 (alias group 7)
    //                                                 == bound out1 (alias group 7)
    //   out0 + out1 -> [read_both_alias_groups] -> out
    const auto bytes = makeIndependentAliasGroupsVgf();
    const VGF vgf(bytes.data(), bytes.size());
    ASSERT_EQ(vgf.getNumSegments(), 3);

    const auto firstIntermediate = vgf.getResource(5);
    const auto secondIntermediate = vgf.getResource(6);
    const auto firstAliasedOutput = vgf.getResource(7);
    const auto secondAliasedOutput = vgf.getResource(8);
    ASSERT_TRUE(firstIntermediate.aliasGroupId.has_value());
    ASSERT_TRUE(secondIntermediate.aliasGroupId.has_value());
    ASSERT_TRUE(firstAliasedOutput.aliasGroupId.has_value());
    ASSERT_TRUE(secondAliasedOutput.aliasGroupId.has_value());
    EXPECT_EQ(*firstIntermediate.aliasGroupId, *firstAliasedOutput.aliasGroupId);
    EXPECT_EQ(*secondIntermediate.aliasGroupId, *secondAliasedOutput.aliasGroupId);
    EXPECT_NE(*firstIntermediate.aliasGroupId, *secondIntermediate.aliasGroupId);

    Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    Buffer thirdInputBuffer(physicalDevice, device, bufferSize);
    Buffer fourthInputBuffer(physicalDevice, device, bufferSize);
    Buffer firstAliasedOutputBuffer(physicalDevice, device, bufferSize);
    Buffer secondAliasedOutputBuffer(physicalDevice, device, bufferSize);
    Buffer outputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55};
    const std::vector<int32_t> secondInput = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};
    const std::vector<int32_t> thirdInput = {55, 34, 21, 13, 8, 5, 3, 2, 1, 1};
    const std::vector<int32_t> fourthInput = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const std::vector<int32_t> zeros(elements, 0);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    thirdInputBuffer.write(thirdInput);
    fourthInputBuffer.write(fourthInput);
    firstAliasedOutputBuffer.write(zeros);
    secondAliasedOutputBuffer.write(zeros);
    outputBuffer.write(zeros);

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto firstSegmentBindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(firstSegmentBindings.size(), 3);
    session.bindBuffer(firstInputBuffer.buffer, firstSegmentBindings[0]);
    session.bindBuffer(secondInputBuffer.buffer, firstSegmentBindings[1]);

    const auto secondSegmentBindings = vgf.getDescriptorBindings(1);
    ASSERT_EQ(secondSegmentBindings.size(), 3);
    session.bindBuffer(thirdInputBuffer.buffer, secondSegmentBindings[0]);
    session.bindBuffer(fourthInputBuffer.buffer, secondSegmentBindings[1]);

    const auto thirdSegmentBindings = vgf.getDescriptorBindings(2);
    ASSERT_EQ(thirdSegmentBindings.size(), 3);
    session.bindBuffer(firstAliasedOutputBuffer.buffer, thirdSegmentBindings[0],
                       {*firstAliasedOutputBuffer.memory, 0, firstAliasedOutputBuffer.memorySize});
    session.bindBuffer(secondAliasedOutputBuffer.buffer, thirdSegmentBindings[1],
                       {*secondAliasedOutputBuffer.memory, 0, secondAliasedOutputBuffer.memorySize});
    session.bindBuffer(outputBuffer.buffer, thirdSegmentBindings[2]);

    session.configure();
    session.run();

    EXPECT_EQ(outputBuffer.read(elements),
              addVectors(addVectors(firstInput, secondInput), addVectors(thirdInput, fourthInput)));
}

TEST_F(VgfRuntimeFullTest, RunOutputAliasedToIntermediateBuffer) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    // VGF layout:
    //   in0 + in1 -> [write_intermediate_alias] -> tmp (alias group 11)
    //                                                 == bound alias output (alias group 11) + zero
    //                                                     -> [read_output_alias] -> out
    const auto bytes = makeOutputAliasedToIntermediateBufferVgf();
    const VGF vgf(bytes.data(), bytes.size());
    ASSERT_EQ(vgf.getNumSegments(), 2);

    const auto intermediate = vgf.getResource(4);
    const auto aliasedOutput = vgf.getResource(5);
    ASSERT_TRUE(intermediate.aliasGroupId.has_value());
    ASSERT_TRUE(aliasedOutput.aliasGroupId.has_value());
    EXPECT_EQ(*intermediate.aliasGroupId, *aliasedOutput.aliasGroupId);

    Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    Buffer zeroInputBuffer(physicalDevice, device, bufferSize);
    Buffer aliasedOutputBuffer(physicalDevice, device, bufferSize);
    Buffer finalOutputBuffer(physicalDevice, device, bufferSize);

    const std::vector<int32_t> firstInput = {3, 5, 8, 13, 21, 34, 55, 89, 144, 233};
    const std::vector<int32_t> secondInput = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
    const std::vector<int32_t> zeros(elements, 0);

    firstInputBuffer.write(firstInput);
    secondInputBuffer.write(secondInput);
    zeroInputBuffer.write(zeros);
    aliasedOutputBuffer.write(zeros);
    finalOutputBuffer.write(zeros);

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto firstSegmentBindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(firstSegmentBindings.size(), 3);
    session.bindBuffer(firstInputBuffer.buffer, firstSegmentBindings[0]);
    session.bindBuffer(secondInputBuffer.buffer, firstSegmentBindings[1]);

    const auto secondSegmentBindings = vgf.getDescriptorBindings(1);
    ASSERT_EQ(secondSegmentBindings.size(), 3);
    session.bindBuffer(aliasedOutputBuffer.buffer, secondSegmentBindings[0],
                       {*aliasedOutputBuffer.memory, 0, aliasedOutputBuffer.memorySize});
    session.bindBuffer(zeroInputBuffer.buffer, secondSegmentBindings[1]);
    session.bindBuffer(finalOutputBuffer.buffer, secondSegmentBindings[2]);

    session.configure();
    session.run();

    EXPECT_EQ(finalOutputBuffer.read(elements), addVectors(firstInput, secondInput));
    EXPECT_EQ(aliasedOutputBuffer.read(elements), addVectors(firstInput, secondInput));
}

TEST_F(VgfRuntimeFullTest, ConfigureOutputAliasedToIntermediateBufferRequiresMemoryInfo) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize bufferSize = elements * sizeof(int32_t);

    const auto bytes = makeOutputAliasedToIntermediateBufferVgf();
    const VGF vgf(bytes.data(), bytes.size());
    ASSERT_EQ(vgf.getNumSegments(), 2);

    Buffer firstInputBuffer(physicalDevice, device, bufferSize);
    Buffer secondInputBuffer(physicalDevice, device, bufferSize);
    Buffer zeroInputBuffer(physicalDevice, device, bufferSize);
    Buffer aliasedOutputBuffer(physicalDevice, device, bufferSize);
    Buffer finalOutputBuffer(physicalDevice, device, bufferSize);

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto firstSegmentBindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(firstSegmentBindings.size(), 3);
    session.bindBuffer(firstInputBuffer.buffer, firstSegmentBindings[0]);
    session.bindBuffer(secondInputBuffer.buffer, firstSegmentBindings[1]);

    const auto secondSegmentBindings = vgf.getDescriptorBindings(1);
    ASSERT_EQ(secondSegmentBindings.size(), 3);
    session.bindBuffer(aliasedOutputBuffer.buffer, secondSegmentBindings[0]);
    session.bindBuffer(zeroInputBuffer.buffer, secondSegmentBindings[1]);
    session.bindBuffer(finalOutputBuffer.buffer, secondSegmentBindings[2]);

    EXPECT_THROW(session.configure(), std::runtime_error);
}

TEST_F(VgfRuntimeFullTest, RunOutputBufferAliasedToIntermediateTensor) {
    constexpr size_t elements = 10;
    constexpr vk::DeviceSize aliasedBufferSize = 64 * sizeof(int32_t);
    constexpr vk::DeviceSize outputBufferSize = elements * sizeof(int32_t);

    // VGF layout:
    //   tensor in -> [write_intermediate_tensor_alias] -> tmp tensor (alias group 17)
    //                                                     == bound alias buffer (alias group 17) + zero
    //                                                         -> [read_bound_output_buffer_alias] -> out buffer
    const auto bytes = makeOutputBufferAliasedToIntermediateTensorVgf();
    const VGF vgf(bytes.data(), bytes.size());
    ASSERT_EQ(vgf.getNumSegments(), 2);

    const auto intermediate = vgf.getResource(1);
    const auto aliasedOutputBuffer = vgf.getResource(2);
    ASSERT_TRUE(intermediate.aliasGroupId.has_value());
    ASSERT_TRUE(aliasedOutputBuffer.aliasGroupId.has_value());
    EXPECT_EQ(*intermediate.aliasGroupId, *aliasedOutputBuffer.aliasGroupId);
    EXPECT_EQ(intermediate.descriptorType, vk::DescriptorType::eTensorARM);
    EXPECT_EQ(aliasedOutputBuffer.descriptorType, vk::DescriptorType::eStorageBuffer);

    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    Buffer aliasedOutputBufferResource(physicalDevice, device, aliasedBufferSize);
    Buffer zeroInputBuffer(physicalDevice, device, outputBufferSize);
    Buffer finalOutputBuffer(physicalDevice, device, outputBufferSize);

    const auto input = makeMaxpoolInput(inputTensor.shape, 5);
    const std::vector<int32_t> zeros(elements, 0);
    inputTensor.write(input);
    aliasedOutputBufferResource.write(std::vector<int32_t>(64, 0));
    zeroInputBuffer.write(zeros);
    finalOutputBuffer.write(zeros);

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto firstBindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(firstBindings.size(), 2);
    session.bindTensor(inputTensor.tensor, firstBindings[0]);

    const auto secondBindings = vgf.getDescriptorBindings(1);
    ASSERT_EQ(secondBindings.size(), 3);
    session.bindBuffer(aliasedOutputBufferResource.buffer, secondBindings[0],
                       {*aliasedOutputBufferResource.memory, 0, aliasedOutputBufferResource.memorySize});
    session.bindBuffer(zeroInputBuffer.buffer, secondBindings[1]);
    session.bindBuffer(finalOutputBuffer.buffer, secondBindings[2]);

    session.configure();
    session.run();

    EXPECT_EQ(finalOutputBuffer.read(elements),
              int32WordsFromBytes(expectedMaxpool(input, inputTensor.shape), elements));
}

TEST_F(VgfRuntimeFullTest, RunMaxpoolDataVGF) {
    const auto bytes = makeMaxpoolVgf();
    const VGF vgf(bytes.data(), bytes.size());

    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);

    const auto input = makeMaxpoolInput(inputTensor.shape);
    inputTensor.write(input);
    outputTensor.fill(0, outputTensor.numElements());

    const auto bindings = vgf.getDescriptorBindings(0);
    session.bindTensor(inputTensor.tensor, bindings[0]);
    session.bindTensor(outputTensor.tensor, bindings[1]);
    session.configure();
    session.run();

    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
}

TEST_F(VgfRuntimeFullTest, RunMaxpoolFileVGF) {
    const auto bytes = makeMaxpoolVgf();
    const auto path = std::filesystem::current_path() / "vgf_runtime_full_maxpool.vgf";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    const VGF vgf(path);
    std::filesystem::remove(path);

    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);

    const auto input = makeMaxpoolInput(inputTensor.shape, 0);
    inputTensor.write(input);
    outputTensor.fill(0, outputTensor.numElements());

    const auto bindings = vgf.getDescriptorBindings(0);
    session.bindTensor(inputTensor.tensor, bindings[0]);
    session.bindTensor(outputTensor.tensor, bindings[1]);
    session.configure();
    session.run();

    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(input, inputTensor.shape));
}

TEST_F(VgfRuntimeFullTest, RunMaxpoolRepeatedDifferentInput) {
    const auto bytes = makeMaxpoolVgf();
    const VGF vgf(bytes.data(), bytes.size());
    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);

    const auto bindings = vgf.getDescriptorBindings(0);
    session.bindTensor(inputTensor.tensor, bindings[0]);
    session.bindTensor(outputTensor.tensor, bindings[1]);
    session.configure();

    const auto firstInput = makeMaxpoolInput(inputTensor.shape, 3);
    inputTensor.write(firstInput);
    outputTensor.fill(0, outputTensor.numElements());
    session.run();
    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(firstInput, inputTensor.shape));

    const auto secondInput = makeMaxpoolInput(inputTensor.shape, 41);
    inputTensor.write(secondInput);
    outputTensor.fill(0, outputTensor.numElements());
    session.run();
    EXPECT_EQ(outputTensor.read(outputTensor.numElements()), expectedMaxpool(secondInput, inputTensor.shape));
}

TEST_F(VgfRuntimeFullTest, RunTwoMaxpoolGraphSegments) {
    const auto bytes = makeTwoSegmentMaxpoolVgf();
    const VGF vgf(bytes.data(), bytes.size());
    ASSERT_EQ(vgf.getNumSegments(), 2);

    Tensor firstInputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    Tensor secondOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    const auto firstInput = makeMaxpoolInput(firstInputTensor.shape, 7);
    firstInputTensor.write(firstInput);
    secondOutputTensor.fill(0, secondOutputTensor.numElements());

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto firstBindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(firstBindings.size(), 2);
    session.bindTensor(firstInputTensor.tensor, firstBindings[0]);

    const auto secondBindings = vgf.getDescriptorBindings(1);
    ASSERT_EQ(secondBindings.size(), 2);
    session.bindTensor(secondOutputTensor.tensor, secondBindings[1]);

    session.configure();
    session.run();

    const auto firstExpected = expectedMaxpool(firstInput, firstInputTensor.shape);
    EXPECT_EQ(secondOutputTensor.read(secondOutputTensor.numElements()), expectedMaxpool(firstExpected, {1, 8, 8, 16}));
}
