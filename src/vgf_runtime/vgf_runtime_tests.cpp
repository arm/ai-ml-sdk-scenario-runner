/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf.hpp"

#include "vgf/encoder.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace mlsdk::vgf_runtime;

template <typename T, typename U> bool viewEquals(DataView<T> view, std::initializer_list<U> expected) {
    return view.size() == expected.size() &&
           std::equal(view.begin(), view.end(), expected.begin(), expected.end(),
                      [](T actual, U expectedValue) { return actual == static_cast<T>(expectedValue); });
}

std::string writeVgf(const std::function<void(mlsdk::vgflib::Encoder &)> &populate) {
    auto encoder = mlsdk::vgflib::CreateEncoder(VK_HEADER_VERSION);
    populate(*encoder);
    encoder->Finish();

    std::stringstream stream;
    EXPECT_TRUE(encoder->WriteTo(stream));
    return stream.str();
}

bool pointsInside(const void *ptr, const std::string &buffer) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data());
    return address >= begin && address < begin + buffer.size();
}

} // namespace

TEST(VGF, DecodesSegmentsModulesBindingsResourcesAndDispatch) {
    const std::vector<uint32_t> code = {0x07230203, 0x00010000, 0, 1};
    const auto data = writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::COMPUTE, "shader", "main", code);
        const auto input = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_SFLOAT, {4}, {4});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_FORMAT_R8G8B8A8_UNORM, {2, 2, 1}, {});
        const auto intermediate =
            encoder.AddIntermediateResource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_FORMAT_R32_UINT, {8}, {});
        const auto inputBinding = encoder.AddBindingSlot(3, input);
        const auto outputBinding = encoder.AddBindingSlot(5, output);
        const auto intermediateBinding = encoder.AddBindingSlot(7, intermediate);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({inputBinding, outputBinding, intermediateBinding}, 2);
        const auto pushConstantRange = encoder.AddPushConstRange(VK_SHADER_STAGE_COMPUTE_BIT, 0, 16);
        encoder.AddSegmentInfo(module, "segment", {descriptorSet}, {inputBinding}, {outputBinding}, {}, {1, 2, 3},
                               {pushConstantRange});
    });

    const VGF vgf(data.data(), data.size());

    EXPECT_EQ(vgf.getNumSegments(), 1);
    EXPECT_EQ(vgf.getNumSPIRVModules(), 1);
    EXPECT_EQ(vgf.getNumResources(), 3);
    EXPECT_EQ(vgf.getNumConstants(), 0);
    EXPECT_EQ(vgf.getNumConstants(0), 0);

    const auto segment = vgf.getSegment(0);
    EXPECT_EQ(segment.index, 0);
    EXPECT_EQ(segment.name, "segment");
    EXPECT_EQ(segment.type, ModuleType::SHADER);
    EXPECT_EQ(segment.moduleIndex, 0);
    EXPECT_EQ(segment.moduleName, "shader");
    EXPECT_EQ(segment.entryPoint, "main");

    const auto module = vgf.getSPIRVModule(segment.moduleIndex);
    EXPECT_EQ(module.index, 0);
    EXPECT_EQ(module.name, "shader");
    EXPECT_EQ(module.entryPoint, "main");
    EXPECT_TRUE(pointsInside(module.code.data(), data));
    ASSERT_EQ(module.code.size(), code.size());
    EXPECT_TRUE(std::equal(code.begin(), code.end(), module.code.begin()));

    const auto bindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(bindings.size(), 3);
    EXPECT_EQ(bindings[0].set, 2);
    EXPECT_EQ(bindings[0].binding, 3);
    EXPECT_EQ(bindings[0].resourceIndex, 0);
    EXPECT_EQ(bindings[0].descriptorType, vk::DescriptorType::eStorageBuffer);
    EXPECT_EQ(bindings[0].resourceCategory, ResourceCategory::INPUT);
    EXPECT_EQ(bindings[1].binding, 5);
    EXPECT_EQ(bindings[1].descriptorType, vk::DescriptorType::eStorageImage);
    EXPECT_EQ(bindings[1].resourceCategory, ResourceCategory::OUTPUT);
    EXPECT_EQ(bindings[2].binding, 7);
    EXPECT_EQ(bindings[2].resourceCategory, ResourceCategory::INTERMEDIATE);

    const auto inputResource = vgf.getResource(0);
    const auto outputResource = vgf.getResource(1);
    const auto intermediateResource = vgf.getResource(2);
    ASSERT_TRUE(outputResource.descriptorType.has_value());
    EXPECT_EQ(*outputResource.descriptorType, vk::DescriptorType::eStorageImage);
    EXPECT_EQ(outputResource.format, vk::Format::eR8G8B8A8Unorm);
    EXPECT_TRUE(viewEquals(outputResource.shape, {2, 2, 1}));
    EXPECT_TRUE(viewEquals(inputResource.stride, {4}));
    EXPECT_EQ(intermediateResource.index, 2);
    EXPECT_EQ(intermediateResource.category, ResourceCategory::INTERMEDIATE);

    const auto dispatchShape = vgf.getDispatchShape(0);
    EXPECT_TRUE(viewEquals(dispatchShape, {1, 2, 3}));
    EXPECT_EQ(vgf.getNumPushConstantRanges(0), 1);
    const auto pushConstantRange = vgf.getPushConstantRange(0, 0);
    EXPECT_EQ(pushConstantRange.stageFlags, VK_SHADER_STAGE_COMPUTE_BIT);
    EXPECT_EQ(pushConstantRange.offset, 0);
    EXPECT_EQ(pushConstantRange.size, 16);
}

TEST(VGF, DecodesConstantsSamplersAndFileBackedData) {
    const std::vector<uint32_t> code = {0x07230203, 0x00010000, 0, 2};
    const std::array<int32_t, 4> constantData = {1, 2, 3, 4};
    const auto data = writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "graph", "main", code);
        const auto image = encoder.AddInputResource(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_FORMAT_R8G8B8A8_UNORM,
                                                    {4, 4, 1}, {});
        encoder.AddSamplerConfig(image, VK_FILTER_LINEAR, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                 VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
        const auto constantResource = encoder.AddConstantResource(VK_FORMAT_R32_SINT, {4}, {});
        const auto constant =
            encoder.AddConstant(constantResource, constantData.data(), constantData.size() * sizeof(int32_t), 1);
        const auto imageBinding = encoder.AddBindingSlot(0, image);
        const auto descriptorSet = encoder.AddDescriptorSetInfo({imageBinding}, 0);
        encoder.AddSegmentInfo(module, "graph_segment", {descriptorSet}, {imageBinding}, {}, {constant});
    });

    const auto path = std::filesystem::current_path() / "vgf_runtime_file_test.vgf";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(data.data(), static_cast<std::streamsize>(data.size()));
    }

    const VGF vgf(path);
    std::filesystem::remove(path);

    EXPECT_EQ(vgf.getNumSegments(), 1);
    EXPECT_EQ(vgf.getNumSPIRVModules(), 1);
    EXPECT_EQ(vgf.getNumResources(), 2);
    EXPECT_EQ(vgf.getNumConstants(), 1);
    EXPECT_EQ(vgf.getNumConstants(0), 1);
    EXPECT_EQ(vgf.getNumPushConstantRanges(0), 0);

    const auto segment = vgf.getSegment(0);
    EXPECT_EQ(segment.type, ModuleType::GRAPH);
    const auto module = vgf.getSPIRVModule(segment.moduleIndex);
    EXPECT_EQ(module.name, "graph");

    const auto resource = vgf.getResource(0);
    ASSERT_TRUE(resource.sampler.has_value());
    EXPECT_EQ(resource.sampler->minFilter, VK_FILTER_LINEAR);
    EXPECT_EQ(resource.sampler->magFilter, VK_FILTER_NEAREST);
    EXPECT_EQ(resource.sampler->addressModeU, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    EXPECT_EQ(resource.sampler->addressModeV, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    EXPECT_EQ(resource.sampler->borderColor, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);

    const auto constantResource = vgf.getResource(1);
    EXPECT_EQ(constantResource.category, ResourceCategory::CONSTANT);
    EXPECT_FALSE(constantResource.descriptorType.has_value());
    EXPECT_EQ(constantResource.format, vk::Format::eR32Sint);
    EXPECT_TRUE(viewEquals(constantResource.shape, {4}));

    const auto constantIndexes = vgf.getConstantIndexes(0);
    EXPECT_TRUE(viewEquals(constantIndexes, {0}));

    const auto constant = vgf.getConstant(0, 0);
    EXPECT_EQ(constant.index, 0);
    EXPECT_EQ(constant.resourceIndex, 1);
    EXPECT_EQ(constant.format, vk::Format::eR32Sint);
    EXPECT_TRUE(viewEquals(constant.shape, {4}));
    EXPECT_EQ(constant.sparsityDimension, 1);
    ASSERT_EQ(constant.data.size(), constantData.size() * sizeof(int32_t));
    EXPECT_EQ(*reinterpret_cast<const int32_t *>(constant.data.data()), 1);
}
