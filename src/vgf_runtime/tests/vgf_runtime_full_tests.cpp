/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "session.hpp"
#include "vgf.hpp"
#include "vgf_runtime_test_utils.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace mlsdk::vgf_runtime;
using namespace mlsdk::vgf_runtime::test;

class VgfRuntimeFullTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const vk::ApplicationInfo applicationInfo("vgf-runtime-full-test", 1, nullptr, 0, VK_API_VERSION_1_3);
        instance = vk::raii::Instance(context, vk::InstanceCreateInfo({}, &applicationInfo));

        for (auto &candidate : vk::raii::PhysicalDevices(instance)) {
            const auto extensions = candidate.enumerateDeviceExtensionProperties();
            if (!hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME) ||
                !hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME)) {
                continue;
            }
            const auto candidateQueueFamilyIndex = findDataGraphQueueFamily(candidate);
            if (candidateQueueFamilyIndex != UINT32_MAX) {
                physicalDevice = candidate;
                queueFamilyIndex = candidateQueueFamilyIndex;
                break;
            }
        }
        if (queueFamilyIndex == UINT32_MAX) {
            GTEST_SKIP() << "No Vulkan device with VK_ARM_data_graph, VK_ARM_tensors, and compute queue support";
        }

        const float queuePriority = 1.0F;
        const vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.shaderInt16 = true;
        deviceFeatures.shaderInt64 = true;

        vulkan12Features.storageBuffer8BitAccess = true;
        vulkan12Features.shaderInt8 = true;
        vulkan12Features.vulkanMemoryModel = true;

        vulkan13Features.synchronization2 = true;
        vulkan13Features.maintenance4 = true;
        vulkan13Features.pipelineCreationCacheControl = true;
        vulkan13Features.pNext = &vulkan12Features;

        tensorFeatures.tensors = true;
        tensorFeatures.shaderTensorAccess = true;
        tensorFeatures.tensorNonPacked = true;
        tensorFeatures.pNext = &vulkan13Features;

        dataGraphFeatures.dataGraph = true;
        dataGraphFeatures.dataGraphShaderModule = true;
        dataGraphFeatures.pNext = &tensorFeatures;

        std::vector<const char *> deviceExtensions = {VK_ARM_DATA_GRAPH_EXTENSION_NAME, VK_ARM_TENSORS_EXTENSION_NAME};
        device = vk::raii::Device(
            physicalDevice,
            {vk::DeviceCreateFlags(), queueCreateInfo, {}, deviceExtensions, &deviceFeatures, &dataGraphFeatures});
        queue = device.getQueue(queueFamilyIndex, 0);
    }

    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
    vk::raii::Queue queue{nullptr};
    uint32_t queueFamilyIndex = UINT32_MAX;

    vk::PhysicalDeviceVulkan12Features vulkan12Features;
    vk::PhysicalDeviceVulkan13Features vulkan13Features;
    vk::PhysicalDeviceTensorFeaturesARM tensorFeatures;
    vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeatures;
};

std::vector<uint32_t> assembleSecondMaxpoolSpirv() {
    std::ifstream templateFile(VGF_RUNTIME_MAXPOOL_8X8_TO_4X4_SPVASM);
    std::string spvasm((std::istreambuf_iterator<char>(templateFile)), {});
    replaceAll(spvasm, "INPUT_SET", "0");
    replaceAll(spvasm, "INPUT_BINDING", "0");
    replaceAll(spvasm, "OUTPUT_SET", "0");
    replaceAll(spvasm, "OUTPUT_BINDING", "1");

    return assembleSpirv(spvasm);
}

std::string makeTwoSegmentMaxpoolVgf() {
    const auto &firstCode = assembleMaxpoolSpirv("maxpool_16x16_to_8x8", {0, 0, 0, 1});
    const auto &secondCode = assembleSecondMaxpoolSpirv();
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
                               {firstOutputBinding}, {});

        const auto secondOutput =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 4, 4, 16}, {});
        const auto secondInputBinding = encoder.AddBindingSlot(0, firstOutput);
        const auto secondOutputBinding = encoder.AddBindingSlot(1, secondOutput);
        const auto secondDescriptorSet = encoder.AddDescriptorSetInfo({secondInputBinding, secondOutputBinding});
        encoder.AddSegmentInfo(secondModule, "second_graph_segment", {secondDescriptorSet}, {secondInputBinding},
                               {secondOutputBinding}, {});
    });
}

std::vector<uint32_t> assembleAddInt32BuffersSpirv() {
    std::ifstream templateFile(VGF_RUNTIME_ADD_INT32_BUFFERS_SPVASM);
    std::string spvasm((std::istreambuf_iterator<char>(templateFile)), {});
    return assembleSpirv(spvasm);
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
                               {firstInputBinding, secondInputBinding}, {outputBinding}, {}, {10, 1, 1});
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
    Tensor firstOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    Tensor secondOutputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 4, 4, 16});

    const auto firstInput = makeMaxpoolInput(firstInputTensor.shape, 7);
    firstInputTensor.write(firstInput);
    firstOutputTensor.fill(0, firstOutputTensor.numElements());
    secondOutputTensor.fill(0, secondOutputTensor.numElements());

    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);
    const auto firstBindings = vgf.getDescriptorBindings(0);
    ASSERT_EQ(firstBindings.size(), 2);
    session.bindTensor(firstInputTensor.tensor, firstBindings[0]);
    session.bindTensor(firstOutputTensor.tensor, firstBindings[1]);

    const auto secondBindings = vgf.getDescriptorBindings(1);
    ASSERT_EQ(secondBindings.size(), 2);
    session.bindTensor(firstOutputTensor.tensor, secondBindings[0]);
    session.bindTensor(secondOutputTensor.tensor, secondBindings[1]);

    session.configure();
    session.run();

    const auto firstExpected = expectedMaxpool(firstInput, firstInputTensor.shape);
    EXPECT_EQ(firstOutputTensor.read(firstOutputTensor.numElements()), firstExpected);
    EXPECT_EQ(secondOutputTensor.read(secondOutputTensor.numElements()),
              expectedMaxpool(firstExpected, firstOutputTensor.shape));
}
