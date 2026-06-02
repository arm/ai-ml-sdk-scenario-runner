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

#include <cstdint>
#include <filesystem>
#include <fstream>
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
            GTEST_SKIP() << "No Vulkan device with VK_ARM_data_graph and VK_ARM_tensors support";
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

} // namespace

TEST_F(VgfRuntimeFullTest, RunMaxpoolDataVGF) {
    const auto bytes = makeMaxpoolVgf();
    const VGF vgf(bytes.data(), bytes.size());

    Tensor inputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 16, 16, 16});
    Tensor outputTensor(physicalDevice, device, vk::Format::eR8Sint, {1, 8, 8, 16});
    Session session(physicalDevice, device, queueFamilyIndex, queue, vgf);

    const auto input = makeMaxpoolInput();
    inputTensor.write(input);
    outputTensor.fill(0, kMaxpoolOutputElements);

    const auto bindings = vgf.getDescriptorBindings(0);
    session.bindTensor(inputTensor.tensor, bindings[0]);
    session.bindTensor(outputTensor.tensor, bindings[1]);
    session.configure();
    session.run();

    EXPECT_EQ(outputTensor.read(kMaxpoolOutputElements), expectedMaxpool(input));
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

    const auto input = makeMaxpoolInput(0);
    inputTensor.write(input);
    outputTensor.fill(0, kMaxpoolOutputElements);

    const auto bindings = vgf.getDescriptorBindings(0);
    session.bindTensor(inputTensor.tensor, bindings[0]);
    session.bindTensor(outputTensor.tensor, bindings[1]);
    session.configure();
    session.run();

    EXPECT_EQ(outputTensor.read(kMaxpoolOutputElements), expectedMaxpool(input));
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

    const auto firstInput = makeMaxpoolInput(3);
    inputTensor.write(firstInput);
    outputTensor.fill(0, kMaxpoolOutputElements);
    session.run();
    EXPECT_EQ(outputTensor.read(kMaxpoolOutputElements), expectedMaxpool(firstInput));

    const auto secondInput = makeMaxpoolInput(41);
    inputTensor.write(secondInput);
    outputTensor.fill(0, kMaxpoolOutputElements);
    session.run();
    EXPECT_EQ(outputTensor.read(kMaxpoolOutputElements), expectedMaxpool(secondInput));
}
