/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "device_selection.hpp"

#include "scenario_runner/scenario_options.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;
using namespace mlsdk::scenariorunner::detail;
vk::QueueFamilyProperties queueFamily(vk::QueueFlags flags, uint32_t count = 1) {
    vk::QueueFamilyProperties properties;
    properties.queueFlags = flags;
    properties.queueCount = count;
    return properties;
}

vk::QueueFlags dataGraphQueueFlags(vk::QueueFlags additionalFlags = {}) {
    return vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eTransfer | additionalFlags;
}

DeviceCandidate candidate(vk::PhysicalDeviceType type, DeviceCapability capabilities,
                          std::vector<vk::QueueFamilyProperties> queueFamilies,
                          std::vector<uint32_t> opticalFlowQueueFamilyIndices = {}) {
    return {type, capabilities, std::move(queueFamilies), std::move(opticalFlowQueueFamilyIndices)};
}

TEST(DeviceRequirements, DgbOnlyDoesNotRequireUnrelatedCapabilities) {
    ScenarioBuildData buildData;
    DispatchDataGraphBinaryData dispatch{RawDataId{0}};
    dispatch.bindings.push_back({0, 0, ImageId{0}, std::nullopt, vk::DescriptorType::eStorageImage});
    buildData.commands.emplace_back(std::move(dispatch));

    const auto requirements = getDeviceRequirements(buildData, ScenarioOptions{});

    EXPECT_EQ(requirements.queueFlags, vk::QueueFlags{vk::QueueFlagBits::eDataGraphARM});
    EXPECT_EQ(requirements.capabilities, DeviceCapability::DataGraphBinaries);

    const std::vector<DeviceCandidate> candidates{
        candidate(vk::PhysicalDeviceType::eIntegratedGpu, DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags())}),
    };
    EXPECT_NO_THROW(static_cast<void>(selectDevice(candidates, requirements)));
}

TEST(DeviceRequirements, MixedScenarioCombinesRequirements) {
    ScenarioBuildData buildData;
    buildData.resources.addTensor(TensorInfo{});
    buildData.commands.emplace_back(DispatchDataGraphBinaryData{RawDataId{0}});
    buildData.commands.emplace_back(DispatchDataGraphData{ShaderId{0}});
    buildData.commands.emplace_back(DispatchComputeData{ShaderId{1}});
    buildData.commands.emplace_back(DispatchFragmentData{ShaderId{2}, ShaderId{3}});

    const TypedBinding opticalFlowBinding{0, 0, ImageId{0}, std::nullopt, vk::DescriptorType::eStorageImage};
    buildData.commands.emplace_back(
        DispatchOpticalFlowData{opticalFlowBinding, opticalFlowBinding, opticalFlowBinding});

    ScenarioOptions options;
    options.neuralStatisticsDumpDir = "statistics";
    const auto requirements = getDeviceRequirements(buildData, options);

    EXPECT_EQ(requirements.queueFlags, vk::QueueFlags{vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eCompute |
                                                      vk::QueueFlagBits::eGraphics});
    EXPECT_EQ(requirements.capabilities,
              DeviceCapability::DataGraphBinaries | DeviceCapability::DataGraphShaderModules |
                  DeviceCapability::Tensors | DeviceCapability::DataGraphOpticalFlow |
                  DeviceCapability::DataGraphNeuralAcceleratorStatistics | DeviceCapability::DynamicRendering);
}

TEST(DeviceSelection, RejectsUnsupportedPreferredDevices) {
    DeviceRequirements requirements;
    requirements.queueFlags = vk::QueueFlagBits::eDataGraphARM;
    requirements.capabilities = DeviceCapability::DataGraphBinaries | DeviceCapability::DataGraphOpticalFlow;

    const std::vector<DeviceCandidate> candidates{
        candidate(vk::PhysicalDeviceType::eDiscreteGpu, DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags())}),
        candidate(vk::PhysicalDeviceType::eIntegratedGpu, DeviceCapability::DataGraphOpticalFlow,
                  {queueFamily(dataGraphQueueFlags())}),
        candidate(vk::PhysicalDeviceType::eCpu,
                  DeviceCapability::DataGraphBinaries | DeviceCapability::DataGraphOpticalFlow,
                  {queueFamily(dataGraphQueueFlags())}, {0}),
    };

    const auto selection = selectDevice(candidates, requirements);

    EXPECT_EQ(selection.deviceIndex, 2U);
    EXPECT_EQ(selection.queueFamilyIndex, 0U);
}

TEST(DeviceSelection, SelectsQueueFamilySupportingOpticalFlow) {
    DeviceRequirements requirements;
    requirements.queueFlags = vk::QueueFlagBits::eDataGraphARM;
    requirements.capabilities = DeviceCapability::DataGraphOpticalFlow;

    const std::vector<DeviceCandidate> candidates{
        candidate(vk::PhysicalDeviceType::eDiscreteGpu, DeviceCapability::DataGraphOpticalFlow,
                  {queueFamily(dataGraphQueueFlags()), queueFamily(dataGraphQueueFlags())}, {1}),
    };

    const auto selection = selectDevice(candidates, requirements);

    EXPECT_EQ(selection.deviceIndex, 0U);
    EXPECT_EQ(selection.queueFamilyIndex, 1U);
}

TEST(DeviceSelection, RejectsZeroCountAndSplitQueueFamilies) {
    DeviceRequirements requirements;
    requirements.queueFlags = vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eCompute;
    requirements.capabilities = DeviceCapability::DataGraphBinaries;

    const std::vector<DeviceCandidate> candidates{
        candidate(vk::PhysicalDeviceType::eDiscreteGpu, DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags(vk::QueueFlagBits::eCompute), 0)}),
        candidate(vk::PhysicalDeviceType::eIntegratedGpu, DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags()), queueFamily(vk::QueueFlagBits::eCompute)}),
        candidate(vk::PhysicalDeviceType::eCpu, DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags(vk::QueueFlagBits::eCompute))}),
    };

    const auto selection = selectDevice(candidates, requirements);

    EXPECT_EQ(selection.deviceIndex, 2U);
    EXPECT_EQ(selection.queueFamilyIndex, 0U);
}

TEST(DeviceSelection, ReportsClearReasonWhenNoDeviceIsSuitable) {
    DeviceRequirements requirements;
    requirements.queueFlags = vk::QueueFlagBits::eDataGraphARM;
    requirements.capabilities = DeviceCapability::DataGraphBinaries;

    try {
        static_cast<void>(selectDevice({candidate(vk::PhysicalDeviceType::eDiscreteGpu, DeviceCapability::DataGraph,
                                                  {queueFamily(dataGraphQueueFlags())})},
                                       requirements));
        FAIL() << "Expected device selection to fail";
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("supports all capabilities required by the scenario"), std::string::npos);
    }

    try {
        static_cast<void>(selectDevice({candidate(vk::PhysicalDeviceType::eCpu, DeviceCapability::DataGraphBinaries,
                                                  {queueFamily(dataGraphQueueFlags(), 0)})},
                                       requirements));
        FAIL() << "Expected device selection to fail";
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("nonempty queue family supporting all required queue flags and data graph operations"),
                  std::string::npos);
    }
}

} // namespace
