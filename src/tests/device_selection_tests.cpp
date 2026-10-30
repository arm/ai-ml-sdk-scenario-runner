/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "device_selection.hpp"

#include "scenario_runner/scenario_options.hpp"

#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include "vgf-utils/temp_folder.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;
using namespace mlsdk::scenariorunner::detail;
namespace vgflib = mlsdk::vgflib;

std::shared_ptr<const VgfView> createVgf(TempFolder &tempFolder, const std::string &filename,
                                         const std::vector<vgflib::ModuleType> &moduleTypes, bool hasTensorIntermediate,
                                         std::optional<int64_t> constantSparsityDimension = std::nullopt,
                                         std::optional<vk::DescriptorType> imageDescriptorType = std::nullopt,
                                         std::optional<uint32_t> imageSegmentIndex = std::nullopt) {
    auto encoder = vgflib::CreateEncoder(123);

    std::vector<vgflib::DescriptorSetInfoRef> descriptorSets;
    std::vector<vgflib::BindingSlotRef> segmentOutputs;
    if (hasTensorIntermediate) {
        const auto tensor = encoder->AddIntermediateResource(vgflib::ToDescriptorType(VK_DESCRIPTOR_TYPE_TENSOR_ARM),
                                                             vgflib::ToFormatType(VK_FORMAT_R8_SINT), {1}, {});
        const auto binding = encoder->AddBindingSlot(0, tensor);
        descriptorSets.emplace_back(encoder->AddDescriptorSetInfo({binding}));
        segmentOutputs.emplace_back(binding);
    }

    std::optional<vgflib::BindingSlotRef> imageBinding;
    std::optional<vgflib::DescriptorSetInfoRef> imageDescriptorSet;
    if (imageDescriptorType) {
        const auto image = encoder->AddIntermediateResource(
            vgflib::ToDescriptorType(static_cast<VkDescriptorType>(*imageDescriptorType)),
            vgflib::ToFormatType(VK_FORMAT_R8_UINT), {1, 2, 2, 1}, {});
        imageBinding.emplace(encoder->AddBindingSlot(0, image));
        imageDescriptorSet.emplace(encoder->AddDescriptorSetInfo({*imageBinding}));
    }

    std::vector<vgflib::GraphConstantBindingRef> constantBindings;
    if (constantSparsityDimension) {
        const auto resource = encoder->AddConstantResource(vgflib::ToFormatType(VK_FORMAT_R8_SINT), {4}, {});
        const std::vector<uint8_t> data{1, 0, 2, 0};
        const auto constant = encoder->AddConstant(resource, data.data(), data.size(), *constantSparsityDimension);
        // NOLINTNEXTLINE(modernize-use-emplace)
        constantBindings.push_back({0, constant});
    }

    encoder->AddModelSequenceInputsOutputs({}, {}, {}, {});
    for (size_t index = 0; index < moduleTypes.size(); ++index) {
        const auto module = encoder->AddModule(moduleTypes[index], "module_" + std::to_string(index), "main");
        auto segmentDescriptorSets = descriptorSets;
        auto outputs = segmentOutputs;
        if (imageDescriptorSet && (!imageSegmentIndex || index == *imageSegmentIndex)) {
            segmentDescriptorSets.emplace_back(*imageDescriptorSet);
            outputs.emplace_back(*imageBinding);
        }
        encoder->AddSegmentInfo(module, "segment_" + std::to_string(index), segmentDescriptorSets, {}, outputs,
                                constantBindings, {1, 1, 1});
    }
    encoder->Finish();

    const std::filesystem::path path = tempFolder.relative(filename);
    {
        std::ofstream output(path, std::ios::binary);
        encoder->WriteTo(output);
    }
    return loadVgfView(path.string());
}

vk::QueueFamilyProperties queueFamily(vk::QueueFlags flags, uint32_t count = 1) {
    vk::QueueFamilyProperties properties;
    properties.queueFlags = flags;
    properties.queueCount = count;
    return properties;
}

vk::QueueFlags dataGraphQueueFlags(vk::QueueFlags additionalFlags = {}) {
    return vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eTransfer | additionalFlags;
}

DeviceCandidate candidate(vk::PhysicalDeviceType type, DeviceCapabilities capabilities,
                          std::vector<vk::QueueFamilyProperties> queueFamilies,
                          std::vector<uint32_t> opticalFlowQueueFamilyIndices = {}) {
    return {type, capabilities, std::move(queueFamilies), std::move(opticalFlowQueueFamilyIndices)};
}

constexpr DeviceCapability graphCapabilities[] = {
    DeviceCapability::DataGraphShaderModules, DeviceCapability::DataGraphBinaries,
    DeviceCapability::DataGraphOpticalFlow,   DeviceCapability::DataGraphNeuralAcceleratorStatistics,
    DeviceCapability::DataGraphImageLayouts,
};

TEST(DeviceCapabilities, SupportMasksDoNotImplyDependencies) {
    for (const auto capability : graphCapabilities) {
        SCOPED_TRACE(static_cast<uint32_t>(capability));
        EXPECT_FALSE(hasCapabilities(capability, DeviceCapability::DataGraph));
        EXPECT_TRUE(hasCapabilities(capability, capability));
    }
    EXPECT_FALSE(hasCapabilities(DeviceCapability::Tensors, DeviceCapability::ShaderTensorAccess));
    EXPECT_FALSE(hasCapabilities(DeviceCapability::ShaderTensorAccess, DeviceCapability::Tensors));
    EXPECT_FALSE(hasCapabilities(DeviceCapability::DataGraph, DeviceCapability::DataGraphShaderModules));
}

TEST(DeviceRequirements, ExpandsGraphDependenciesWithoutChangingOtherRequirements) {
    for (const auto capability : graphCapabilities) {
        SCOPED_TRACE(static_cast<uint32_t>(capability));
        DeviceRequirements requirements;
        requirements.queueFlags = vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer;
        requirements.capabilities = capability | DeviceCapability::DynamicRendering;

        const auto expanded = expandDeviceRequirements(requirements);

        EXPECT_EQ(expanded.capabilities, requirements.capabilities | DeviceCapability::DataGraph);
        EXPECT_EQ(expanded.queueFlags, requirements.queueFlags);
        EXPECT_EQ(expandDeviceRequirements(expanded).capabilities, expanded.capabilities);
        EXPECT_EQ(requirements.capabilities, capability | DeviceCapability::DynamicRendering);
    }
}

TEST(DeviceRequirements, DgbOnlyDoesNotRequireUnrelatedCapabilities) {
    ScenarioBuildData buildData;
    DispatchDataGraphBinaryData dispatch{RawDataId{0}};
    dispatch.bindings.emplace_back(TypedBinding{0, 0, ImageId{0}, std::nullopt, vk::DescriptorType::eStorageImage});
    buildData.commands.emplace_back(std::move(dispatch));

    const auto requirements = getDeviceRequirements(buildData, ScenarioOptions{});

    EXPECT_EQ(requirements.queueFlags, vk::QueueFlags{vk::QueueFlagBits::eDataGraphARM});
    EXPECT_EQ(requirements.capabilities, DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries);

    const std::vector<DeviceCandidate> candidates{
        candidate(vk::PhysicalDeviceType::eIntegratedGpu,
                  DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries,
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
              DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries |
                  DeviceCapability::DataGraphShaderModules | DeviceCapability::Tensors |
                  DeviceCapability::ShaderTensorAccess | DeviceCapability::DataGraphOpticalFlow |
                  DeviceCapability::DataGraphNeuralAcceleratorStatistics | DeviceCapability::DynamicRendering);
}

TEST(DeviceRequirements, VgfUsesOnlyCapabilitiesRequiredByItsContents) {
    TempFolder tempFolder("device_requirements");

    ScenarioBuildData computeBuildData;
    const auto computeVgf = computeBuildData.resources.addVgf(
        VgfInfo{"compute VGF", createVgf(tempFolder, "compute.vgf", {vgflib::ModuleType::COMPUTE}, false), 0, {}});
    computeBuildData.commands.emplace_back(DispatchVgfData{computeVgf});
    const auto computeRequirements = getDeviceRequirements(computeBuildData, ScenarioOptions{});
    EXPECT_EQ(computeRequirements.queueFlags, vk::QueueFlags{vk::QueueFlagBits::eCompute});
    EXPECT_EQ(computeRequirements.capabilities, DeviceCapabilities{});

    ScenarioBuildData mixedBuildData;
    const auto mixedVgf = mixedBuildData.resources.addVgf(
        VgfInfo{"mixed VGF",
                createVgf(tempFolder, "mixed.vgf", {vgflib::ModuleType::GRAPH, vgflib::ModuleType::COMPUTE}, true),
                0,
                {}});
    mixedBuildData.commands.emplace_back(DispatchVgfData{mixedVgf});
    const auto mixedRequirements = getDeviceRequirements(mixedBuildData, ScenarioOptions{});
    EXPECT_EQ(mixedRequirements.queueFlags,
              vk::QueueFlags{vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eCompute});
    EXPECT_EQ(mixedRequirements.capabilities, DeviceCapability::DataGraph | DeviceCapability::DataGraphShaderModules |
                                                  DeviceCapability::Tensors | DeviceCapability::ShaderTensorAccess);
}

TEST(DeviceRequirements, VgfBinaryGraphDoesNotRequireShaderModules) {
    TempFolder tempFolder("device_requirements_binary_vgf");
    ScenarioBuildData buildData;
    const auto vgf = buildData.resources.addVgf(VgfInfo{
        "mixed VGF",
        createVgf(tempFolder, "mixed_binary.vgf", {vgflib::ModuleType::GRAPH, vgflib::ModuleType::COMPUTE}, false),
        0,
        {}});
    DispatchVgfData dispatch{vgf};
    dispatch.dataGraphBinaries.emplace(0, ExternalDataGraphBinary{RawDataId{0}});
    buildData.commands.emplace_back(std::move(dispatch));

    const auto requirements = getDeviceRequirements(buildData, ScenarioOptions{});
    EXPECT_EQ(requirements.queueFlags, vk::QueueFlags{vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eCompute});
    EXPECT_EQ(requirements.capabilities, DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries);
}

TEST(DeviceSelection, RejectsUnsupportedPreferredDevices) {
    DeviceRequirements requirements;
    requirements.queueFlags = vk::QueueFlagBits::eDataGraphARM;
    requirements.capabilities = DeviceCapability::DataGraphBinaries | DeviceCapability::DataGraphOpticalFlow;

    const std::vector<DeviceCandidate> candidates{
        candidate(vk::PhysicalDeviceType::eDiscreteGpu,
                  DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags())}),
        candidate(vk::PhysicalDeviceType::eIntegratedGpu,
                  DeviceCapability::DataGraph | DeviceCapability::DataGraphOpticalFlow,
                  {queueFamily(dataGraphQueueFlags())}),
        candidate(vk::PhysicalDeviceType::eCpu,
                  DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries |
                      DeviceCapability::DataGraphOpticalFlow,
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
        candidate(vk::PhysicalDeviceType::eDiscreteGpu,
                  DeviceCapability::DataGraph | DeviceCapability::DataGraphOpticalFlow,
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
        candidate(vk::PhysicalDeviceType::eDiscreteGpu,
                  DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags(vk::QueueFlagBits::eCompute), 0)}),
        candidate(vk::PhysicalDeviceType::eIntegratedGpu,
                  DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries,
                  {queueFamily(dataGraphQueueFlags()), queueFamily(vk::QueueFlagBits::eCompute)}),
        candidate(vk::PhysicalDeviceType::eCpu, DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries,
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
        static_cast<void>(selectDevice(
            {candidate(vk::PhysicalDeviceType::eCpu, DeviceCapability::DataGraph | DeviceCapability::DataGraphBinaries,
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
