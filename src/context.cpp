/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "context.hpp"

#include "scenario_runner/scenario_options.hpp"

#include "device_selection.hpp"
#include "logging.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <vulkan/vulkan_beta.h>

namespace mlsdk::scenariorunner {
namespace {
bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, const std::string &extensionName,
                  const std::vector<std::string> &disabledExtensions = {}) {
    if (std::find(disabledExtensions.begin(), disabledExtensions.end(), extensionName) != disabledExtensions.end()) {
        return false;
    }
    return std::any_of(extensions.cbegin(), extensions.cend(), [extensionName](const auto &ext) {
        return std::string_view{static_cast<const char *>(ext.extensionName)} == extensionName;
    });
}

std::string formatVersion(uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + '.' + std::to_string(VK_API_VERSION_MINOR(version)) + '.' +
           std::to_string(VK_API_VERSION_PATCH(version));
}

std::string_view formatFeature(VkBool32 supported) { return supported == VK_TRUE ? "true" : "false"; }

detail::DeviceCandidate queryDeviceCandidate(const vk::raii::PhysicalDevice &device,
                                             const ScenarioOptions &scenarioOptions) {
    // Fetch the properties used to rank the device and select its queue.
    const auto properties = device.getProperties();
    detail::DeviceCandidate candidate;
    candidate.type = properties.deviceType;

    // Fetch the extensions needed by scenario capabilities.
    const auto extensions = device.enumerateDeviceExtensionProperties(nullptr);
    const bool dataGraphExtension =
        hasExtension(extensions, VK_ARM_DATA_GRAPH_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    const bool tensorsExtension =
        hasExtension(extensions, VK_ARM_TENSORS_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    const bool opticalFlowExtension =
        hasExtension(extensions, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    const bool neuralAcceleratorStatisticsExtension = hasExtension(
        extensions, VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME, scenarioOptions.disabledExtensions);
#ifdef VK_ARM_data_graph_binaries
    const bool dataGraphBinariesExtension =
        hasExtension(extensions, VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME, scenarioOptions.disabledExtensions);
#endif

    // Fetch the features needed by scenario capabilities.
    const auto features =
        device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                            vk::PhysicalDeviceDataGraphFeaturesARM, vk::PhysicalDeviceTensorFeaturesARM,
                            vk::PhysicalDeviceDataGraphOpticalFlowFeaturesARM,
                            vk::PhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM
#ifdef VK_ARM_data_graph_binaries
                            ,
                            vk::PhysicalDeviceDataGraphBinariesFeaturesARM
#endif
                            >();
    const auto &dataGraphFeatures = features.template get<vk::PhysicalDeviceDataGraphFeaturesARM>();
    const auto &tensorFeatures = features.template get<vk::PhysicalDeviceTensorFeaturesARM>();

    // Record capabilities supported by their required extensions and features.
    if (dataGraphExtension && dataGraphFeatures.dataGraph) {
        candidate.capabilities |= detail::DeviceCapability::DataGraph;
        if (dataGraphFeatures.dataGraphShaderModule) {
            candidate.capabilities |= detail::DeviceCapability::DataGraphShaderModules;
        }
    }
    if (tensorsExtension) {
        if (tensorFeatures.shaderTensorAccess) {
            candidate.capabilities |= detail::DeviceCapability::ShaderTensorAccess;
        }
        if (tensorFeatures.tensors && tensorFeatures.tensorNonPacked) {
            candidate.capabilities |= detail::DeviceCapability::Tensors;
        }
    }
    if (detail::hasCapabilities(candidate.capabilities, detail::DeviceCapability::DataGraph) && opticalFlowExtension) {
        candidate.capabilities |= detail::DeviceCapability::DataGraphImageLayouts;
        if (features.template get<vk::PhysicalDeviceDataGraphOpticalFlowFeaturesARM>().dataGraphOpticalFlow) {
            candidate.capabilities |= detail::DeviceCapability::DataGraphOpticalFlow;
        }
    }
    if (detail::hasCapabilities(candidate.capabilities, detail::DeviceCapability::DataGraph) &&
        neuralAcceleratorStatisticsExtension &&
        features.template get<vk::PhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM>()
            .dataGraphNeuralAcceleratorStatistics) {
        candidate.capabilities |= detail::DeviceCapability::DataGraphNeuralAcceleratorStatistics;
    }
    if (features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering) {
        candidate.capabilities |= detail::DeviceCapability::DynamicRendering;
    }
#ifdef VK_ARM_data_graph_binaries
    if (detail::hasCapabilities(candidate.capabilities, detail::DeviceCapability::DataGraph) &&
        dataGraphBinariesExtension &&
        features.template get<vk::PhysicalDeviceDataGraphBinariesFeaturesARM>().dataGraphBinaries) {
        candidate.capabilities |= detail::DeviceCapability::DataGraphBinaries;
    }
#endif

    // Fetch the queue-family properties and data graph operations used for queue selection.
    candidate.queueFamilies = device.getQueueFamilyProperties();
    for (uint32_t index = 0; index < static_cast<uint32_t>(candidate.queueFamilies.size()); ++index) {
        const auto &queueFamily = candidate.queueFamilies[index];
        if (detail::hasCapabilities(candidate.capabilities, detail::DeviceCapability::DataGraphOpticalFlow) &&
            (queueFamily.queueFlags & vk::QueueFlagBits::eDataGraphARM) != vk::QueueFlags{}) {
            const auto dataGraphProperties = device.getQueueFamilyDataGraphPropertiesARM(index);
            const bool supportsOpticalFlow =
                std::any_of(dataGraphProperties.begin(), dataGraphProperties.end(), [](const auto &properties) {
                    return properties.operation.operationType ==
                           vk::PhysicalDeviceDataGraphOperationTypeARM::eOpticalFlow;
                });
            if (supportsOpticalFlow) {
                candidate.opticalFlowQueueFamilyIndices.push_back(index);
            }
        }
    }
    return candidate;
}
} // namespace

Context::Context(const ScenarioOptions &scenarioOptions, const detail::DeviceRequirements &requestedRequirements)
    : _gpuDebugMarkersEnabled(scenarioOptions.enableGPUDebugMarkers),
      _sessionMemoryDumpEnabled(!scenarioOptions.sessionRAMsDumpDir.empty()),
      _robustnessFeaturesEnabled(scenarioOptions.enableRobustnessFeatures) {
    const auto requirements = detail::expandDeviceRequirements(requestedRequirements);
    // Create instance
    const vk::ApplicationInfo appInfo("Scenario-Runner", 1, nullptr, 0, VK_API_VERSION_1_3);

    std::vector<const char *> enabledLayers;
    std::vector<const char *> enabledExtensions;

    if (_gpuDebugMarkersEnabled) {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    vk::InstanceCreateFlags flags{};
    const auto instanceExtensions = _ctx.enumerateInstanceExtensionProperties();
    if (hasExtension(instanceExtensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
                     scenarioOptions.disabledExtensions)) {
        enabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }

    const vk::InstanceCreateInfo instanceInfo(flags, &appInfo, static_cast<uint32_t>(enabledLayers.size()),
                                              enabledLayers.data(), static_cast<uint32_t>(enabledExtensions.size()),
                                              enabledExtensions.data());

    _instance = vk::raii::Instance(_ctx, instanceInfo);

    // Create physical device
    auto physicalDevices = vk::raii::PhysicalDevices(_instance);

    std::vector<detail::DeviceCandidate> candidates;
    candidates.reserve(physicalDevices.size());
    for (const auto &candidateDevice : physicalDevices) {
        candidates.emplace_back(queryDeviceCandidate(candidateDevice, scenarioOptions));
    }
    const auto selection = detail::selectDevice(candidates, requirements);
    _physicalDev = physicalDevices[selection.deviceIndex];
    _familyQueueIdx = selection.queueFamilyIndex;

    vk::PhysicalDeviceProperties properties = _physicalDev.getProperties();
    std::string deviceName = std::string(properties.deviceName.data());
    std::string deviceType = vk::to_string(properties.deviceType);
    std::ostringstream vendorID;
    vendorID << std::hex << std::setw(4) << std::setfill('0') << properties.vendorID << std::dec;

    mlsdk::logging::info("Device: " + deviceName + ", Type: " + deviceType + ", Vendor: 0x" + vendorID.str());

    // Get device capabilities
    const std::vector<vk::ExtensionProperties> &extensions = _physicalDev.enumerateDeviceExtensionProperties(nullptr);

    _optionals.data_graph = detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::DataGraph);
    _optionals.custom_border_color =
        hasExtension(extensions, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.mark_boundary =
        hasExtension(extensions, VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.neural_accelerator_statistics = detail::hasCapabilities(
        requirements.capabilities, detail::DeviceCapability::DataGraphNeuralAcceleratorStatistics);
    _optionals.maintenance4 =
        hasExtension(extensions, VK_KHR_MAINTENANCE_4_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.maintenance5 =
        hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.deferred_operation =
        hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.replicated_composites = hasExtension(extensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME,
                                                    scenarioOptions.disabledExtensions);
    _optionals.shader_bfloat16 =
        hasExtension(extensions, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.shader_float8 =
        hasExtension(extensions, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.optical_flow =
        detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::DataGraphOpticalFlow);
    _optionals.robustness2 =
        hasExtension(extensions, VK_KHR_ROBUSTNESS_2_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.descriptor_indexing =
        hasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.pipeline_robustness =
        hasExtension(extensions, VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.portability_subset =
        hasExtension(extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, scenarioOptions.disabledExtensions);
#ifdef VK_ARM_data_graph_binaries
    _optionals.data_graph_binaries =
        detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::DataGraphBinaries);
#endif

    // Create device
    const float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), _familyQueueIdx, 1,
                                                          &queuePriority);

    void *prevPNext = nullptr;
    vk::PhysicalDeviceCustomBorderColorFeaturesEXT customBorderColorFeatures{true};
    if (_optionals.custom_border_color) {
        prevPNext = &customBorderColorFeatures;
    }

    vk::PhysicalDeviceFrameBoundaryFeaturesEXT frameBoundaryFeatures{true, prevPNext};
    if (_optionals.mark_boundary) {
        prevPNext = &frameBoundaryFeatures;
    }

    vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT physicalDevReplicateCompositesFeat{true, prevPNext};
    if (_optionals.replicated_composites) {
        prevPNext = &physicalDevReplicateCompositesFeat;
    }

    const auto availableFeatures =
        _physicalDev
            .getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                          vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceShaderBfloat16FeaturesKHR,
                          vk::PhysicalDeviceShaderFloat8FeaturesEXT, vk::PhysicalDeviceRobustness2FeaturesKHR,
                          vk::PhysicalDeviceDescriptorIndexingFeatures, vk::PhysicalDevicePipelineRobustnessFeatures>();

    const auto &availableCoreFeatures = availableFeatures.template get<vk::PhysicalDeviceFeatures2>().features;
    const auto &[available11Features, available12Features, availableBfloat16, availableFloat8] =
        availableFeatures
            .template get<vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features,
                          vk::PhysicalDeviceShaderBfloat16FeaturesKHR, vk::PhysicalDeviceShaderFloat8FeaturesEXT>();

    std::ostringstream deviceCapabilities;
    deviceCapabilities << "Vulkan device shader capabilities: apiVersion=" << formatVersion(properties.apiVersion)
                       << ", driverVersion=" << properties.driverVersion
                       << ", shaderInt64=" << formatFeature(availableCoreFeatures.shaderInt64)
                       << ", shaderInt8=" << formatFeature(available12Features.shaderInt8)
                       << ", shaderFloat16=" << formatFeature(available12Features.shaderFloat16)
                       << ", storageBuffer8BitAccess=" << formatFeature(available12Features.storageBuffer8BitAccess)
                       << ", storagePushConstant16=" << formatFeature(available11Features.storagePushConstant16)
                       << ", bufferDeviceAddress=" << formatFeature(available12Features.bufferDeviceAddress);
    mlsdk::logging::info(deviceCapabilities.str());

    const auto &availableRobustness2 = availableFeatures.template get<vk::PhysicalDeviceRobustness2FeaturesKHR>();
    const auto &availablePipelineRobustness =
        availableFeatures.template get<vk::PhysicalDevicePipelineRobustnessFeatures>();
    if (!_optionals.neural_accelerator_statistics &&
        (scenarioOptions.shouldDumpNeuralDebugDatabase() || scenarioOptions.shouldDumpNeuralStatistics())) {
        throw std::runtime_error("Neural Accelerator dump requested, but "
                                 "VK_ARM_data_graph_neural_accelerator_statistics is not supported or enabled");
    }
    if (_optionals.pipeline_robustness && !availablePipelineRobustness.pipelineRobustness) {
        mlsdk::logging::warning("VK_EXT_pipeline_robustness extension is present, but "
                                "pipelineRobustness feature is not supported. "
                                "Disabling pipeline robustness support.");
        _optionals.pipeline_robustness = false;
    }
    const bool enableRobustness2 = scenarioOptions.enableRobustnessFeatures && _optionals.robustness2;
    const bool enableDescriptorIndexing = scenarioOptions.enableRobustnessFeatures && _optionals.descriptor_indexing &&
                                          available12Features.descriptorIndexing;
    const bool enablePipelineRobustness = scenarioOptions.enableRobustnessFeatures && _optionals.pipeline_robustness;

    vk::PhysicalDeviceVulkan11Features physicalDev11Feat;
    physicalDev11Feat.storageBuffer16BitAccess = available11Features.storageBuffer16BitAccess;
    physicalDev11Feat.uniformAndStorageBuffer16BitAccess = available11Features.uniformAndStorageBuffer16BitAccess;
    physicalDev11Feat.storagePushConstant16 = available11Features.storagePushConstant16;
    physicalDev11Feat.pNext = prevPNext;

    vk::PhysicalDeviceVulkan12Features physicalDev2Feat;
    physicalDev2Feat.hostQueryReset = true;
    physicalDev2Feat.storageBuffer8BitAccess = true;
    physicalDev2Feat.uniformAndStorageBuffer8BitAccess = available12Features.uniformAndStorageBuffer8BitAccess;
    physicalDev2Feat.shaderInt8 = true;
    physicalDev2Feat.shaderFloat16 = available12Features.shaderFloat16;
    physicalDev2Feat.vulkanMemoryModel = true;
    physicalDev2Feat.vulkanMemoryModelDeviceScope = available12Features.vulkanMemoryModelDeviceScope;
    if (enableDescriptorIndexing) {
        physicalDev2Feat.descriptorIndexing = true;
    }
    physicalDev2Feat.pNext = &physicalDev11Feat;

    vk::PhysicalDeviceVulkan13Features physicalDev3Feat;
    physicalDev3Feat.synchronization2 = true;
    physicalDev3Feat.maintenance4 = true;
    physicalDev3Feat.pipelineCreationCacheControl = true;
    physicalDev3Feat.dynamicRendering =
        detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::DynamicRendering);
    physicalDev3Feat.pNext = &physicalDev2Feat;

    void *featureChain = &physicalDev3Feat;

    vk::PhysicalDeviceTensorFeaturesARM tensorFeat{};
    if (detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::ShaderTensorAccess)) {
        const bool tensorResources =
            detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::Tensors);
        tensorFeat.tensorNonPacked = tensorResources;
        tensorFeat.shaderTensorAccess = true;
        tensorFeat.tensors = tensorResources;
        tensorFeat.pNext = featureChain;
        featureChain = &tensorFeat;
    }

    vk::PhysicalDeviceShaderFloat8FeaturesEXT float8Feat{};
    if (_optionals.shader_float8) {
        float8Feat.shaderFloat8 = availableFloat8.shaderFloat8;
        float8Feat.pNext = featureChain;
        featureChain = &float8Feat;
    }

    vk::PhysicalDeviceDataGraphOpticalFlowFeaturesARM dataGraphOpticalFlowFeat{};
    if (_optionals.optical_flow) {
        dataGraphOpticalFlowFeat.dataGraphOpticalFlow = true;
        dataGraphOpticalFlowFeat.pNext = featureChain;
        featureChain = &dataGraphOpticalFlowFeat;
    }

    vk::PhysicalDeviceDataGraphNeuralAcceleratorStatisticsFeaturesARM dataGraphNeuralAcceleratorStatisticsFeat{};
    if (_optionals.neural_accelerator_statistics) {
        dataGraphNeuralAcceleratorStatisticsFeat.dataGraphNeuralAcceleratorStatistics = true;
        dataGraphNeuralAcceleratorStatisticsFeat.pNext = featureChain;
        featureChain = &dataGraphNeuralAcceleratorStatisticsFeat;
    }

    vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeat{};
    if (_optionals.data_graph) {
        dataGraphFeat.dataGraph = true;
        dataGraphFeat.dataGraphShaderModule =
            detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::DataGraphShaderModules);
        dataGraphFeat.pNext = featureChain;
        featureChain = &dataGraphFeat;
    }

    vk::PhysicalDeviceShaderBfloat16FeaturesKHR bfloat16Feat{};
    if (_optionals.shader_bfloat16) {
        bfloat16Feat.shaderBFloat16Type = availableBfloat16.shaderBFloat16Type;
        bfloat16Feat.pNext = featureChain;
        featureChain = &bfloat16Feat;
    }

    vk::PhysicalDeviceRobustness2FeaturesKHR robustness2Feat{};
    if (enableRobustness2) {
        robustness2Feat.robustBufferAccess2 = availableRobustness2.robustBufferAccess2;
        robustness2Feat.robustImageAccess2 = availableRobustness2.robustImageAccess2;
        robustness2Feat.nullDescriptor = availableRobustness2.nullDescriptor;
        robustness2Feat.pNext = featureChain;
        featureChain = &robustness2Feat;
    }

    vk::PhysicalDevicePipelineRobustnessFeatures pipelineRobustnessFeat{};
    if (enablePipelineRobustness) {
        pipelineRobustnessFeat.pipelineRobustness = availablePipelineRobustness.pipelineRobustness;
        pipelineRobustnessFeat.pNext = featureChain;
        featureChain = &pipelineRobustnessFeat;
    }

#ifdef VK_ARM_data_graph_binaries
    vk::PhysicalDeviceDataGraphBinariesFeaturesARM dataGraphBinariesFeat{};
    if (_optionals.data_graph_binaries) {
        dataGraphBinariesFeat.dataGraphBinaries = true;
        dataGraphBinariesFeat.pNext = featureChain;
        featureChain = &dataGraphBinariesFeat;

        using DgbProperties = vk::PhysicalDeviceDataGraphBinariesPropertiesARM;
        const auto binaryProperties = _physicalDev.getProperties2<vk::PhysicalDeviceProperties2, DgbProperties>();
        const auto alignment = binaryProperties.template get<DgbProperties>().dataGraphPipelineBinaryDataAlignment;
        if (alignment == 0 || alignment > std::numeric_limits<size_t>::max() || (alignment & (alignment - 1)) != 0) {
            throw std::runtime_error("Invalid dataGraphPipelineBinaryDataAlignment reported by the device");
        }
        _dataGraphPipelineBinaryDataAlignment = static_cast<size_t>(alignment);
    }
#endif

    vk::PhysicalDeviceFeatures deviceFeat;
    deviceFeat.shaderInt16 = true;
    deviceFeat.shaderInt64 = true;
    deviceFeat.fragmentStoresAndAtomics = availableCoreFeatures.fragmentStoresAndAtomics;

    std::vector<const char *> vulkanDeviceExtensions;
    if (_optionals.data_graph) {
        vulkanDeviceExtensions.push_back(VK_ARM_DATA_GRAPH_EXTENSION_NAME);
    }
    if (detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::ShaderTensorAccess)) {
        vulkanDeviceExtensions.push_back(VK_ARM_TENSORS_EXTENSION_NAME);
    }

    if (_optionals.custom_border_color) {
        vulkanDeviceExtensions.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    }
    if (_optionals.mark_boundary) {
        vulkanDeviceExtensions.push_back(VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME);
    }
    if (_optionals.neural_accelerator_statistics) {
        vulkanDeviceExtensions.push_back(VK_ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_EXTENSION_NAME);
    }
    if (_optionals.maintenance4) {
        vulkanDeviceExtensions.push_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    }
    if (_optionals.maintenance5) {
        vulkanDeviceExtensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    }
    if (_optionals.deferred_operation) {
        vulkanDeviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }
    if (_optionals.replicated_composites) {
        vulkanDeviceExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
    }
    if (_optionals.shader_bfloat16) {
        vulkanDeviceExtensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    }
    if (_optionals.shader_float8) {
        vulkanDeviceExtensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    }
    if (_optionals.optical_flow ||
        detail::hasCapabilities(requirements.capabilities, detail::DeviceCapability::DataGraphImageLayouts)) {
        vulkanDeviceExtensions.push_back(VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME);
    }
    if (enableRobustness2) {
        vulkanDeviceExtensions.push_back(VK_KHR_ROBUSTNESS_2_EXTENSION_NAME);
    }
    if (enableDescriptorIndexing) {
        vulkanDeviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    }
    if (enablePipelineRobustness) {
        vulkanDeviceExtensions.push_back(VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    }
    if (_optionals.portability_subset) {
        vulkanDeviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#ifdef VK_ARM_data_graph_binaries
    if (_optionals.data_graph_binaries) {
        vulkanDeviceExtensions.push_back(VK_ARM_DATA_GRAPH_BINARIES_EXTENSION_NAME);
    }
#endif

    const vk::DeviceCreateInfo deviceCreateInfo = {
        vk::DeviceCreateFlags(),
        1, // queueCreateInfoCount
        &deviceQueueCreateInfo,
        0, // enabledLayerCount
        nullptr,
        static_cast<uint32_t>(vulkanDeviceExtensions.size()), // enabledExtensionsCount
        vulkanDeviceExtensions.data(),
        &deviceFeat,  // pEnabledFeatures
        featureChain, // pNext
    };
    _dev = vk::raii::Device(_physicalDev, deviceCreateInfo);
} // namespace scenario-runner

const vk::raii::Device &Context::device() const { return _dev; }

const vk::raii::Instance &Context::instance() const { return _instance; }

const vk::raii::PhysicalDevice &Context::physicalDevice() const { return _physicalDev; }

uint32_t Context::familyQueueIdx() const { return _familyQueueIdx; }
} // namespace mlsdk::scenariorunner
