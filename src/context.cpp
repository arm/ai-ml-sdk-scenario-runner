/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "context.hpp"

#include "logging.hpp"
#include "scenario.hpp"

#include <limits>
#include <vector>

#include <iomanip>
#include <iostream>

namespace mlsdk::scenariorunner {
namespace {
uint32_t findQueue(const std::vector<vk::QueueFamilyProperties> &queueProps) {
    for (uint32_t i = 0; i < queueProps.size(); ++i) {
        const vk::QueueFamilyProperties &prop = queueProps[i];
        if (prop.queueFlags & vk::QueueFlagBits::eCompute) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

bool hasExtension(const std::vector<vk::ExtensionProperties> &extensions, const std::string &extensionName,
                  const std::vector<std::string> &disabledExtensions) {
    if (std::find(disabledExtensions.begin(), disabledExtensions.end(), extensionName) != disabledExtensions.end()) {
        return false;
    }
    return std::any_of(extensions.cbegin(), extensions.cend(), [extensionName](const auto &ext) {
        return std::string_view{static_cast<const char *>(ext.extensionName)} == extensionName;
    });
}
} // namespace

Context::Context(const ScenarioOptions &scenarioOptions)
    : _gpuDebugMarkersEnabled(scenarioOptions.enableGPUDebugMarkers),
      _sessionMemoryDumpEnabled(!scenarioOptions.sessionRAMsDumpDir.empty()) {
    // Create instance
    const vk::ApplicationInfo appInfo("Scenario-Runner", 1, nullptr, 0, VK_API_VERSION_1_3);

    std::vector<const char *> enabledLayers;
    std::vector<const char *> enabledExtensions;

    if (_gpuDebugMarkersEnabled) {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const vk::InstanceCreateInfo instanceInfo(
        vk::InstanceCreateFlags(), &appInfo, static_cast<uint32_t>(enabledLayers.size()), enabledLayers.data(),
        static_cast<uint32_t>(enabledExtensions.size()), enabledExtensions.data());
    _instance = vk::raii::Instance(_ctx, instanceInfo);

    // Create physical device
    auto _physicalDevices = vk::raii::PhysicalDevices(_instance);

    // Sort physical devices prioritizing discrete GPUs
    _physicalDev = *std::max_element(
        _physicalDevices.begin(), _physicalDevices.end(), [this](const auto &left, const auto &right) {
            // Select discrete GPU
            std::map<vk::PhysicalDeviceType, int> priorityOrder = {
                {vk::PhysicalDeviceType::eDiscreteGpu, 5},   //
                {vk::PhysicalDeviceType::eIntegratedGpu, 4}, //
                {vk::PhysicalDeviceType::eVirtualGpu, 3},    //
                {vk::PhysicalDeviceType::eCpu, 2},           //
                {vk::PhysicalDeviceType::eOther, 1},         //
            };

            return priorityOrder[left.getProperties().deviceType] < priorityOrder[right.getProperties().deviceType];
        });

    vk::PhysicalDeviceProperties properties = _physicalDev.getProperties();
    std::string deviceName = std::string(properties.deviceName.data());
    std::string deviceType = vk::to_string(properties.deviceType);
    std::ostringstream vendorID;
    vendorID << std::hex << std::setw(4) << std::setfill('0') << properties.vendorID << std::dec;

    mlsdk::logging::info("Device: " + deviceName + ", Type: " + deviceType + ", Vendor: 0x" + vendorID.str());

    const std::vector<vk::QueueFamilyProperties> queueProps = _physicalDev.getQueueFamilyProperties();
    _computeQueueIdx = findQueue(queueProps);
    if (_computeQueueIdx == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Cannot find queue index");
    }

    // Get device capabilities
    const std::vector<vk::ExtensionProperties> &extensions = _physicalDev.enumerateDeviceExtensionProperties(nullptr);

    _optionals.custom_border_color =
        hasExtension(extensions, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.mark_boundary =
        hasExtension(extensions, VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.maintenance5 =
        hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.deferred_operation =
        hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, scenarioOptions.disabledExtensions);
    _optionals.replicated_composites = hasExtension(extensions, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME,
                                                    scenarioOptions.disabledExtensions);

    // Create device
    const float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), _computeQueueIdx, 1,
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
        _physicalDev.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                                  vk::PhysicalDeviceVulkan12Features>();

    const auto &[available11Features, available12Features] =
        availableFeatures.template get<vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features>();

    vk::PhysicalDeviceVulkan11Features physicalDev11Feat;
    physicalDev11Feat.storageBuffer16BitAccess = available11Features.storageBuffer16BitAccess;
    physicalDev11Feat.uniformAndStorageBuffer16BitAccess = available11Features.uniformAndStorageBuffer16BitAccess;
    physicalDev11Feat.pNext = prevPNext;

    vk::PhysicalDeviceVulkan12Features physicalDev2Feat;
    physicalDev2Feat.hostQueryReset = true;
    physicalDev2Feat.storageBuffer8BitAccess = true;
    physicalDev2Feat.uniformAndStorageBuffer8BitAccess = available12Features.uniformAndStorageBuffer8BitAccess;
    physicalDev2Feat.shaderInt8 = true;
    physicalDev2Feat.shaderFloat16 = available12Features.shaderFloat16;
    physicalDev2Feat.vulkanMemoryModel = true;
    physicalDev2Feat.vulkanMemoryModelDeviceScope = available12Features.vulkanMemoryModelDeviceScope;
    physicalDev2Feat.pNext = &physicalDev11Feat;

    vk::PhysicalDeviceVulkan13Features physicalDev3Feat;
    physicalDev3Feat.synchronization2 = true;
    physicalDev3Feat.maintenance4 = true;
    physicalDev3Feat.pipelineCreationCacheControl = true;
    physicalDev3Feat.pNext = &physicalDev2Feat;

    vk::PhysicalDeviceTensorFeaturesARM tensorFeat;
    tensorFeat.shaderTensorAccess = true;
    tensorFeat.pNext = &physicalDev3Feat;

    vk::PhysicalDeviceDataGraphFeaturesARM dataGraphFeat;
    dataGraphFeat.dataGraph = true;
    dataGraphFeat.pNext = &tensorFeat;

    vk::PhysicalDeviceFeatures deviceFeat;
    deviceFeat.shaderInt16 = true;
    deviceFeat.shaderInt64 = true;

    std::vector<const char *> vulkanDeviceExtensions = {
        VK_ARM_DATA_GRAPH_EXTENSION_NAME,
        VK_ARM_TENSORS_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
    };

    if (_optionals.custom_border_color) {
        vulkanDeviceExtensions.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    }
    if (_optionals.mark_boundary) {
        vulkanDeviceExtensions.push_back(VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME);
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

    const vk::DeviceCreateInfo deviceCreateInfo = {
        vk::DeviceCreateFlags(),
        1, // queueCreateInfoCount
        &deviceQueueCreateInfo,
        0, // enabledLayerCount
        nullptr,
        static_cast<uint32_t>(vulkanDeviceExtensions.size()), // enabledExtensionsCount
        vulkanDeviceExtensions.data(),
        &deviceFeat,    // pEnabledFeatures
        &dataGraphFeat, // pNext
    };
    _dev = vk::raii::Device(_physicalDev, deviceCreateInfo);
} // namespace scenario-runner

const vk::raii::Device &Context::device() const { return _dev; }

const vk::raii::PhysicalDevice &Context::physicalDevice() const { return _physicalDev; }

uint32_t Context::computeFamilyQueueIdx() const { return _computeQueueIdx; }
} // namespace mlsdk::scenariorunner
