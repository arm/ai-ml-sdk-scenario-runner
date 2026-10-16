/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "device_selection.hpp"

#include "scenario_runner/scenario_options.hpp"

#include "vgf_view.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace mlsdk::scenariorunner::detail {
namespace {

template <typename... Functions> struct Overloaded : Functions... {
    using Functions::operator()...;
};

template <typename... Functions> Overloaded(Functions...) -> Overloaded<Functions...>;

int deviceTypePriority(vk::PhysicalDeviceType type) {
    switch (type) {
    case vk::PhysicalDeviceType::eDiscreteGpu:
        return 5;
    case vk::PhysicalDeviceType::eIntegratedGpu:
        return 4;
    case vk::PhysicalDeviceType::eVirtualGpu:
        return 3;
    case vk::PhysicalDeviceType::eCpu:
        return 2;
    case vk::PhysicalDeviceType::eOther:
    default:
        return 1;
    }
}

std::optional<uint32_t> findQueueFamily(const DeviceCandidate &candidate, const DeviceRequirements &requirements) {
    for (uint32_t index = 0; index < static_cast<uint32_t>(candidate.queueFamilies.size()); ++index) {
        const auto &properties = candidate.queueFamilies[index];
        if (properties.queueCount == 0 || !supportsQueueRequirements(properties.queueFlags, requirements.queueFlags)) {
            continue;
        }

        if (hasCapabilities(requirements.capabilities, DeviceCapability::DataGraphOpticalFlow) &&
            std::find(candidate.opticalFlowQueueFamilyIndices.begin(), candidate.opticalFlowQueueFamilyIndices.end(),
                      index) == candidate.opticalFlowQueueFamilyIndices.end()) {
            continue;
        }
        return index;
    }
    return std::nullopt;
}

vk::QueueFlags getRequiredQueueFlags(const std::vector<PipelineStage> &stages, MemoryAccess access) {
    vk::QueueFlags requiredQueueFlags;
    bool includesAllStages = false;
    for (const auto stage : stages) {
        switch (stage) {
        case PipelineStage::Compute:
            requiredQueueFlags |= vk::QueueFlagBits::eCompute;
            break;
        case PipelineStage::Graphics:
            requiredQueueFlags |= vk::QueueFlagBits::eGraphics;
            break;
        case PipelineStage::Graph:
            requiredQueueFlags |= vk::QueueFlagBits::eDataGraphARM;
            break;
        case PipelineStage::All:
            includesAllStages = true;
            break;
        default:
            throw std::runtime_error("Invalid barrier stage flag");
        }
    }

    if (includesAllStages) {
        if (access == MemoryAccess::ComputeShaderRead || access == MemoryAccess::ComputeShaderWrite) {
            requiredQueueFlags |= vk::QueueFlagBits::eCompute;
        } else if (access == MemoryAccess::GraphRead || access == MemoryAccess::GraphWrite) {
            requiredQueueFlags |= vk::QueueFlagBits::eDataGraphARM;
        }
    }
    return requiredQueueFlags;
}

template <typename BarrierId>
vk::QueueFlags getRequiredBarrierQueueFlags(const std::vector<BarrierId> &barriers, const ResourceManager &resources) {
    vk::QueueFlags requiredQueueFlags;
    for (const auto barrier : barriers) {
        const auto &barrierInfo = resources.get(barrier);
        requiredQueueFlags |= getRequiredQueueFlags(barrierInfo.srcStages, barrierInfo.srcAccess);
        requiredQueueFlags |= getRequiredQueueFlags(barrierInfo.dstStages, barrierInfo.dstAccess);
    }
    return requiredQueueFlags;
}

vk::QueueFlags getRequiredBarrierQueueFlags(const PipelineBarrierData &pipelineBarrier,
                                            const ResourceManager &resources) {
    return getRequiredBarrierQueueFlags(pipelineBarrier.memoryBarriers, resources) |
           getRequiredBarrierQueueFlags(pipelineBarrier.imageBarriers, resources) |
           getRequiredBarrierQueueFlags(pipelineBarrier.tensorBarriers, resources) |
           getRequiredBarrierQueueFlags(pipelineBarrier.bufferBarriers, resources);
}

} // namespace

bool hasCapabilities(DeviceCapability available, DeviceCapability required) noexcept {
    return (static_cast<uint32_t>(available) & static_cast<uint32_t>(required)) == static_cast<uint32_t>(required);
}

bool supportsQueueRequirements(vk::QueueFlags availableFlags, vk::QueueFlags workloadFlags) {
    const auto requiredWorkloadFlags = workloadFlags & ~vk::QueueFlagBits::eTransfer;
    if ((availableFlags & requiredWorkloadFlags) != requiredWorkloadFlags) {
        return false;
    }

    // Resource initialization and readback require the selected queue to support transfers.
    // Vulkan guarantees transfer support for graphics and compute queues even when
    // VK_QUEUE_TRANSFER_BIT is not reported for the queue family.
    constexpr vk::QueueFlags transferCapableFlags =
        vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
    return (availableFlags & transferCapableFlags) != vk::QueueFlags{};
}

DeviceRequirements getDeviceRequirements(const ScenarioBuildData &buildData, const ScenarioOptions &scenarioOptions) {
    DeviceRequirements requirements{vk::QueueFlags{}};
    const auto tensors = buildData.resources.tensors();
    if (tensors.begin() != tensors.end()) {
        requirements.capabilities |= DeviceCapability::Tensors;
    }
    if (scenarioOptions.shouldDumpNeuralDebugDatabase() || scenarioOptions.shouldDumpNeuralStatistics()) {
        requirements.capabilities |= DeviceCapability::DataGraphNeuralAcceleratorStatistics;
    }

    for (const auto &command : buildData.commands) {
        std::visit(Overloaded{
                       [&](const DispatchComputeData &) { requirements.queueFlags |= vk::QueueFlagBits::eCompute; },
                       [&](const DispatchFragmentData &) {
                           requirements.queueFlags |= vk::QueueFlagBits::eGraphics;
                           requirements.capabilities |= DeviceCapability::DynamicRendering;
                       },
                       [&](const DispatchVgfData &dispatch) {
                           const auto &vgf = buildData.resources.get(dispatch.vgf);
                           if (!vgf.src) {
                               throw std::runtime_error("VGF source missing: " + vgf.debugName);
                           }
                           for (uint32_t segmentIndex = 0; segmentIndex < vgf.src->getNumSegments(); ++segmentIndex) {
                               switch (vgf.src->getSegmentType(segmentIndex)) {
                               case ModuleType::GRAPH:
                                   requirements.queueFlags |= vk::QueueFlagBits::eDataGraphARM;
                                   requirements.capabilities |=
                                       DeviceCapability::DataGraphShaderModules | DeviceCapability::ShaderTensorAccess;
                                   if (vgf.src->hasImageBindings(segmentIndex)) {
                                       requirements.capabilities |= DeviceCapability::DataGraphImageLayouts;
                                   }
                                   break;
                               case ModuleType::SHADER:
                                   requirements.queueFlags |= vk::QueueFlagBits::eCompute;
                                   break;
                               }
                           }
                           if (vgf.src->hasTensorIntermediates()) {
                               requirements.capabilities |= DeviceCapability::Tensors;
                           }
                       },
                       [&](const DispatchDataGraphBinaryData &) {
                           requirements.queueFlags |= vk::QueueFlagBits::eDataGraphARM;
                           requirements.capabilities |= DeviceCapability::DataGraphBinaries;
                       },
                       [&](const DispatchDataGraphData &dispatch) {
                           requirements.queueFlags |= vk::QueueFlagBits::eDataGraphARM;
                           requirements.capabilities |=
                               DeviceCapability::DataGraphShaderModules | DeviceCapability::ShaderTensorAccess;
                           if (std::any_of(dispatch.bindings.begin(), dispatch.bindings.end(), [](const auto &binding) {
                                   return std::holds_alternative<ImageId>(binding.resource);
                               })) {
                               requirements.capabilities |= DeviceCapability::DataGraphImageLayouts;
                           }
                       },
                       [&](const DispatchOpticalFlowData &) {
                           requirements.queueFlags |= vk::QueueFlagBits::eDataGraphARM;
                           requirements.capabilities |= DeviceCapability::DataGraphOpticalFlow;
                       },
                       [&](const PipelineBarrierData &pipelineBarrier) {
                           requirements.queueFlags |=
                               getRequiredBarrierQueueFlags(pipelineBarrier, buildData.resources);
                       },
                       [&](const FrameBoundaryData &) {},
                   },
                   command);
    }

    // Partial uploads to multi-mip images generate missing levels using graphics-only blit commands.
    for (const auto &image : buildData.resources.images()) {
        if (image.info.mips > 1) {
            requirements.queueFlags |= vk::QueueFlagBits::eGraphics;
            break;
        }
    }

    // Image layout transitions during construction also contribute to queue selection.
    for (const auto &initialization : buildData.initializations) {
        const auto *imageInitialization = std::get_if<ImageInitialization>(&initialization);
        if (imageInitialization == nullptr) {
            continue;
        }
        if (!imageInitialization->data.has_value() && buildData.groupManager.isAliased(imageInitialization->id)) {
            // initializeResourceData calls Image::transitionLayout, whose barriers use both shader stages.
            requirements.queueFlags |= vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics;
        }
    }

    // Treat graph-stage barriers as data-graph workloads for device selection.
    if ((requirements.queueFlags & vk::QueueFlagBits::eDataGraphARM) != vk::QueueFlags{}) {
        requirements.capabilities |= DeviceCapability::DataGraph;
    }

    return requirements;
}

DeviceSelection selectDevice(const std::vector<DeviceCandidate> &candidates, const DeviceRequirements &requirements) {
    if (candidates.empty()) {
        throw std::runtime_error("No Vulkan physical devices found");
    }

    std::optional<DeviceSelection> selection;
    int selectedPriority = 0;
    bool hasCapableDevice = false;

    for (size_t deviceIndex = 0; deviceIndex < candidates.size(); ++deviceIndex) {
        const auto &candidate = candidates[deviceIndex];
        if (!hasCapabilities(candidate.capabilities, requirements.capabilities)) {
            continue;
        }
        hasCapableDevice = true;

        const auto queueFamilyIndex = findQueueFamily(candidate, requirements);
        if (!queueFamilyIndex) {
            continue;
        }

        const int priority = deviceTypePriority(candidate.type);
        if (!selection || priority > selectedPriority) {
            selection = DeviceSelection{deviceIndex, *queueFamilyIndex};
            selectedPriority = priority;
        }
    }

    if (!selection) {
        if (!hasCapableDevice) {
            throw std::runtime_error("No Vulkan physical device supports all capabilities required by the scenario");
        }
        throw std::runtime_error(
            "No capable Vulkan physical device has a nonempty queue family supporting all required queue flags and "
            "data graph operations");
    }
    return *selection;
}

} // namespace mlsdk::scenariorunner::detail
