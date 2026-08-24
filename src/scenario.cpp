/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "scenario.hpp"
#include "command_types.hpp"
#include "frame_capturer.hpp"
#include "glsl_compiler.hpp"
#include "guid.hpp"
#ifdef SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT
#    include "hlsl_compiler.hpp"
#endif
#include "image_formats.hpp"
#include "iresource.hpp"
#include "json_writer.hpp"
#include "logging.hpp"
#include "optical_flow_utils.hpp"
#include "scenario_builder.hpp"
#include "utils.hpp"

#include "vgf-utils/numpy.hpp"

#include <algorithm>
#include <string_view>
#include <unordered_set>

namespace mlsdk::scenariorunner {
namespace {
template <typename... Functions> struct Overloaded : Functions... {
    using Functions::operator()...;
};

template <typename... Functions> Overloaded(Functions...) -> Overloaded<Functions...>;

BufferData loadBufferData(const BufferDesc &desc) {
    BufferData bufferData;
    if (!desc.src.has_value()) {
        bufferData.data.resize(desc.size, 0);
        return bufferData;
    }

    MemoryMap mapped(desc.src.value());
    const auto parsedData = vgfutils::numpy::parse(mapped);
    bufferData.data.resize(parsedData.size());
    std::memcpy(bufferData.data.data(), parsedData.ptr, parsedData.size());
    return bufferData;
}

TensorData loadTensorData(const TensorDesc &desc) {
    TensorData tensorData;
    tensorData.shape = desc.dims;

    if (!desc.src.has_value()) {
        const auto format = getVkFormatFromString(desc.format);
        const auto expectedSize = elementSizeFromVkFormat(format) * totalElementsFromShape(desc.dims);
        tensorData.data.resize(expectedSize, 0);
        return tensorData;
    }

    MemoryMap mapped(desc.src.value());
    const auto parsedData = vgfutils::numpy::parse(mapped);

    tensorData.data.resize(parsedData.size());
    std::memcpy(tensorData.data.data(), parsedData.ptr, parsedData.size());
    tensorData.shape = parsedData.shape;
    return tensorData;
}

std::vector<GraphConstantInfo> collectGraphConstants(const std::vector<GraphConstantResourceId> &constantIds,
                                                     const ResourceManager &resources) {
    std::vector<GraphConstantInfo> constants;
    constants.reserve(constantIds.size());

    for (const auto id : constantIds) {
        constants.emplace_back(resources.get(id));
    }

    return constants;
}

void applyGraphResourceShaderMetadata(ShaderInfo &shaderInfo, const DataGraphInfo &dataGraph,
                                      const std::string &moduleName) {
    if (dataGraph.pushConstantsSize > 0) {
        shaderInfo.pushConstantsSize = dataGraph.pushConstantsSize;
    }

    for (const auto &specializationConstantMap : dataGraph.specializationConstantMaps) {
        if (specializationConstantMap.shaderTarget == moduleName) {
            shaderInfo.specializationConstants = specializationConstantMap.specializationConstants;
            return;
        }
    }
}

std::optional<RawDataId> getGraphPushData(const std::vector<ResolvedPushConstantMap> &pushConstants,
                                          const std::string &moduleName) {
    for (const auto &pushConstant : pushConstants) {
        if (pushConstant.shaderTarget == moduleName) {
            return pushConstant.pushData;
        }
    }
    return std::nullopt;
}

std::string resourceType(const std::unique_ptr<ResourceDesc> &resource) {
    switch (resource->resourceType) {
    case ResourceType::Unknown:
        return "Unknown";
    case ResourceType::Buffer:
        return "Buffer";
    case ResourceType::DataGraph:
        return "DataGraph";
    case ResourceType::Shader:
        return "Shader";
    case ResourceType::RawData:
        return "RawData";
    case ResourceType::Tensor:
        return "Tensor";
    case ResourceType::Image:
        return "Image";
    case ResourceType::ImageBarrier:
        return "ImageBarrier";
    case ResourceType::MemoryBarrier:
        return "MemoryBarrier";
    case ResourceType::TensorBarrier:
        return "TensorBarrier";
    case ResourceType::BufferBarrier:
        return "BufferBarrier";
    case ResourceType::GraphConstant:
        return "GraphConstant";
    }
    throw std::runtime_error("Unknown resource type in ScenarioSpec");
}

struct ResourceInfoFactory {
    BufferInfo createInfo(const BufferDesc &buffer) const {
        BufferInfo info{};
        info.debugName = buffer.guidStr;
        info.size = buffer.size;
        if (buffer.memoryGroup.has_value()) {
            info.memoryOffset = buffer.memoryGroup->offset;
        }
        return info;
    }

    RawDataInfo createInfo(const RawDataDesc &rawData) const { return {rawData.guidStr, rawData.src.value()}; }

    DataGraphInfo createInfo(const DataGraphDesc &dataGraph) const {
        return {dataGraph.guidStr, dataGraph.src.value(), dataGraph.pushConstantsSize,
                dataGraph.specializationConstantMaps};
    }

    GraphConstantInfo createInfo(const GraphConstantDesc &graphConstant) const {
        if (!graphConstant.src.has_value()) {
            throw std::runtime_error("Graph constant missing src: " + graphConstant.guidStr);
        }

        GraphConstantInfo info(graphConstant.guidStr, getVkFormatFromString(graphConstant.format), graphConstant.dims);
        MemoryMap mapped(graphConstant.src.value());
        const auto constantData = vgfutils::numpy::parse(mapped);

        if (constantData.shape.size() != info.dims.size()) {
            throw std::runtime_error("Graph constant dims mismatch for: " + graphConstant.guidStr);
        }
        for (size_t i = 0; i < info.dims.size(); ++i) {
            if (info.dims[i] != constantData.shape[i]) {
                throw std::runtime_error("Graph constant dims mismatch for: " + graphConstant.guidStr);
            }
        }

        // Validate that the NumPy payload size matches the declared format and shape.
        const uint64_t expectedDataSize =
            static_cast<uint64_t>(elementSizeFromVkFormat(info.format)) * totalElementsFromShape(info.dims);
        const auto actualDataSize = constantData.size();
        if (actualDataSize != expectedDataSize) {
            throw std::runtime_error(
                "Graph constant size does not match format and dims for: " + graphConstant.guidStr + "; expected " +
                std::to_string(expectedDataSize) + " vs " + std::to_string(actualDataSize));
        }

        info.data.resize(static_cast<size_t>(actualDataSize));
        std::memcpy(info.data.data(), constantData.ptr, static_cast<size_t>(actualDataSize));
        return info;
    }

    ImageInfo createInfo(const ImageDesc &image) const {
        ImageInfo info{};
        info.debugName = image.guidStr;
        info.targetFormat = getVkFormatFromString(image.format);
        info.shape.resize(image.dims.size());
        std::copy(image.dims.begin(), image.dims.end(), info.shape.begin());
        info.mips = image.mips;
        // Image sampler settings
        if (image.minFilter) {
            info.samplerSettings.minFilter = image.minFilter.value();
        }
        if (image.magFilter) {
            info.samplerSettings.magFilter = image.magFilter.value();
        }
        if (image.mipFilter) {
            info.samplerSettings.mipFilter = image.mipFilter.value();
        }
        if (image.borderAddressMode) {
            info.samplerSettings.addressModeU = image.borderAddressMode.value();
            info.samplerSettings.addressModeV = image.borderAddressMode.value();
            info.samplerSettings.addressModeW = image.borderAddressMode.value();
        }
        if (image.borderColor) {
            info.samplerSettings.borderColor = image.borderColor.value();
        }
        if (image.customBorderColor) {
            if (info.samplerSettings.borderColor == BorderColor::FloatCustomEXT) {
                info.samplerSettings.customBorderColor =
                    std::get<std::array<float, 4>>(image.customBorderColor.value());
            } else {
                info.samplerSettings.customBorderColor = std::get<std::array<int, 4>>(image.customBorderColor.value());
            }
        }

        if (image.src) {
            info.isInput = true;
            const auto &filename = image.src.value();
            const auto *handler = getImageFormatHandler(filename);
            if (handler) {
                info.format = handler->getFormat(filename);
            } else {
                info.format = info.targetFormat;
            }
        } else {
            info.format = info.targetFormat; // Output file does not change type
            info.isInput = false;
        }

        if (image.tiling) {
            info.tiling = image.tiling;
        }

        switch (image.shaderAccess) {
        case ShaderAccessType::ReadOnly:
            info.isSampled = true;
            break;
        case ShaderAccessType::WriteOnly:
        case ShaderAccessType::ImageRead:
            info.isStorage = true;
            break;
        case ShaderAccessType::ReadWrite:
            info.isSampled = true;
            info.isStorage = true;
            break;
        default:
            throw std::runtime_error("Unknown shader access type in ScenarioSpec");
        }

        if (info.targetFormat == vk::Format::eR32Sfloat && info.format == vk::Format::eD32SfloatS8Uint) {
            // Convert depth type to single channel color type, dropping stencil component
            info.format = info.targetFormat;
        }

        if (image.memoryGroup.has_value()) {
            info.memoryOffset = image.memoryGroup->offset;
        }

        info.isColorAttachment = image.colorAttachment;
        return info;
    }

    TensorInfo createInfo(const TensorDesc &tensor, bool descriptorBufferCaptureReplay) const {
        TensorInfo info;
        info.debugName = tensor.guidStr;
        if (tensor.memoryGroup.has_value()) {
            info.memoryOffset = tensor.memoryGroup->offset;
        }
        info.format = getVkFormatFromString(tensor.format);
        info.shape.resize(tensor.dims.size());
        std::copy(tensor.dims.begin(), tensor.dims.end(), info.shape.begin());
        if (tensor.tiling) {
            info.tiling = tensor.tiling.value();
        }
        info.descriptorBufferCaptureReplay = descriptorBufferCaptureReplay;

        return info;
    }

    ShaderInfo createInfo(const ShaderDesc &shader) const {
        return {shader.guidStr,
                shader.entry,
                shader.pushConstantsSize,
                shader.specializationConstants,
                shader.src.value_or(std::string{}),
                shader.shaderType,
                shader.stage,
                shader.buildOpts,
                shader.includeDirs};
    }
};

template <typename Id>
Id resolveResourceId(const std::unordered_map<Guid, TypedResourceId> &resourceIds, const Guid &guid,
                     std::string_view expectedType);

MemoryResourceId resolveMemoryResourceId(const std::unordered_map<Guid, TypedResourceId> &resourceIds,
                                         const Guid &guid);

void fill(const BaseBarrierDesc &barrier, BaseBarrierInfo &info) {
    info.debugName = barrier.guidStr;
    info.srcAccess = barrier.srcAccess;
    info.dstAccess = barrier.dstAccess;
    info.srcStages = barrier.srcStages;
    info.dstStages = barrier.dstStages;
}

struct BarrierInfoFactory {
    const std::unordered_map<Guid, TypedResourceId> &_resourceIds;

    ImageBarrierInfo createInfo(const ImageBarrierDesc &imageBarrier) const {
        ImageBarrierInfo info{};
        fill(imageBarrier, info);
        info.image = resolveResourceId<ImageId>(_resourceIds, imageBarrier.imageResource, "Image");
        info.oldLayout = imageBarrier.oldLayout;
        info.newLayout = imageBarrier.newLayout;
        info.range = imageBarrier.imageRange;
        return info;
    }

    MemoryBarrierInfo createInfo(const MemoryBarrierDesc &memoryBarrier) const {
        MemoryBarrierInfo info{};
        fill(memoryBarrier, info);
        return info;
    }

    TensorBarrierInfo createInfo(const TensorBarrierDesc &tensorBarrier) const {
        TensorBarrierInfo info{};
        fill(tensorBarrier, info);
        info.tensor = resolveResourceId<TensorId>(_resourceIds, tensorBarrier.tensorResource, "Tensor");
        return info;
    }

    BufferBarrierInfo createInfo(const BufferBarrierDesc &bufferBarrier) const {
        BufferBarrierInfo info{};
        fill(bufferBarrier, info);
        info.buffer = resolveResourceId<BufferId>(_resourceIds, bufferBarrier.bufferResource, "Buffer");
        info.offset = bufferBarrier.offset;
        info.size = bufferBarrier.size;
        return info;
    }
};

constexpr vk::DescriptorType convertDescriptorType(const DescriptorType descriptorType) {
    switch (descriptorType) {
    case DescriptorType::StorageImage:
        return vk::DescriptorType::eStorageImage;
    case DescriptorType::Auto:
        throw std::runtime_error("Cannot infer the descriptor type without context");
    default:
        throw std::runtime_error("Descriptor type is invalid");
    }
}

vk::DescriptorType getResourceDescriptorType(const ResourceManager &resources, const MemoryResourceId &resource) {
    const auto getDescriptorType = Overloaded{
        [&](BufferId id) {
            (void)resources.get(id);
            return vk::DescriptorType::eStorageBuffer;
        },
        [&](TensorId id) {
            (void)resources.get(id);
            return vk::DescriptorType::eTensorARM;
        },
        [&](ImageId id) {
            return resources.get(id).isSampled ? vk::DescriptorType::eCombinedImageSampler
                                               : vk::DescriptorType::eStorageImage;
        },
    };
    return std::visit(getDescriptorType, resource);
}

TypedBinding convertBinding(const ResourceManager &resources,
                            const std::unordered_map<Guid, TypedResourceId> &resourceIds, const BindingDesc &binding) {
    const auto resource = resolveMemoryResourceId(resourceIds, binding.resourceRef);
    const auto vkType = binding.descriptorType == DescriptorType::Auto ? getResourceDescriptorType(resources, resource)
                                                                       : convertDescriptorType(binding.descriptorType);
    return {binding.set, binding.id, resource, binding.lod, vkType};
}

std::vector<TypedBinding> convertBindings(const ResourceManager &resources,
                                          const std::unordered_map<Guid, TypedResourceId> &resourceIds,
                                          const std::vector<BindingDesc> &bindingDescs) {
    std::vector<TypedBinding> bindings;
    bindings.reserve(bindingDescs.size());
    for (const auto &binding : bindingDescs) {
        bindings.push_back(convertBinding(resources, resourceIds, binding));
    }
    return bindings;
}

class Creator final : public IResourceCreator {
  public:
    Creator(ResourceManager &resources, DataManager &dataManager) : _resources{resources}, _dataManager{dataManager} {}

    BufferId createBuffer(BufferInfo &&info) override {
        const auto id = _resources.addBuffer(std::move(info));
        _dataManager.createBuffer(id, _resources.get(id));
        return id;
    }

    TensorId createTensor(TensorInfo &&info) override {
        const auto id = _resources.addTensor(std::move(info));
        _dataManager.createTensor(id, _resources.get(id));
        return id;
    }

    ImageId createImage(ImageInfo &&info) override {
        const auto id = _resources.addImage(std::move(info));
        _dataManager.createImage(id, _resources.get(id));
        return id;
    }

  private:
    ResourceManager &_resources;
    DataManager &_dataManager;
};

const TypedResourceId &resolveTypedResourceId(const std::unordered_map<Guid, TypedResourceId> &resourceIds,
                                              const Guid &guid, std::string_view expectedType) {
    const auto resource = resourceIds.find(guid);
    if (resource == resourceIds.end()) {
        throw std::runtime_error(std::string(expectedType) + " resource not found.");
    }
    return resource->second;
}

template <typename Id>
Id resolveResourceId(const std::unordered_map<Guid, TypedResourceId> &resourceIds, const Guid &guid,
                     std::string_view expectedType) {
    const auto &resourceId = resolveTypedResourceId(resourceIds, guid, expectedType);
    const auto *id = std::get_if<Id>(&resourceId);
    if (id == nullptr) {
        throw std::runtime_error("Resource UID has the wrong type; expected " + std::string(expectedType) + ".");
    }
    return *id;
}

MemoryResourceId resolveMemoryResourceId(const std::unordered_map<Guid, TypedResourceId> &resourceIds,
                                         const Guid &guid) {
    const auto &resourceId = resolveTypedResourceId(resourceIds, guid, "Memory");
    if (const auto *id = std::get_if<BufferId>(&resourceId)) {
        return *id;
    }
    if (const auto *id = std::get_if<ImageId>(&resourceId)) {
        return *id;
    }
    if (const auto *id = std::get_if<TensorId>(&resourceId)) {
        return *id;
    }
    throw std::runtime_error("Resource UID has the wrong type; expected a memory resource.");
}

template <typename Key>
MemoryGroupId getOrCreateMemoryGroup(GroupManager &groupManager, std::unordered_map<Key, MemoryGroupId> &memoryGroupIds,
                                     const Key &key) {
    const auto existingGroup = memoryGroupIds.find(key);
    if (existingGroup != memoryGroupIds.end()) {
        return existingGroup->second;
    }
    const auto group = groupManager.createMemoryGroup();
    memoryGroupIds.emplace(key, group);
    return group;
}

void registerMemoryGroup(ScenarioBuilder &builder, std::unordered_map<Guid, MemoryGroupId> &memoryGroupIds,
                         MemoryResourceId resource, const std::optional<MemoryGroup> &memoryGroup) {
    if (memoryGroup.has_value()) {
        const auto existingGroup = memoryGroupIds.find(memoryGroup->memoryUid);
        if (existingGroup != memoryGroupIds.end()) {
            builder.addResourceToMemoryGroup(existingGroup->second, resource);
            return;
        }
        const auto group = builder.createMemoryGroup();
        memoryGroupIds.emplace(memoryGroup->memoryUid, group);
        builder.addResourceToMemoryGroup(group, resource);
    }
}

template <typename Id>
Id resolveResourceUid(const std::unordered_map<Guid, TypedResourceId> &resourceIds, std::string_view uid,
                      std::string_view resourceType, std::string_view operation) {
    const auto resource = resourceIds.find(Guid(std::string(uid)));
    if (resource == resourceIds.end()) {
        throw std::runtime_error("Scenario::" + std::string(operation) + ": resource UID '" + std::string(uid) +
                                 "' not found.");
    }

    const auto *id = std::get_if<Id>(&resource->second);
    if (id == nullptr) {
        throw std::runtime_error("Scenario::" + std::string(operation) + ": resource UID '" + std::string(uid) +
                                 "' does not identify a " + std::string(resourceType) + " resource.");
    }
    return *id;
}

struct CommandDataFactory {
    const ResourceManager &_resources;
    const std::unordered_map<Guid, TypedResourceId> &_resourceIds;

    ShaderId getShaderId(const Guid &guid) const { return resolveResourceId<ShaderId>(_resourceIds, guid, "Shader"); }

    GraphConstantResourceId getGraphConstantResourceId(const Guid &guid) const {
        return resolveResourceId<GraphConstantResourceId>(_resourceIds, guid, "Graph constant");
    }

    DataGraphId getDataGraphId(const Guid &guid) const {
        return resolveResourceId<DataGraphId>(_resourceIds, guid, "Data graph");
    }

    std::optional<RawDataId> getRawDataId(const std::optional<Guid> &guid) const {
        if (!guid) {
            return std::nullopt;
        }
        return resolveResourceId<RawDataId>(_resourceIds, *guid, "Raw data");
    }

    DispatchComputeData createData(const DispatchComputeDesc &dispatchCompute) {
        DispatchComputeData data{getShaderId(dispatchCompute.shaderRef)};
        data.debugName = dispatchCompute.debugName;
        data.bindings = convertBindings(_resources, _resourceIds, dispatchCompute.bindings);
        data.computeDispatch.gwcx = dispatchCompute.rangeND[0];
        data.computeDispatch.gwcy = dispatchCompute.rangeND[1];
        data.computeDispatch.gwcz = dispatchCompute.rangeND[2];
        data.computeDispatch.profileName = dispatchCompute.debugName;
        data.implicitBarrier = dispatchCompute.implicitBarrier;
        data.pushData = getRawDataId(dispatchCompute.pushDataRef);
        return data;
    }

    DispatchFragmentData createData(const DispatchFragmentDesc &dispatchFragment) {
        DispatchFragmentData data{getShaderId(dispatchFragment.vertexShaderRef),
                                  getShaderId(dispatchFragment.fragmentShaderRef)};
        data.debugName = dispatchFragment.debugName;
        data.bindings = convertBindings(_resources, _resourceIds, dispatchFragment.bindings);
        data.colorAttachments.reserve(dispatchFragment.colorAttachments.size());
        for (const auto &attachmentDesc : dispatchFragment.colorAttachments) {
            data.colorAttachments.push_back(
                {resolveResourceId<ImageId>(_resourceIds, attachmentDesc.resourceRef, "Image"), attachmentDesc.lod});
        }
        if (dispatchFragment.renderExtent) {
            const auto &extent = dispatchFragment.renderExtent.value();
            data.renderExtent = vk::Extent2D(extent[0], extent[1]);
        }
        data.implicitBarrier = dispatchFragment.implicitBarrier;
        data.pushData = getRawDataId(dispatchFragment.pushDataRef);
        return data;
    }

    DispatchBarrierData createData(const DispatchBarrierDesc &dispatchBarrier) {
        DispatchBarrierData data;
        for (const auto &ref : dispatchBarrier.bufferBarriersRef) {
            data.bufferBarriers.push_back(
                resolveResourceId<BufferBarrierId>(_resourceIds, Guid(ref), "Buffer barrier"));
        }
        for (const auto &ref : dispatchBarrier.imageBarriersRef) {
            data.imageBarriers.push_back(resolveResourceId<ImageBarrierId>(_resourceIds, Guid(ref), "Image barrier"));
        }
        for (const auto &ref : dispatchBarrier.memoryBarriersRef) {
            data.memoryBarriers.push_back(
                resolveResourceId<MemoryBarrierId>(_resourceIds, Guid(ref), "Memory barrier"));
        }
        for (const auto &ref : dispatchBarrier.tensorBarriersRef) {
            data.tensorBarriers.push_back(
                resolveResourceId<TensorBarrierId>(_resourceIds, Guid(ref), "Tensor barrier"));
        }
        return data;
    }

    DispatchDataGraphData createData(const DispatchDataGraphDesc &dispatchDataGraph) {
        DispatchDataGraphData data{getDataGraphId(dispatchDataGraph.dataGraphRef)};
        data.debugName = dispatchDataGraph.debugName;
        data.bindings = convertBindings(_resources, _resourceIds, dispatchDataGraph.bindings);
        data.pushConstants.reserve(dispatchDataGraph.pushConstants.size());
        for (const auto &pushConstant : dispatchDataGraph.pushConstants) {
            data.pushConstants.push_back(
                {resolveResourceId<RawDataId>(_resourceIds, pushConstant.pushDataRef, "Raw data"),
                 pushConstant.shaderTarget});
        }
        data.shaderSubstitutions.reserve(dispatchDataGraph.shaderSubstitutions.size());
        for (const auto &substitution : dispatchDataGraph.shaderSubstitutions) {
            data.shaderSubstitutions.push_back({getShaderId(substitution.shaderRef), substitution.target});
        }
        data.implicitBarrier = dispatchDataGraph.implicitBarrier;
        return data;
    }

    DispatchSpirvGraphData createData(const DispatchSpirvGraphDesc &dispatchSpirvGraph) {
        DispatchSpirvGraphData data{getShaderId(dispatchSpirvGraph.dataGraphRef)};
        data.debugName = dispatchSpirvGraph.debugName;
        data.bindings = convertBindings(_resources, _resourceIds, dispatchSpirvGraph.bindings);
        data.graphConstants.reserve(dispatchSpirvGraph.graphConstants.size());
        for (const auto &graphConstant : dispatchSpirvGraph.graphConstants) {
            data.graphConstants.push_back(getGraphConstantResourceId(graphConstant));
        }
        data.implicitBarrier = dispatchSpirvGraph.implicitBarrier;
        return data;
    }

    DispatchOpticalFlowData createData(const DispatchOpticalFlowDesc &dispatchOpticalFlow) {
        DispatchOpticalFlowData data{convertBinding(_resources, _resourceIds, dispatchOpticalFlow.searchImage),
                                     convertBinding(_resources, _resourceIds, dispatchOpticalFlow.templateImage),
                                     convertBinding(_resources, _resourceIds, dispatchOpticalFlow.outputImage)};
        data.debugName = dispatchOpticalFlow.debugName;
        if (dispatchOpticalFlow.hintMotionVectors.has_value()) {
            data.hintMotionVectors =
                convertBinding(_resources, _resourceIds, dispatchOpticalFlow.hintMotionVectors.value());
        }
        if (dispatchOpticalFlow.outputCost.has_value()) {
            data.outputCost = convertBinding(_resources, _resourceIds, dispatchOpticalFlow.outputCost.value());
        }

        data.width = dispatchOpticalFlow.width;
        data.height = dispatchOpticalFlow.height;
        data.performanceLevel = dispatchOpticalFlow.performanceLevel;
        data.executionFlags = dispatchOpticalFlow.executionFlags;
        data.gridSize = dispatchOpticalFlow.gridSize;
        data.meanFlowL1NormHint = dispatchOpticalFlow.meanFlowL1NormHint;

        data.implicitBarrier = dispatchOpticalFlow.implicitBarrier;
        return data;
    }

    MarkBoundaryData createData(const MarkBoundaryDesc &markBoundary) {
        MarkBoundaryData data;

        for (const auto &resourceRef : markBoundary.resources) {
            const Guid guid(resourceRef);
            const auto &resource = resolveTypedResourceId(_resourceIds, guid, "Memory");
            if (const auto *buffer = std::get_if<BufferId>(&resource)) {
                data.buffers.push_back(*buffer);
            } else if (const auto *image = std::get_if<ImageId>(&resource)) {
                data.images.push_back(*image);
            } else if (const auto *tensor = std::get_if<TensorId>(&resource)) {
                data.tensors.push_back(*tensor);
            } else {
                throw std::runtime_error("Unsupported resource");
            }
        }
        return data;
    }
};

auto getFamilyQueue(const ScenarioSpec &spec) {
    if (spec.requiresGraphicsFamilyQueue) {
        return FamilyQueue::Graphics;
    }
    if (spec.useComputeFamilyQueue) {
        return FamilyQueue::Compute;
    }
    return FamilyQueue::DataGraph;
}

// Map performance level to Vulkan enum
auto getOpticalFlowPerformanceLevel(OpticalFlowPerformanceLevel performanceLevel) {
    switch (performanceLevel) {
    case OpticalFlowPerformanceLevel::Unknown:
        return vk::DataGraphOpticalFlowPerformanceLevelARM::eUnknown;
    case OpticalFlowPerformanceLevel::Slow:
        return vk::DataGraphOpticalFlowPerformanceLevelARM::eSlow;
    case OpticalFlowPerformanceLevel::Medium:
        return vk::DataGraphOpticalFlowPerformanceLevelARM::eMedium;
    case OpticalFlowPerformanceLevel::Fast:
        return vk::DataGraphOpticalFlowPerformanceLevelARM::eFast;
    default:
        throw std::runtime_error("Unrecognised performance level, expected unknown, slow, medium, or fast.");
    }
}

// Map grid size to Vulkan enums
auto getOpticalFlowGridSize(OpticalFlowGridSize gridSize) {
    switch (gridSize) {
    case OpticalFlowGridSize::e1x1:
        return vk::DataGraphOpticalFlowGridSizeFlagBitsARM::e1X1;
    case OpticalFlowGridSize::e2x2:
        return vk::DataGraphOpticalFlowGridSizeFlagBitsARM::e2X2;
    case OpticalFlowGridSize::e4x4:
        return vk::DataGraphOpticalFlowGridSizeFlagBitsARM::e4X4;
    case OpticalFlowGridSize::e8x8:
        return vk::DataGraphOpticalFlowGridSizeFlagBitsARM::e8X8;
    default:
        throw std::runtime_error("Unrecognised grid size, expected 1x1, 2x2, 4x4, or 8x8.");
    }
}

void verifyOpticalFlowData(const DataManager &dataManager, const DispatchOpticalFlowData &dispatchOpticalFlow) {
    verifyOpticalFlowConfig(dataManager, dispatchOpticalFlow.searchImage, dispatchOpticalFlow.templateImage,
                            dispatchOpticalFlow.outputImage, dispatchOpticalFlow.hintMotionVectors,
                            dispatchOpticalFlow.outputCost, dispatchOpticalFlow.width, dispatchOpticalFlow.height,
                            dispatchOpticalFlow.gridSize);
}

} // namespace

Scenario::Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec)
    : _opts{opts}, _ctx{opts, getFamilyQueue(scenarioSpec)}, _scenarioSpec(scenarioSpec), _compute(_ctx) {
    buildJsonScenarioData(scenarioSpec);
    setupResources();
    setupRuntimeCommands();
}

BufferId Scenario::getBufferId(std::string_view uid) const {
    return resolveResourceUid<BufferId>(_resourceIds, uid, "Buffer", "getBufferId");
}

ImageId Scenario::getImageId(std::string_view uid) const {
    return resolveResourceUid<ImageId>(_resourceIds, uid, "Image", "getImageId");
}

TensorId Scenario::getTensorId(std::string_view uid) const {
    return resolveResourceUid<TensorId>(_resourceIds, uid, "Tensor", "getTensorId");
}

void Scenario::upload(BufferId id, const BufferDataView &data) {
    if (!_dataManager.hasBuffer(id)) {
        throw std::runtime_error("Scenario::upload: Buffer resource not found.");
    }
    _dataManager.getBuffer(id).upload(_ctx, data);
}

void Scenario::upload(ImageId id, const ImageDataView &data) {
    if (!_dataManager.hasImage(id)) {
        throw std::runtime_error("Scenario::upload: Image resource not found.");
    }
    _dataManager.getImageMut(id).upload(_ctx, data);
}

void Scenario::upload(TensorId id, const TensorDataView &data) {
    if (!_dataManager.hasTensor(id)) {
        throw std::runtime_error("Scenario::upload: Tensor resource not found.");
    }
    _dataManager.getTensor(id).upload(_ctx, data);
}

BufferData Scenario::download(BufferId id) const {
    if (!_dataManager.hasBuffer(id)) {
        throw std::runtime_error("Scenario::download: Buffer resource not found.");
    }
    return _dataManager.getBuffer(id).download(_ctx);
}

ImageData Scenario::download(ImageId id) {
    if (!_dataManager.hasImage(id)) {
        throw std::runtime_error("Scenario::download: Image resource not found.");
    }
    return _dataManager.getImageMut(id).download(_ctx);
}

TensorData Scenario::download(TensorId id) const {
    if (!_dataManager.hasTensor(id)) {
        throw std::runtime_error("Scenario::download: Tensor resource not found.");
    }
    return _dataManager.getTensor(id).download(_ctx);
}

const ShaderInfo &Scenario::getShader(ShaderId id) const { return _resources.get(id); }

MemoryResourceId Scenario::getMemoryResourceId(const Guid &guid) const {
    return resolveMemoryResourceId(_resourceIds, guid);
}

const ShaderInfo &Scenario::getSubstitutionShader(const std::vector<ResolvedShaderSubstitution> &shaderSubstitutions,
                                                  const std::string &moduleName) const {
    for (const auto &shaderSub : shaderSubstitutions) {
        if (shaderSub.target == moduleName) {
            return getShader(shaderSub.shader);
        }
    }
    throw std::runtime_error("Could not perform shader substitution");
}

void Scenario::run(int repeatCount, bool dryRun) {
    if (repeatCount <= 0) {
        throw std::invalid_argument("Scenario repeat count must be greater than zero; received " +
                                    std::to_string(repeatCount) + ".");
    }

    for (int iteration = 0; iteration < repeatCount; ++iteration) {
        mlsdk::logging::debug("Iteration: " + std::to_string(iteration));
        runIteration(iteration, repeatCount, dryRun);
    }
    saveResults(dryRun);
}

void Scenario::runIteration(int iteration, int repeatCount, bool dryRun) {
    if (_opts.captureFrame && !_frameCapturer) {
        _frameCapturer = std::make_unique<FrameCapturer>();
    }

    if (_hasRun) {
        resetForNextRun();
    }
    if (_frameCapturer) {
        _frameCapturer->begin();
    }

    if (!dryRun) {
        if (hasAliasedOptimalTensors()) {
            _compute.prepareCommandBuffer();
            handleAliasedLayoutTransitions();
        }
        _compute.submitAndWaitOnFence(_perfCounters, iteration);
    }
    saveProfilingData(iteration, repeatCount, dryRun);

    _hasRun = true;

    if (_frameCapturer) {
        _frameCapturer->end();
    }
}

void Scenario::resetForNextRun() {
    _compute.reset();
    for (const auto &[id, info] : _resources.images()) {
        if (info.tiling == Tiling::Optimal) {
            _dataManager.getImageMut(id).resetLayout();
        }
    }
}

void Scenario::createRuntimeResources() {
    for (const auto &[id, info] : _resources.buffers()) {
        _dataManager.createBuffer(id, info);
    }
    for (const auto &[id, info] : _resources.images()) {
        _dataManager.createImage(id, info);
    }
    for (const auto &[id, info] : _resources.tensors()) {
        _dataManager.createTensor(id, info);
    }
    for (const auto &[id, info] : _resources.rawData()) {
        _dataManager.createRawData(id, info);
    }
    for (const auto &[id, info] : _resources.dataGraphs()) {
        PerfCounterGuard guard(_perfCounters, "Parse VGF: " + info.debugName, "Scenario Setup");
        _dataManager.createVgfView(id, info);
    }
}

void Scenario::createRuntimeBarriers() {
    for (const auto &[id, info] : _resources.imageBarriers()) {
        _dataManager.createImageBarrier(id, info);
    }
    for (const auto &[id, info] : _resources.memoryBarriers()) {
        _dataManager.createMemoryBarrier(id, info);
    }
    for (const auto &[id, info] : _resources.tensorBarriers()) {
        _dataManager.createTensorBarrier(id, info);
    }
    for (const auto &[id, info] : _resources.bufferBarriers()) {
        _dataManager.createBufferBarrier(id, info);
    }
}

namespace {
void registerJsonResourceInfo(ScenarioBuilder &builder, const ScenarioSpec &scenarioSpec,
                              const ResourceInfoFactory &resourceInfoFactory, bool captureFrame,
                              std::unordered_map<Guid, MemoryGroupId> &jsonMemoryGroupIds,
                              std::unordered_map<Guid, TypedResourceId> &resourceIds) {

    for (const auto &resource : scenarioSpec.resources) {
        switch (resource->resourceType) {
        case ResourceType::Buffer: {
            const auto &buffer = reinterpret_cast<const std::unique_ptr<BufferDesc> &>(resource);
            const auto id = builder.addBuffer(resourceInfoFactory.createInfo(*buffer));
            resourceIds.emplace(resource->guid, id);
            registerMemoryGroup(builder, jsonMemoryGroupIds, id, buffer->memoryGroup);
        } break;
        case ResourceType::RawData: {
            const auto &rawData = reinterpret_cast<const std::unique_ptr<RawDataDesc> &>(resource);
            const auto id = builder.addRawData(resourceInfoFactory.createInfo(*rawData));
            resourceIds.emplace(resource->guid, id);
        } break;
        case ResourceType::Image: {
            const auto &image = reinterpret_cast<const std::unique_ptr<ImageDesc> &>(resource);
            const auto id = builder.addImage(resourceInfoFactory.createInfo(*image));
            resourceIds.emplace(resource->guid, id);
            registerMemoryGroup(builder, jsonMemoryGroupIds, id, image->memoryGroup);
        } break;
        case ResourceType::DataGraph: {
            const auto &dataGraph = reinterpret_cast<const std::unique_ptr<DataGraphDesc> &>(resource);
            const auto id = builder.addDataGraph(resourceInfoFactory.createInfo(*dataGraph));
            resourceIds.emplace(resource->guid, id);
        } break;
        case ResourceType::Tensor: {
            const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource);
            const auto id = builder.addTensor(resourceInfoFactory.createInfo(*tensor, captureFrame));
            resourceIds.emplace(resource->guid, id);
            registerMemoryGroup(builder, jsonMemoryGroupIds, id, tensor->memoryGroup);
        } break;
        case ResourceType::Shader: {
            const auto &shader = reinterpret_cast<const std::unique_ptr<ShaderDesc> &>(resource);
            const auto id = builder.addShader(resourceInfoFactory.createInfo(*shader));
            resourceIds.emplace(resource->guid, id);
        } break;
        case ResourceType::GraphConstant: {
            const auto &graphConstant = reinterpret_cast<const std::unique_ptr<GraphConstantDesc> &>(resource);
            const auto id = builder.addGraphConstant(resourceInfoFactory.createInfo(*graphConstant));
            resourceIds.emplace(resource->guid, id);
        } break;
        default:
            // Barriers can precede the regular resources they reference, so register them in a second pass.
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }
}

void registerJsonBarrierInfo(ScenarioBuilder &builder, const ScenarioSpec &scenarioSpec,
                             std::unordered_map<Guid, TypedResourceId> &resourceIds) {
    // Barrier descriptions can reference resources declared later in the JSON,
    // so translate them only after all regular resources have typed IDs.
    BarrierInfoFactory barrierInfoFactory{resourceIds};
    for (const auto &resource : scenarioSpec.resources) {
        switch (resource->resourceType) {
        case ResourceType::ImageBarrier: {
            const auto &imageBarrier = reinterpret_cast<const std::unique_ptr<ImageBarrierDesc> &>(resource);
            const auto id = builder.addImageBarrier(barrierInfoFactory.createInfo(*imageBarrier));
            resourceIds.emplace(resource->guid, id);
        } break;
        case ResourceType::MemoryBarrier: {
            const auto &memoryBarrier = reinterpret_cast<const std::unique_ptr<MemoryBarrierDesc> &>(resource);
            const auto id = builder.addMemoryBarrier(barrierInfoFactory.createInfo(*memoryBarrier));
            resourceIds.emplace(resource->guid, id);
        } break;
        case ResourceType::TensorBarrier: {
            const auto &tensorBarrier = reinterpret_cast<const std::unique_ptr<TensorBarrierDesc> &>(resource);
            const auto id = builder.addTensorBarrier(barrierInfoFactory.createInfo(*tensorBarrier));
            resourceIds.emplace(resource->guid, id);
        } break;
        case ResourceType::BufferBarrier: {
            const auto &bufferBarrier = reinterpret_cast<const std::unique_ptr<BufferBarrierDesc> &>(resource);
            const auto id = builder.addBufferBarrier(barrierInfoFactory.createInfo(*bufferBarrier));
            resourceIds.emplace(resource->guid, id);
        } break;
        default:
            // Regular resources were registered in the first pass.
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }
}
} // namespace

void Scenario::buildJsonScenarioData(const ScenarioSpec &scenarioSpec) {
    mlsdk::logging::info("Setup resources, count: " + std::to_string(scenarioSpec.resources.size()));
    ScenarioBuilder builder;
    ResourceInfoFactory resourceInfoFactory;
    std::unordered_map<Guid, MemoryGroupId> jsonMemoryGroupIds;
    std::unordered_map<Guid, TypedResourceId> resourceIds;

    registerJsonResourceInfo(builder, scenarioSpec, resourceInfoFactory, _opts.captureFrame, jsonMemoryGroupIds,
                             resourceIds);
    registerJsonBarrierInfo(builder, scenarioSpec, resourceIds);
    resolveCommands(builder, scenarioSpec, resourceIds);
    auto buildData = builder.takeBuildData();
    _resources = std::move(buildData.resources);
    _resourceIds = std::move(resourceIds);
    _commands = std::move(buildData.commands);
    _groupManager = std::move(buildData.groupManager);
}

void Scenario::setupResources() {
    createRuntimeResources();

    Creator vgfResourceCreator{_resources, _dataManager};
    // Per data graph, map VGF alias group IDs to runtime memory group IDs.
    std::unordered_map<DataGraphId, std::unordered_map<uint32_t, MemoryGroupId>> vgfMemoryGroupIds;

    for (const auto &command : _commands) {
        const auto *dispatchDataGraph = std::get_if<DispatchDataGraphData>(&command);
        if (dispatchDataGraph == nullptr) {
            continue;
        }
        const auto &vgfView = _dataManager.getVgfView(dispatchDataGraph->dataGraph);
        auto [creationResultIt, created] = _vgfResourceCreationResults.try_emplace(dispatchDataGraph->dataGraph);
        if (created) {
            creationResultIt->second = vgfView.createIntermediateResources(vgfResourceCreator);
        }
        const auto &creationResult = creationResultIt->second;
        for (const auto &[aliasGroupId, resourceIds] : creationResult.memoryGroups) {
            auto &aliasGroupIds = vgfMemoryGroupIds[dispatchDataGraph->dataGraph];
            const auto group = getOrCreateMemoryGroup(_groupManager, aliasGroupIds, aliasGroupId);
            for (const auto &resourceId : resourceIds) {
                _groupManager.addResourceToGroup(group, resourceId);
            }
        }

        // External resources are resolved for each dispatch because bindings may differ.
        for (const auto &binding : dispatchDataGraph->bindings) {
            const auto aliasGroupId = vgfView.getModelResourceAliasGroup(binding.id);
            if (!aliasGroupId.has_value()) {
                continue;
            }
            const auto resourceId = binding.resource;
            auto &aliasGroupIds = vgfMemoryGroupIds[dispatchDataGraph->dataGraph];
            const auto group = getOrCreateMemoryGroup(_groupManager, aliasGroupIds, *aliasGroupId);
            _groupManager.addResourceToGroup(group, resourceId);
        }
    }
    _groupManager.finalize();

    // Setup aliasing resources, foundation before accessing tensors
    for (const auto &entry : _resources.buffers()) {
        const auto id = entry.id;
        _dataManager.getBufferMut(id).setup(_ctx, _groupManager.getMemoryManager(id));
    }
    for (const auto &entry : _resources.images()) {
        const auto id = entry.id;
        _dataManager.getImageMut(id).setup(_ctx, _groupManager.getMemoryManager(id));
    }

    // Setup tensors, aliasing tensors are dependent on other resources having been constructed
    for (const auto &entry : _resources.tensors()) {
        const auto id = entry.id;
        _dataManager.getTensorMut(id).setup(_ctx, _groupManager.getMemoryManager(id));
    }

    createRuntimeBarriers();

    // Allocate resource memory before loading runtime input data.
    for (const auto &entry : _resources.tensors()) {
        _dataManager.getTensorMut(entry.id).allocateMemory(_ctx);
    }
    for (const auto &entry : _resources.images()) {
        _dataManager.getImageMut(entry.id).allocateMemory(_ctx);
    }
    for (const auto &entry : _resources.buffers()) {
        _dataManager.getBufferMut(entry.id).allocateMemory(_ctx);
    }

    loadJsonResourceData();
}

void Scenario::loadJsonResourceData() {
    // Preserve description-based CLI initialization by routing it through the
    // same typed Scenario upload API used by in-memory clients.
    for (const auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case ResourceType::Tensor: {
            const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource);
            PerfCounterGuard guard(_perfCounters, "Load Tensor: " + tensor->guidStr, "Scenario Setup");
            const auto id = resolveResourceId<TensorId>(_resourceIds, tensor->guid, "Tensor");
            if (tensor->src || !_groupManager.isAliased(id)) {
                const auto tensorData = loadTensorData(*tensor);
                upload(id, {tensorData.data.data(), tensorData.data.size(), tensorData.shape, tensorData.format});
            }
        } break;
        case ResourceType::Image: {
            const auto &image = reinterpret_cast<const std::unique_ptr<ImageDesc> &>(resource);
            PerfCounterGuard guard(_perfCounters, "Load Image: " + image->guidStr, "Scenario Setup");
            const auto id = resolveResourceId<ImageId>(_resourceIds, image->guid, "Image");
            auto &imageRec = _dataManager.getImageMut(id);
            if (image->src || !_groupManager.isAliased(id)) {
                imageRec.fillFromDescription(_ctx, *image);
            } else {
                imageRec.transitionLayout(_ctx, vk::ImageLayout::eGeneral);
            }
        } break;
        case ResourceType::Buffer: {
            const auto &buffer = reinterpret_cast<const std::unique_ptr<BufferDesc> &>(resource);
            PerfCounterGuard guard(_perfCounters, "Load Buffer: " + buffer->guidStr, "Scenario Setup");
            const auto id = resolveResourceId<BufferId>(_resourceIds, buffer->guid, "Buffer");
            if (buffer->src || !_groupManager.isAliased(id)) {
                const auto bufferData = loadBufferData(*buffer);
                upload(id, {bufferData.data.data(), bufferData.data.size()});
            }
        } break;
        default:
            // Only buffers, images, and tensors have JSON-provided runtime data to load.
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }
}

void Scenario::resolveCommands(ScenarioBuilder &builder, const ScenarioSpec &scenarioSpec,
                               const std::unordered_map<Guid, TypedResourceId> &resourceIds) {
    CommandDataFactory factory{builder._data.resources, resourceIds};
    for (const auto &command : scenarioSpec.commands) {
        switch (command->commandType) {
        case CommandType::DispatchCompute:
            builder.addDispatchCompute(factory.createData(reinterpret_cast<const DispatchComputeDesc &>(*command)));
            break;
        case CommandType::DispatchBarrier:
            builder.addDispatchBarrier(factory.createData(reinterpret_cast<const DispatchBarrierDesc &>(*command)));
            break;
        case CommandType::DispatchDataGraph:
            builder.addDispatchDataGraph(factory.createData(reinterpret_cast<const DispatchDataGraphDesc &>(*command)));
            break;
        case CommandType::DispatchSpirvGraph:
            builder.addDispatchSpirvGraph(
                factory.createData(reinterpret_cast<const DispatchSpirvGraphDesc &>(*command)));
            break;
        case CommandType::DispatchFragment:
            builder.addDispatchFragment(factory.createData(reinterpret_cast<const DispatchFragmentDesc &>(*command)));
            break;
        case CommandType::DispatchOpticalFlow:
            builder.addDispatchOpticalFlow(
                factory.createData(reinterpret_cast<const DispatchOpticalFlowDesc &>(*command)));
            break;
        case CommandType::MarkBoundary:
            builder.addMarkBoundary(factory.createData(reinterpret_cast<const MarkBoundaryDesc &>(*command)));
            break;
        default:
            throw std::runtime_error("Unknown CommandType in commands");
        }
    }
}

void Scenario::setupRuntimeCommands() {
    if (_opts.enablePipelineCaching) {
        mlsdk::logging::info("Load Pipeline Cache");
        PerfCounterGuard guard(_perfCounters, "Load Pipeline Cache.", "Load Pipeline Cache");
        _pipelineCache = std::make_shared<PipelineCache>(_ctx, _opts.pipelineCachePath, _opts.clearPipelineCache,
                                                         _opts.failOnPipelineCacheMiss);
    }
    // Setup commands
    mlsdk::logging::info("Setup commands");

    uint32_t nQueries = 0;
    const auto setupCommand = Overloaded{
        [&](const DispatchComputeData &data) { createComputePipeline(data, nQueries); },
        [&](const DispatchBarrierData &data) { _compute.registerPipelineBarrier(data, _dataManager); },
        [&](const DispatchDataGraphData &data) { createDataGraphPipeline(data, nQueries); },
        [&](const DispatchSpirvGraphData &data) { createSpirvGraphPipeline(data, nQueries); },
        [&](const DispatchFragmentData &data) { createFragmentPipeline(data, nQueries); },
        [&](const DispatchOpticalFlowData &data) {
            verifyOpticalFlowData(_dataManager, data);
            createOpticalFlowPipeline(data, nQueries);
        },
        [&](const MarkBoundaryData &data) {
            if (_ctx._optionals.mark_boundary) {
                _compute.registerMarkBoundary(data, _dataManager);
            } else {
                mlsdk::logging::warning("Frame boundary extension not present");
            }
        },
    };
    for (const auto &command : _commands) {
        std::visit(setupCommand, command);
    }
    if (_pipelineCache) {
        PerfCounterGuard guard(_perfCounters, "Save Pipeline Cache (setup)", "Save Pipeline Cache", false);
        _pipelineCache->save();
    }
    // Setup profiling
    if (!_opts.profilingPath.empty() && nQueries != 0) {
        mlsdk::logging::info("Setup profiling");
        _compute.setupQueryPool(nQueries);
    }
}

bool Scenario::hasAliasedOptimalTensors() const {
    // If any tensors in any memgroup have optimal tiling
    for ([[maybe_unused]] const auto &[_, resources] : _groupManager.getGroups()) {
        if (resources.size() <= 1) {
            continue;
        }
        for (const auto &resource : resources) {
            const auto *tensor = std::get_if<TensorId>(&resource);
            if (tensor != nullptr && _resources.get(*tensor).tiling == Tiling::Optimal) {
                return true;
            }
        }
    }
    return false;
}

void Scenario::handleAliasedLayoutTransitions() {

    // Validation pass: ensure all resources in a group have the same tiling type
    for ([[maybe_unused]] const auto &[_, resources] : _groupManager.getGroups()) {
        bool allLinear = true;
        bool allOptimal = true;
        for (const auto &resource : resources) {
            if (const auto *tensor = std::get_if<TensorId>(&resource)) {
                if (_resources.get(*tensor).tiling == Tiling::Linear) {
                    allOptimal = false;
                } else {
                    allLinear = false;
                }
            } else if (const auto *image = std::get_if<ImageId>(&resource)) {
                const auto tiling =
                    _resources.get(*image).tiling.value_or(resources.size() > 1 ? Tiling::Linear : Tiling::Optimal);
                if (tiling == Tiling::Linear) {
                    allOptimal = false;
                } else {
                    allLinear = false;
                }
            }
        }

        assert(!(allLinear && allOptimal));
        if (!allLinear && !allOptimal) {
            throw std::runtime_error("Aliased resources must have identical tiling.");
        }
    }

    std::unordered_set<MemoryResourceId> usedResources;
    const auto addBindings = [&](const std::vector<TypedBinding> &bindings) {
        for (const auto &binding : bindings) {
            usedResources.insert(binding.resource);
        }
    };
    const auto addCommandBindings = Overloaded{
        [&](const DispatchComputeData &dispatch) { addBindings(dispatch.bindings); },
        [&](const DispatchDataGraphData &dispatch) { addBindings(dispatch.bindings); },
        [&](const DispatchSpirvGraphData &dispatch) { addBindings(dispatch.bindings); },
        [&](const DispatchFragmentData &dispatch) { addBindings(dispatch.bindings); },
        [&](const DispatchOpticalFlowData &dispatch) {
            usedResources.insert(dispatch.searchImage.resource);
            usedResources.insert(dispatch.templateImage.resource);
            usedResources.insert(dispatch.outputImage.resource);
            if (dispatch.hintMotionVectors) {
                usedResources.insert(dispatch.hintMotionVectors->resource);
            }
            if (dispatch.outputCost) {
                usedResources.insert(dispatch.outputCost->resource);
            }
        },
        [](const DispatchBarrierData &) {},
        [](const MarkBoundaryData &) {},
    };
    for (const auto &command : _commands) {
        std::visit(addCommandBindings, command);
    }

    for ([[maybe_unused]] const auto &[_, resources] : _groupManager.getGroups()) {
        if (resources.size() <= 1) {
            continue;
        }

        for (const auto &resource : resources) {
            if (usedResources.count(resource) == 0) {
                continue;
            }
            if (const auto *tensor = std::get_if<TensorId>(&resource)) {
                if (_resources.get(*tensor).tiling != Tiling::Optimal) {
                    continue;
                }
                for (const auto &alias : resources) {
                    if (const auto *imageId = std::get_if<ImageId>(&alias)) {
                        auto &image = _dataManager.getImageMut(*imageId);
                        if (image.getImageLayout() != vk::ImageLayout::eTensorAliasingARM) {
                            image.addTransitionLayoutCommand(_compute.getCommandBuffer(),
                                                             vk::ImageLayout::eTensorAliasingARM);
                        }
                    }
                }
            } else if (const auto *imageId = std::get_if<ImageId>(&resource)) {
                const auto &imageInfo = _resources.get(*imageId);
                if (imageInfo.tiling != Tiling::Optimal) {
                    continue;
                }
                auto &image = _dataManager.getImageMut(*imageId);
                const auto targetLayout = imageInfo.isSampled && !imageInfo.isStorage
                                              ? vk::ImageLayout::eShaderReadOnlyOptimal
                                              : vk::ImageLayout::eGeneral;
                if (image.getImageLayout() != targetLayout) {
                    image.addTransitionLayoutCommand(_compute.getCommandBuffer(), targetLayout);
                }
            }
        }
    }
}

std::pair<const char *, size_t> getPushConstantData(const std::optional<RawDataId> &pushData,
                                                    const DataManager &dataManager) {
    if (pushData) {
        const auto &rawData = dataManager.getRawData(pushData.value());
        return std::make_pair(rawData.data(), rawData.size());
    }
    return std::make_pair(nullptr, 0U);
}

void Scenario::createComputePipeline(const DispatchComputeData &dispatchCompute, uint32_t &nQueries) {
    // Create Compute shader pipeline
    const auto &shaderInfo = getShader(dispatchCompute.shader);
    if (shaderInfo.stage != ShaderStage::Compute && shaderInfo.stage != ShaderStage::Unknown) {
        throw std::runtime_error("DispatchCompute requires a compute shader stage, given: " +
                                 std::to_string(static_cast<int>(shaderInfo.stage)));
    }
    const Compute::PipelineCreateArguments args{dispatchCompute.debugName, dispatchCompute.bindings, _pipelineCache};

    PerfCounterGuard guard(_perfCounters, "Create Pipeline: " + shaderInfo.debugName, "Pipeline Setup");
    _compute.createPipeline(args, shaderInfo);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
    const auto [pushConstantData, pushConstantSize] = getPushConstantData(dispatchCompute.pushData, _dataManager);
    _compute.registerPipelineFenced(_dataManager, dispatchCompute.bindings, pushConstantData, pushConstantSize,
                                    dispatchCompute.implicitBarrier, dispatchCompute.computeDispatch);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
    mlsdk::logging::debug("Shader Pipeline: " + shaderInfo.debugName + " created");
}

void Scenario::createFragmentPipeline(const DispatchFragmentData &dispatchFragment, uint32_t &nQueries) {
    const auto &vertexShaderInfo = getShader(dispatchFragment.vertexShader);
    const auto &fragmentShaderInfo = getShader(dispatchFragment.fragmentShader);
    if (vertexShaderInfo.stage != ShaderStage::Vertex) {
        throw std::runtime_error("dispatch_fragment vertex_shader_ref must reference a vertex shader");
    }
    if (fragmentShaderInfo.stage != ShaderStage::Fragment) {
        throw std::runtime_error("dispatch_fragment fragment_shader_ref must reference a fragment shader");
    }

    const Compute::PipelineCreateArguments args{dispatchFragment.debugName, dispatchFragment.bindings, _pipelineCache};
    PerfCounterGuard guard(_perfCounters, "Create Graphics Pipeline: " + fragmentShaderInfo.debugName,
                           "Pipeline Setup");

    std::vector<vk::Format> colorAttachmentFormats;
    std::vector<GraphicsDispatchAttachment> attachmentInfos;
    colorAttachmentFormats.reserve(dispatchFragment.colorAttachments.size());
    attachmentInfos.reserve(dispatchFragment.colorAttachments.size());

    std::optional<vk::Extent2D> targetExtent = dispatchFragment.renderExtent;
    for (const auto &attachmentSpec : dispatchFragment.colorAttachments) {
        const auto &colorImage = _dataManager.getImage(attachmentSpec.resource);
        const auto &imageInfo = colorImage.getInfo();
        const auto &shape = colorImage.shape();
        if (shape.size() < 3) {
            throw std::runtime_error("Color attachment image does not have enough dimensions for rendering");
        }
        auto attachmentWidth = static_cast<uint32_t>(shape[1]);
        auto attachmentHeight = static_cast<uint32_t>(shape[2]);
        if (attachmentSpec.lod.has_value()) {
            const uint32_t lod = attachmentSpec.lod.value();
            if (lod >= imageInfo.mips) {
                throw std::runtime_error("Color attachment mip level exceeds available mips");
            }
            const uint32_t divisor = 1u << lod;
            attachmentWidth = std::max(1u, attachmentWidth / divisor);
            attachmentHeight = std::max(1u, attachmentHeight / divisor);
        }
        const vk::Extent2D extent(attachmentWidth, attachmentHeight);
        if (!targetExtent.has_value()) {
            targetExtent = extent;
        } else if (targetExtent.value() != extent) {
            throw std::runtime_error("All color attachments must share the same extent");
        }

        colorAttachmentFormats.push_back(imageInfo.targetFormat);
        GraphicsDispatchAttachment attachment{};
        attachment.view =
            attachmentSpec.lod.has_value() ? colorImage.imageView(attachmentSpec.lod.value()) : colorImage.imageView();
        attachment.image = colorImage.image();
        attachment.layout = colorImage.getImageLayout();
        attachmentInfos.push_back(attachment);
    }

    if (!targetExtent.has_value()) {
        throw std::runtime_error("dispatch_fragment requires render_extent when no color attachments are provided");
    }

    _compute.createPipeline(args, vertexShaderInfo, fragmentShaderInfo, colorAttachmentFormats);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    const auto [pushConstantData, pushConstantSize] = getPushConstantData(dispatchFragment.pushData, _dataManager);

    GraphicsDispatchInfo dispatchInfo{};
    dispatchInfo.colorAttachments = std::move(attachmentInfos);
    dispatchInfo.extent = targetExtent.value();

    _compute.registerPipelineFenced(_dataManager, dispatchFragment.bindings, pushConstantData, pushConstantSize,
                                    dispatchFragment.implicitBarrier, dispatchInfo);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    mlsdk::logging::debug("Graphics Pipeline: " + fragmentShaderInfo.debugName + " created");
}

void Scenario::createDataGraphPipeline(const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries) {
    const VgfView &vgfView = _dataManager.getVgfView(dispatchDataGraph.dataGraph);
    for (uint32_t segmentIndex = 0; segmentIndex < vgfView.getNumSegments(); ++segmentIndex) {
        const auto &intermediates = _vgfResourceCreationResults.at(dispatchDataGraph.dataGraph).intermediateResources;
        const auto &sequenceBindings =
            vgfView.resolveBindings(segmentIndex, _dataManager, dispatchDataGraph.bindings, intermediates);
        auto moduleName = vgfView.getModuleName(segmentIndex);
        PerfCounterGuard guard(_perfCounters, "Create Pipeline: " + moduleName, "Pipeline Setup");
        createPipeline(segmentIndex, sequenceBindings, vgfView, dispatchDataGraph, nQueries);
    }
}

void Scenario::createSpirvGraphPipeline(const DispatchSpirvGraphData &dispatchSpirvGraph, uint32_t &nQueries) {
    const auto &shaderInfo = getShader(dispatchSpirvGraph.graphShader);
    if (shaderInfo.shaderType != ShaderType::SPIR_V) {
        throw std::runtime_error("Shader resource used to create Graph Pipeline must be of type SPIR-V");
    }

    if (shaderInfo.src.empty()) {
        throw std::runtime_error("Shader resource missing src: " + shaderInfo.debugName);
    }

    // Validate the bindings
    const auto &sequenceBindings = dispatchSpirvGraph.bindings;
    for (const auto &binding : sequenceBindings) {
        if (std::holds_alternative<TensorId>(binding.resource)) {
            if (binding.vkDescriptorType != vk::DescriptorType::eTensorARM) {
                throw std::runtime_error("DataGraph tensor binding must use a tensor descriptor");
            }
            continue;
        }
        if (std::holds_alternative<ImageId>(binding.resource)) {
            if ((binding.vkDescriptorType != vk::DescriptorType::eStorageImage) &&
                (binding.vkDescriptorType != vk::DescriptorType::eCombinedImageSampler)) {
                throw std::runtime_error("DataGraph image binding must use an image descriptor");
            }
            continue;
        }
        throw std::runtime_error("No resource with this guid found");
    }

    const auto graphConstants = collectGraphConstants(dispatchSpirvGraph.graphConstants, _resources);

    // Create pipeline and record DataGraph dispatch
    PerfCounterGuard guard(_perfCounters, "Create Pipeline: " + shaderInfo.debugName, "Pipeline Setup");
    const Compute::PipelineCreateArguments args{dispatchSpirvGraph.debugName, sequenceBindings, _pipelineCache};
    _compute.createPipeline(args, shaderInfo, _dataManager, graphConstants, _opts.shouldDumpNeuralStatistics(),
                            _opts.neuralStatisticsMode);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
    _compute.registerPipelineFenced(_dataManager, sequenceBindings, nullptr, 0, dispatchSpirvGraph.implicitBarrier);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
    mlsdk::logging::debug("Graph Pipeline: " + shaderInfo.debugName + " created");
}

void Scenario::createOpticalFlowPipeline(const DispatchOpticalFlowData &dispatchOpticalFlow, uint32_t &nQueries) {
    const std::vector<TypedBinding> emptyBindings{};
    const Compute::PipelineCreateArguments args{dispatchOpticalFlow.debugName, emptyBindings, _pipelineCache};

    const auto perfLevel = getOpticalFlowPerformanceLevel(dispatchOpticalFlow.performanceLevel);
    const auto gridSize = getOpticalFlowGridSize(dispatchOpticalFlow.gridSize);

    PerfCounterGuard guard(_perfCounters, "Create Optical Flow Pipeline: " + dispatchOpticalFlow.debugName,
                           "Pipeline Setup");

    std::vector<TypedBinding> bindings;
    bindings.reserve(5);
    bindings.emplace_back(dispatchOpticalFlow.searchImage);
    bindings.emplace_back(dispatchOpticalFlow.templateImage);
    bindings.emplace_back(dispatchOpticalFlow.outputImage);
    if (dispatchOpticalFlow.hintMotionVectors.has_value()) {
        bindings.emplace_back(dispatchOpticalFlow.hintMotionVectors.value());
    }
    if (dispatchOpticalFlow.outputCost.has_value()) {
        bindings.emplace_back(dispatchOpticalFlow.outputCost.value());
    }
    _compute.createPipeline(args, _dataManager, dispatchOpticalFlow.searchImage, dispatchOpticalFlow.templateImage,
                            dispatchOpticalFlow.outputImage, dispatchOpticalFlow.hintMotionVectors,
                            dispatchOpticalFlow.outputCost, perfLevel, gridSize, dispatchOpticalFlow.width,
                            dispatchOpticalFlow.height);

    // Optical flow is a data graph pipeline; profile it as such.
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);

    Compute::OpticalFlowDispatchInfo dispatchInfo{};
    dispatchInfo.opticalFlowFlags = vk::DataGraphOpticalFlowExecuteFlagsARM{dispatchOpticalFlow.executionFlags};
    dispatchInfo.meanFlowL1NormHint = dispatchOpticalFlow.meanFlowL1NormHint;

    _compute.registerPipelineFenced(_dataManager, bindings, nullptr, 0, dispatchOpticalFlow.implicitBarrier, {},
                                    dispatchInfo);

    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);

    mlsdk::logging::debug("Optical Flow Pipeline: " + dispatchOpticalFlow.debugName + " created");
}

void Scenario::createPipeline(const uint32_t segmentIndex, const std::vector<TypedBinding> &sequenceBindings,
                              const VgfView &vgfView, const DispatchDataGraphData &dispatchDataGraph,
                              uint32_t &nQueries) {
    const auto profileName = dispatchDataGraph.debugName + "/" + vgfView.getSegmentName(segmentIndex);
    const Compute::PipelineCreateArguments args{profileName, sequenceBindings, _pipelineCache};
    switch (vgfView.getSegmentType(segmentIndex)) {
    case ModuleType::GRAPH: {
        _compute.createPipeline(args, segmentIndex, vgfView, _dataManager, _opts.shouldDumpNeuralStatistics(),
                                _opts.neuralStatisticsMode);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
        _compute.registerPipelineFenced(_dataManager, sequenceBindings, nullptr, 0, dispatchDataGraph.implicitBarrier);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
        mlsdk::logging::debug("Graph Pipeline: " + vgfView.getModuleName(segmentIndex) + " created");
    } break;
    case ModuleType::SHADER: {
        const auto &dataGraph = _resources.get(dispatchDataGraph.dataGraph);
        const auto moduleName = vgfView.getModuleName(segmentIndex);
        bool hasSPVModule = vgfView.hasSPVModule(segmentIndex);
        bool hasGLSLModule = vgfView.hasGLSLModule(segmentIndex);
        bool hasHLSLModule = vgfView.hasHLSLModule(segmentIndex);

        if (!dispatchDataGraph.shaderSubstitutions.empty()) {
            ShaderInfo shaderInfo = getSubstitutionShader(dispatchDataGraph.shaderSubstitutions, moduleName);
            applyGraphResourceShaderMetadata(shaderInfo, dataGraph, moduleName);
            _compute.createPipeline(args, shaderInfo);
            if (hasSPVModule || hasGLSLModule || hasHLSLModule) {
                mlsdk::logging::warning("Performing shader substitution despite shader module containing code");
            }
        } else {
            ShaderInfo shaderInfo;
            shaderInfo.debugName = vgfView.getModuleName(segmentIndex);
            shaderInfo.entry = vgfView.getModuleEntryPoint(segmentIndex);
            shaderInfo.shaderType = ShaderType::SPIR_V;
            shaderInfo.stage = ShaderStage::Compute;
            applyGraphResourceShaderMetadata(shaderInfo, dataGraph, moduleName);

            if (hasSPVModule) {
                auto spv = vgfView.getSPVModuleCode(segmentIndex);
                _compute.createPipeline(args, shaderInfo, spv.begin(), spv.size());
            } else if (hasGLSLModule) {
                const auto spirv =
                    GlslCompiler::get().compile(vgfView.getGLSLModuleCode(segmentIndex), shaderInfo.stage);
                if (!spirv.first.empty()) {
                    throw std::runtime_error("Compilation error\n" + spirv.first);
                }
                _compute.createPipeline(args, shaderInfo, spirv.second.data(), spirv.second.size());
            } else if (hasHLSLModule) {
#ifdef SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT
                const auto spirv = HlslCompiler::get().compile(vgfView.getHLSLModuleCode(segmentIndex),
                                                               shaderInfo.entry, shaderInfo.debugName);
                if (!spirv.first.empty()) {
                    throw std::runtime_error("Compilation error\n" + spirv.first);
                }
                _compute.createPipeline(args, shaderInfo, spirv.second.data(), spirv.second.size());
#else
                throw std::runtime_error("HLSL shaders are not supported on this platform.");
#endif
            } else {
                throw std::runtime_error("No shader module present and no shader substituion defined.");
            }
        }

        auto dispatchShape = vgfView.getDispatchShape(segmentIndex);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
        const auto [pushConstantData, pushConstantSize] =
            getPushConstantData(getGraphPushData(dispatchDataGraph.pushConstants, moduleName), _dataManager);
        _compute.registerPipelineFenced(_dataManager, sequenceBindings, pushConstantData, pushConstantSize,
                                        dispatchDataGraph.implicitBarrier,
                                        {dispatchShape[0], dispatchShape[1], dispatchShape[2], profileName});
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
        mlsdk::logging::debug("Shader Pipeline: " + vgfView.getModuleName(segmentIndex) + " created");
    } break;
    default:
        throw std::runtime_error("Unknown module type");
    }
}

void Scenario::saveProfilingData(int iteration, int repeatCount, bool dryRun) {
    // Save profiling data
    if (!_opts.profilingPath.empty()) {
        std::optional<RuntimeProfilingData> runtimeProfilingData;
        if (!dryRun) {
            runtimeProfilingData = _compute.getRuntimeProfilingData();
        }
        const auto memoryProfilingData = _compute.getMemoryProfilingData();
        writeProfilingData(runtimeProfilingData, memoryProfilingData, _opts.profilingPath, iteration, repeatCount);
        mlsdk::logging::info("Profiling data stored");
    }
}

void Scenario::saveResults(bool dryRun) {
    if (_pipelineCache) {
        PerfCounterGuard guard(_perfCounters, "Save Pipeline Cache", "Save Pipeline Cache", false);
        _pipelineCache->save();
    }

    // Performance counters should be stored also for dry runs
    ScopeExit<void()> onExit([&]() {
        // Save performance counters
        if (!_opts.perfCountersPath.empty()) {
            writePerfCounters(_perfCounters, _opts.perfCountersPath);
            mlsdk::logging::info("Performance stats stored");
        }
    });

    if (dryRun) {
        return;
    }

    // Save resources that have an output destination
    {
        PerfCounterGuard guard(_perfCounters, "Save Resources", "Save Results", false);
        for (const auto &resourceDesc : _scenarioSpec.resources) {
            const auto &dst = resourceDesc->getDestination();
            if (dst.has_value()) {
                switch (resourceDesc->resourceType) {
                case ResourceType::Buffer:
                    _dataManager.getBuffer(resolveResourceId<BufferId>(_resourceIds, resourceDesc->guid, "Buffer"))
                        .store(_ctx, dst.value());
                    break;
                case ResourceType::Tensor:
                    _dataManager.getTensor(resolveResourceId<TensorId>(_resourceIds, resourceDesc->guid, "Tensor"))
                        .store(_ctx, dst.value());
                    break;
                case ResourceType::Image:
                    _dataManager.getImageMut(resolveResourceId<ImageId>(_resourceIds, resourceDesc->guid, "Image"))
                        .store(_ctx, dst.value());
                    break;
                default:
                    throw std::runtime_error("Output destination is not supported for " + resourceType(resourceDesc) +
                                             " resource " + resourceDesc->guidStr);
                }
                mlsdk::logging::debug(resourceType(resourceDesc) + " " + resourceDesc->guidStr + " output stored");
            }
        }
    }
    mlsdk::logging::info("Results stored");

    // Store Neural Debug Database Dump
    if (_opts.shouldDumpNeuralDebugDatabase()) {
        _compute.dumpNeuralDebugDatabase(_opts.neuralDebugDatabaseDumpDir);
    }

    // Store Neural Statistics Dump
    if (_opts.shouldDumpNeuralStatistics()) {
        _compute.dumpNeuralStatistics(_opts.neuralStatisticsDumpDir, _opts.neuralStatisticsMode);
    }

    // Store Graph Profiling Data Dump
    if (_opts.shouldDumpGraphProfiling()) {
        _compute.dumpGraphProfilingData(_opts.graphProfilingDumpDir);
    }

    // Hexdump the session ram for debugging
    if (!_opts.sessionRAMsDumpDir.empty()) {
        _compute.sessionRAMsDump(_opts.sessionRAMsDumpDir);
    }
}

} // namespace mlsdk::scenariorunner
