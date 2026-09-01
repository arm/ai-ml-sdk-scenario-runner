/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_json_factory.hpp"

#include "image_formats.hpp"
#include "logging.hpp"
#include "scenario.hpp"
#include "utils.hpp"

#include "vgf-utils/memory_map.hpp"
#include "vgf-utils/numpy.hpp"

#include <algorithm>
#include <cstring>
#include <string_view>

namespace mlsdk::scenariorunner {
namespace {
ImageLoadResult loadDataFromNPY(const std::string &filename, vk::Format dataType, const ImageLoadOptions &options) {
    MemoryMap mapped(filename);
    auto dataPtr = vgfutils::numpy::parse(mapped);

    if (dataPtr.shape.size() != 4) {
        throw std::runtime_error("Image data must be 4 dimensional for npy sources");
    }

    if (dataPtr.shape[0] != 1) {
        throw std::runtime_error("Image batch dimension must be 1 for npy sources");
    }

    if ((dataPtr.shape[1] != static_cast<int64_t>(options.expectedHeight)) ||
        (dataPtr.shape[2] != static_cast<int64_t>(options.expectedWidth)) ||
        (dataPtr.shape[3] != static_cast<int64_t>(numComponentsFromVkFormat(dataType)))) {
        throw std::runtime_error("Image description dimensions do not match npy data shape");
    }

    const auto expectedSize = static_cast<uint64_t>(dataPtr.shape[0]) * options.expectedHeight * options.expectedWidth *
                              elementSizeFromVkFormat(dataType);

    if (dataPtr.size() != expectedSize) {
        throw std::runtime_error("Image description size does not match data size: expected " +
                                 std::to_string(expectedSize) + " vs " + std::to_string(dataPtr.size()));
    }

    ImageLoadResult result(dataType, options.expectedWidth, options.expectedHeight);
    result.data.resize(dataPtr.size());
    std::memcpy(result.data.data(), dataPtr.ptr, dataPtr.size());
    return result;
}

ImageLoadResult loadData(const std::string &fileName, vk::Format dataType, const ImageLoadOptions &options) {
    if (const auto *handler = getImageFormatHandler(fileName); handler) {
        return handler->loadData(fileName, options);
    }

    if (lowercaseExtension(fileName) == ".npy") {
        return loadDataFromNPY(fileName, dataType, options);
    }

    throw std::runtime_error("Unsupported image source file type for " + fileName);
}

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

ImageData loadImageData(const ImageDesc &desc, const ImageInfo &info) {
    const auto dataType = info.format == vk::Format::eD32SfloatS8Uint ? vk::Format::eD32Sfloat : info.format;
    const auto baseDataSize =
        static_cast<size_t>(elementSizeFromVkFormat(dataType) * totalElementsFromShape(info.shape));

    std::vector<std::byte> data;
    vk::Format fileFormat = vk::Format::eUndefined;
    uint32_t mipLevels = 1;
    if (desc.src.has_value()) {
        auto result = loadData(desc.src.value(), dataType, ImageLoadOptions{desc.dims[2], desc.dims[1]});
        data = std::move(result.data);
        fileFormat = result.initialFormat;
        mipLevels = result.mipLevels;
    } else {
        data.resize(baseDataSize, std::byte{0});
    }

    if ((dataType == vk::Format::eR32Sfloat || dataType == vk::Format::eD32Sfloat) &&
        fileFormat == vk::Format::eD32SfloatS8Uint) {
        std::vector<std::byte> depthData(baseDataSize);
        bool hasStencilData = false;
        for (uint64_t i = 0; i < totalElementsFromShape(info.shape); ++i) {
            const auto depthStencilIndex = i * elementSizeFromVkFormat(fileFormat);
            const auto depthIndex = i * elementSizeFromVkFormat(dataType);
            if (data[depthStencilIndex + 4] != std::byte{0}) {
                hasStencilData = true;
            }
            std::memcpy(depthData.data() + depthIndex, data.data() + depthStencilIndex,
                        elementSizeFromVkFormat(dataType));
        }
        if (hasStencilData) {
            mlsdk::logging::warning("Ignoring stencil data");
        }
        data = std::move(depthData);
    }

    ImageData imageData;
    imageData.data = std::move(data);
    imageData.shape = info.shape;
    imageData.format = dataType;
    imageData.mipLevels = mipLevels;
    return imageData;
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
    struct DescriptorTypeVisitor {
        const ResourceManager &resources;

        vk::DescriptorType operator()(BufferId id) const {
            static_cast<void>(resources.get(id));
            return vk::DescriptorType::eStorageBuffer;
        }

        vk::DescriptorType operator()(TensorId id) const {
            static_cast<void>(resources.get(id));
            return vk::DescriptorType::eTensorARM;
        }

        vk::DescriptorType operator()(ImageId id) const {
            return resources.get(id).isSampled ? vk::DescriptorType::eCombinedImageSampler
                                               : vk::DescriptorType::eStorageImage;
        }
    };

    return std::visit(DescriptorTypeVisitor{resources}, resource);
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

template <typename Id>
void registerResourceId(std::unordered_map<Guid, TypedResourceId> &resourceIds, const Guid &guid,
                        std::string_view guidName, Id id) {
    if (!resourceIds.emplace(guid, id).second) {
        throw std::runtime_error("Duplicate resource UID: " + std::string(guidName));
    }
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

} // namespace

std::unique_ptr<IScenario> ScenarioJsonFactory::make(const ScenarioOptions &options, const ScenarioSpec &scenarioSpec) {
    ScenarioBuilder builder;
    populate(options, scenarioSpec, builder);
    return builder.build(options);
}

void ScenarioJsonFactory::populate(const ScenarioOptions &options, const ScenarioSpec &scenarioSpec,
                                   ScenarioBuilder &builder) {
    mlsdk::logging::info("Setup resources, count: " + std::to_string(scenarioSpec.resources.size()));
    // Setup resource info
    // (Memory for Tensors and Images is allocated in next pass)
    ResourceInfoFactory resourceInfoFactory;
    std::unordered_map<Guid, MemoryGroupId> jsonMemoryGroupIds;
    std::unordered_map<Guid, TypedResourceId> resourceIds;

    for (const auto &resource : scenarioSpec.resources) {
        switch (resource->resourceType) {
        case ResourceType::Buffer: {
            const auto &buffer = reinterpret_cast<const std::unique_ptr<BufferDesc> &>(resource);
            const auto id = builder.addBuffer(resourceInfoFactory.createInfo(*buffer));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
            registerMemoryGroup(builder, jsonMemoryGroupIds, id, buffer->memoryGroup);
        } break;
        case ResourceType::RawData: {
            const auto &rawData = reinterpret_cast<const std::unique_ptr<RawDataDesc> &>(resource);
            const auto id = builder.addRawData(resourceInfoFactory.createInfo(*rawData));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        case ResourceType::Image: {
            const auto &image = reinterpret_cast<const std::unique_ptr<ImageDesc> &>(resource);
            const auto id = builder.addImage(resourceInfoFactory.createInfo(*image));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
            registerMemoryGroup(builder, jsonMemoryGroupIds, id, image->memoryGroup);
        } break;
        case ResourceType::DataGraph: {
            const auto &dataGraph = reinterpret_cast<const std::unique_ptr<DataGraphDesc> &>(resource);
            const auto id = builder.addDataGraph(resourceInfoFactory.createInfo(*dataGraph));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        case ResourceType::Tensor: {
            const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource);
            const auto id = builder.addTensor(resourceInfoFactory.createInfo(*tensor, options.captureFrame));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
            registerMemoryGroup(builder, jsonMemoryGroupIds, id, tensor->memoryGroup);
        } break;
        case ResourceType::Shader: {
            const auto &shader = reinterpret_cast<const std::unique_ptr<ShaderDesc> &>(resource);
            const auto id = builder.addShader(resourceInfoFactory.createInfo(*shader));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        case ResourceType::GraphConstant: {
            const auto &graphConstant = reinterpret_cast<const std::unique_ptr<GraphConstantDesc> &>(resource);
            const auto id = builder.addGraphConstant(resourceInfoFactory.createInfo(*graphConstant));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }

    // Barrier descriptions can reference resources declared later in the JSON,
    // so translate them only after all regular resources have typed IDs.
    BarrierInfoFactory barrierInfoFactory{resourceIds};
    for (const auto &resource : scenarioSpec.resources) {
        switch (resource->resourceType) {
        case ResourceType::ImageBarrier: {
            const auto &imageBarrier = reinterpret_cast<const std::unique_ptr<ImageBarrierDesc> &>(resource);
            const auto id = builder.addImageBarrier(barrierInfoFactory.createInfo(*imageBarrier));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        case ResourceType::MemoryBarrier: {
            const auto &memoryBarrier = reinterpret_cast<const std::unique_ptr<MemoryBarrierDesc> &>(resource);
            const auto id = builder.addMemoryBarrier(barrierInfoFactory.createInfo(*memoryBarrier));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        case ResourceType::TensorBarrier: {
            const auto &tensorBarrier = reinterpret_cast<const std::unique_ptr<TensorBarrierDesc> &>(resource);
            const auto id = builder.addTensorBarrier(barrierInfoFactory.createInfo(*tensorBarrier));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        case ResourceType::BufferBarrier: {
            const auto &bufferBarrier = reinterpret_cast<const std::unique_ptr<BufferBarrierDesc> &>(resource);
            const auto id = builder.addBufferBarrier(barrierInfoFactory.createInfo(*bufferBarrier));
            registerResourceId(resourceIds, resource->guid, resource->guidStr, id);
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }

    resolveCommands(builder, scenarioSpec, resourceIds);

    auto &buildData = detail::ScenarioBuilderAccess::buildData(builder);
    for (const auto &resource : scenarioSpec.resources) {
        switch (resource->resourceType) {
        case ResourceType::Buffer: {
            const auto &buffer = static_cast<const BufferDesc &>(*resource);
            const auto id = resolveResourceId<BufferId>(resourceIds, resource->guid, "Buffer");
            if (buffer.src.has_value() || !buildData.groupManager.isAliased(id)) {
                buildData.initializations.emplace_back(
                    detail::BufferInitialization{id, loadBufferData(buffer), buffer.guidStr});
            }
        } break;
        case ResourceType::Image: {
            const auto &image = static_cast<const ImageDesc &>(*resource);
            const auto id = resolveResourceId<ImageId>(resourceIds, resource->guid, "Image");
            std::optional<ImageData> data;
            if (image.src.has_value()) {
                data = loadImageData(image, buildData.resources.get(id));
            }
            buildData.initializations.emplace_back(detail::ImageInitialization{id, std::move(data), image.guidStr});
        } break;
        case ResourceType::Tensor: {
            const auto &tensor = static_cast<const TensorDesc &>(*resource);
            const auto id = resolveResourceId<TensorId>(resourceIds, resource->guid, "Tensor");
            if (tensor.src.has_value() || !buildData.groupManager.isAliased(id)) {
                buildData.initializations.emplace_back(
                    detail::TensorInitialization{id, loadTensorData(tensor), tensor.guidStr});
            }
        } break;
        default:
            break;
        }

        if (resource->getDestination().has_value()) {
            buildData.outputs.push_back({resolveTypedResourceId(resourceIds, resource->guid, resourceType(resource)),
                                         resource->getDestination().value(), resource->guidStr});
        }
    }
    buildData.resourceIds = std::move(resourceIds);
}

void ScenarioJsonFactory::resolveCommands(ScenarioBuilder &builder, const ScenarioSpec &scenarioSpec,
                                          const std::unordered_map<Guid, TypedResourceId> &resourceIds) {
    CommandDataFactory factory{detail::ScenarioBuilderAccess::buildData(builder).resources, resourceIds};
    for (const auto &command : scenarioSpec.commands) {
        switch (command->commandType) {
        case CommandType::DispatchCompute:
            builder.addDispatchCompute(factory.createData(reinterpret_cast<DispatchComputeDesc &>(*command)));
            break;
        case CommandType::DispatchBarrier:
            builder.addDispatchBarrier(factory.createData(reinterpret_cast<DispatchBarrierDesc &>(*command)));
            break;
        case CommandType::DispatchDataGraph:
            builder.addDispatchDataGraph(factory.createData(reinterpret_cast<DispatchDataGraphDesc &>(*command)));
            break;
        case CommandType::DispatchSpirvGraph:
            builder.addDispatchSpirvGraph(factory.createData(reinterpret_cast<DispatchSpirvGraphDesc &>(*command)));
            break;
        case CommandType::DispatchFragment:
            builder.addDispatchFragment(factory.createData(reinterpret_cast<DispatchFragmentDesc &>(*command)));
            break;
        case CommandType::DispatchOpticalFlow:
            builder.addDispatchOpticalFlow(factory.createData(reinterpret_cast<DispatchOpticalFlowDesc &>(*command)));
            break;
        case CommandType::MarkBoundary:
            builder.addMarkBoundary(factory.createData(reinterpret_cast<MarkBoundaryDesc &>(*command)));
            break;
        default:
            throw std::runtime_error("Unknown CommandType in commands");
        }
    }
}

} // namespace mlsdk::scenariorunner
