/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "scenario.hpp"
#include "dds_reader.hpp"
#include "frame_capturer.hpp"
#include "glsl_compiler.hpp"
#ifdef SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT
#    include "hlsl_compiler.hpp"
#endif
#include "guid.hpp"
#include "image_formats.hpp"
#include "iresource.hpp"
#include "json_writer.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include "vgf-utils/numpy.hpp"

#include <algorithm>
#include <unordered_set>

namespace mlsdk::scenariorunner {
/// \brief Compute data with typed bindings
struct DispatchComputeData {
    std::string debugName;
    std::vector<TypedBinding> bindings;
    ComputeDispatch computeDispatch{};
    Guid shaderRef;
    bool implicitBarrier{true};
    std::optional<Guid> pushDataRef;
};

/// \brief Fragment (graphics) data with typed bindings
struct DispatchFragmentData {
    std::string debugName;
    std::vector<TypedBinding> bindings;
    Guid vertexShaderRef;
    Guid fragmentShaderRef;
    struct Attachment {
        Guid resourceRef;
        std::optional<uint32_t> lod;
    };
    std::vector<Attachment> colorAttachments;
    std::optional<vk::Extent2D> renderExtent;
    bool implicitBarrier{true};
    std::optional<Guid> pushDataRef;
};

/// \brief Compute data graph with typed bindings
struct DispatchDataGraphData {
    Guid dataGraphRef;
    std::string debugName;
    std::vector<TypedBinding> bindings;
    std::vector<PushConstantMap> pushConstants;
    std::vector<ShaderSubstitution> shaderSubstitutions;
    bool implicitBarrier{true};
};

/// \brief SPIR-V-only data graph with typed bindings and constants
struct DispatchSpirvGraphData {
    Guid dataGraphRef;
    std::string debugName;
    std::vector<TypedBinding> bindings;
    std::vector<Guid> graphConstants;
    bool implicitBarrier{true};
};

/// \brief Optical flow data graph with typed bindings
struct DispatchOpticalFlowData {
    std::string debugName;
    TypedBinding searchImage;
    TypedBinding templateImage;
    TypedBinding outputImage;
    std::optional<TypedBinding> hintMotionVectors;
    std::optional<TypedBinding> outputCost;

    uint32_t width{0};
    uint32_t height{0};
    OpticalFlowPerformanceLevel performanceLevel{OpticalFlowPerformanceLevel::Medium};
    uint32_t executionFlags{0};
    OpticalFlowGridSize gridSize{OpticalFlowGridSize::e1x1};
    uint32_t meanFlowL1NormHint{0};

    bool implicitBarrier{true};
};

namespace {
std::vector<GraphConstantInfo> collectGraphConstants(const std::vector<Guid> &constantUids,
                                                     const std::vector<std::unique_ptr<ResourceDesc>> &resources) {
    std::vector<GraphConstantInfo> constants;
    constants.reserve(constantUids.size());

    // Build a quick lookup map from Guid -> GraphConstantDesc*
    std::unordered_map<Guid, const GraphConstantDesc *> gcMap;
    gcMap.reserve(resources.size());
    for (const auto &res : resources) {
        if (res->resourceType == ResourceType::GraphConstant) {
            gcMap.emplace(res->guid, static_cast<const GraphConstantDesc *>(res.get()));
        }
    }

    for (const auto &uid : constantUids) {
        auto it = gcMap.find(uid);
        if (it == gcMap.end()) {
            throw std::runtime_error("Graph constant not found for provided GUID");
        }
        const GraphConstantDesc *gc = it->second;
        if (!gc->src.has_value()) {
            throw std::runtime_error("Graph constant missing src: " + gc->guidStr);
        }

        GraphConstantInfo spec(gc->guidStr, getVkFormatFromString(gc->format), gc->dims,
                               static_cast<uint32_t>(constants.size()));

        MemoryMap mapped(gc->src.value());
        const auto constantData = vgfutils::numpy::parse(mapped);

        if (constantData.shape.size() == spec.dims.size()) {
            for (size_t i = 0; i < spec.dims.size(); ++i) {
                if (spec.dims[i] != constantData.shape[i]) {
                    throw std::runtime_error("Graph constant dims mismatch for: " + gc->guidStr);
                }
            }
        } else {
            throw std::runtime_error("Graph constant dims mismatch for: " + gc->guidStr);
        }

        // Validate that the NumPy payload size matches the declared format and shape
        const uint64_t expectedDataSize =
            static_cast<uint64_t>(elementSizeFromVkFormat(spec.format)) * totalElementsFromShape(spec.dims);
        const auto actualDataSize = static_cast<uint64_t>(constantData.size());
        if (actualDataSize != expectedDataSize) {
            throw std::runtime_error("Graph constant size does not match format and dims for: " + gc->guidStr +
                                     "; expected " + std::to_string(expectedDataSize) + " vs " +
                                     std::to_string(actualDataSize));
        }

        spec.data.resize(static_cast<size_t>(actualDataSize));
        std::memcpy(spec.data.data(), constantData.ptr, static_cast<size_t>(actualDataSize));

        constants.emplace_back(std::move(spec));
    }

    return constants;
}

std::string resourceType(const std::unique_ptr<ResourceDesc> &resource) {
    switch (resource->resourceType) {
    case (ResourceType::Unknown):
        return "Unknown";
    case (ResourceType::Buffer):
        return "Buffer";
    case (ResourceType::DataGraph):
        return "DataGraph";
    case (ResourceType::Shader):
        return "Shader";
    case (ResourceType::RawData):
        return "RawData";
    case (ResourceType::Tensor):
        return "Tensor";
    case (ResourceType::Image):
        return "Image";
    case (ResourceType::ImageBarrier):
        return "ImageBarrier";
    case (ResourceType::MemoryBarrier):
        return "MemoryBarrier";
    case (ResourceType::TensorBarrier):
        return "TensorBarrier";
    case (ResourceType::BufferBarrier):
        return "BufferBarrier";
    case (ResourceType::GraphConstant):
        return "GraphConstant";
    }
    throw std::runtime_error("Unknown resource type in ScenarioSpec");
}

struct ResourceInfoFactory {
    const GroupManager &_groupManager;

    BufferInfo createInfo(const BufferDesc &buffer) const {
        BufferInfo info{};
        info.debugName = buffer.guidStr;
        info.size = buffer.size;
        if (buffer.memoryGroup.has_value()) {
            info.memoryOffset = buffer.memoryGroup->offset;
        }
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
            info.samplerSettings.borderAddressMode = image.borderAddressMode.value();
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

        info.isAliased = _groupManager.isAliased(image.guid);
        info.isColorAttachment = image.colorAttachment;
        return info;
    }

    TensorInfo createInfo(const TensorDesc &tensor, bool descriptorBufferCaptureReplay) const {
        TensorInfo info;
        info.debugName = tensor.guidStr;
        if (tensor.memoryGroup.has_value()) {
            info.memoryOffset = tensor.memoryGroup->offset;
        }
        info.isAliasedWithImage = _groupManager.hasAliasOfType(tensor.guid, ResourceIdType::Image);
        info.format = getVkFormatFromString(tensor.format);
        info.shape.resize(tensor.dims.size());
        std::copy(tensor.dims.begin(), tensor.dims.end(), info.shape.begin());
        if (tensor.tiling) {
            info.tiling = tensor.tiling.value();
        }
        info.descriptorBufferCaptureReplay = descriptorBufferCaptureReplay;

        return info;
    }
};

void fill(const BaseBarrierDesc &barrier, BaseBarrierData &data) {
    data.debugName = barrier.guidStr;
    data.srcAccess = barrier.srcAccess;
    data.dstAccess = barrier.dstAccess;
    data.srcStages = barrier.srcStages;
    data.dstStages = barrier.dstStages;
}

struct BarrierDataFactory {
    const DataManager &_dataManager;

    ImageBarrierData createInfo(const ImageBarrierDesc &imageBarrier) const {
        // check the image affected by this barrier exists
        if (!_dataManager.hasImage(imageBarrier.imageResource)) {
            throw std::runtime_error("Unknown image ID for image barrier");
        }

        ImageBarrierData data{};
        fill(imageBarrier, data);
        data.oldLayout = imageBarrier.oldLayout;
        data.newLayout = imageBarrier.newLayout;
        data.image = _dataManager.getImage(imageBarrier.imageResource).image();
        data.imageRange = imageBarrier.imageRange;
        return data;
    }

    MemoryBarrierData createInfo(const MemoryBarrierDesc &memoryBarrier) const {
        MemoryBarrierData data{};
        fill(memoryBarrier, data);
        return data;
    }

    TensorBarrierData createInfo(const TensorBarrierDesc &tensorBarrier) const {
        TensorBarrierData data{};
        fill(tensorBarrier, data);
        data.tensor = _dataManager.getTensor(tensorBarrier.tensorResource).tensor();
        return data;
    }

    BufferBarrierData createInfo(const BufferBarrierDesc &bufferBarrier) const {
        BufferBarrierData data{};
        fill(bufferBarrier, data);
        data.offset = bufferBarrier.offset;
        data.size = bufferBarrier.size;
        data.buffer = _dataManager.getBuffer(bufferBarrier.bufferResource).buffer();
        return data;
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

vk::DescriptorType getResourceDescriptorType(const DataManager &dataManager, const Guid &guid) {
    if (dataManager.hasBuffer(guid)) {
        return vk::DescriptorType::eStorageBuffer;
    }
    if (dataManager.hasTensor(guid)) {
        return vk::DescriptorType::eTensorARM;
    }
    if (dataManager.hasImage(guid)) {
        if (dataManager.getImage(guid).isSampled()) {
            return vk::DescriptorType::eCombinedImageSampler;
        }
        return vk::DescriptorType::eStorageImage;
    }
    throw std::runtime_error("Invalid resource descriptor type");
}

TypedBinding convertBinding(const DataManager &dataManager, const BindingDesc &binding) {
    const auto vkType = binding.descriptorType == DescriptorType::Auto
                            ? getResourceDescriptorType(dataManager, binding.resourceRef)
                            : convertDescriptorType(binding.descriptorType);
    return {binding.set, binding.id, binding.resourceRef, binding.lod, vkType};
}

std::vector<TypedBinding> convertBindings(const DataManager &dataManager,
                                          const std::vector<BindingDesc> &bindingDescs) {
    std::vector<TypedBinding> bindings;
    bindings.reserve(bindingDescs.size());
    for (const auto &binding : bindingDescs) {
        bindings.push_back(convertBinding(dataManager, binding));
    }
    return bindings;
}

class Creator final : public IResourceCreator {
  public:
    Creator(const Context &ctx, DataManager &dataManager) : _ctx{ctx}, _dataManager{dataManager} {}

    void createBuffer(Guid guid, const BufferInfo &info) override {
        _dataManager.createBuffer(guid, info);
        auto &buffer = _dataManager.getBufferMut(guid);
        buffer.setup(_ctx);
        buffer.allocateMemory(_ctx);
    }

    void createTensor(Guid guid, const TensorInfo &info) override {
        _dataManager.createTensor(guid, info);
        auto &tensor = _dataManager.getTensorMut(guid);
        tensor.setup(_ctx);
        tensor.allocateMemory(_ctx);
    }

    void createImage(Guid guid, const ImageInfo &info) override {
        _dataManager.createImage(guid, info);
        auto &image = _dataManager.getImageMut(guid);
        image.setup(_ctx);
        image.allocateMemory(_ctx);
    }

  private:
    const Context &_ctx;
    DataManager &_dataManager;
};

struct CommandDataFactory {
    const DataManager &_dataManager;

    DispatchComputeData createData(const DispatchComputeDesc &dispatchCompute) {
        DispatchComputeData data;
        data.debugName = dispatchCompute.debugName;
        data.bindings = convertBindings(_dataManager, dispatchCompute.bindings);
        data.computeDispatch.gwcx = dispatchCompute.rangeND[0];
        data.computeDispatch.gwcy = dispatchCompute.rangeND[1];
        data.computeDispatch.gwcz = dispatchCompute.rangeND[2];
        data.shaderRef = dispatchCompute.shaderRef;
        data.implicitBarrier = dispatchCompute.implicitBarrier;
        data.pushDataRef = dispatchCompute.pushDataRef;
        return data;
    }

    DispatchFragmentData createData(const DispatchFragmentDesc &dispatchFragment) {
        DispatchFragmentData data;
        data.debugName = dispatchFragment.debugName;
        data.bindings = convertBindings(_dataManager, dispatchFragment.bindings);
        data.vertexShaderRef = dispatchFragment.vertexShaderRef;
        data.fragmentShaderRef = dispatchFragment.fragmentShaderRef;
        data.colorAttachments.reserve(dispatchFragment.colorAttachments.size());
        for (const auto &attachmentDesc : dispatchFragment.colorAttachments) {
            data.colorAttachments.push_back(
                DispatchFragmentData::Attachment{attachmentDesc.resourceRef, attachmentDesc.lod});
        }
        if (dispatchFragment.renderExtent) {
            const auto &extent = dispatchFragment.renderExtent.value();
            data.renderExtent = vk::Extent2D(extent[0], extent[1]);
        }
        data.implicitBarrier = dispatchFragment.implicitBarrier;
        data.pushDataRef = dispatchFragment.pushDataRef;
        return data;
    }

    DispatchBarrierData createData(const DispatchBarrierDesc &dispatchBarrier) {
        DispatchBarrierData data;
        for (const auto &ref : dispatchBarrier.bufferBarriersRef) {
            if (_dataManager.hasBufferBarrier(ref)) {
                data.bufferBarriers.push_back(ref);
            } else {
                throw std::runtime_error("Cannot find Buffer memory barrier");
            }
        }
        for (const auto &ref : dispatchBarrier.imageBarriersRef) {
            if (_dataManager.hasImageBarrier(ref)) {
                data.imageBarriers.push_back(ref);
            } else {
                throw std::runtime_error("Cannot find Image memory barrier");
            }
        }
        for (const auto &ref : dispatchBarrier.memoryBarriersRef) {
            if (_dataManager.hasMemoryBarrier(ref)) {
                data.memoryBarriers.push_back(ref);
            } else {
                throw std::runtime_error("Cannot find Memory barrier");
            }
        }
        for (const auto &ref : dispatchBarrier.tensorBarriersRef) {
            if (_dataManager.hasTensorBarrier(ref)) {
                data.tensorBarriers.push_back(ref);
            } else {
                throw std::runtime_error("Cannot find Tensor memory barrier");
            }
        }
        return data;
    }

    DispatchDataGraphData createData(const DispatchDataGraphDesc &dispatchDataGraph) {
        DispatchDataGraphData data;
        data.dataGraphRef = dispatchDataGraph.dataGraphRef;
        data.debugName = dispatchDataGraph.debugName;
        data.bindings = convertBindings(_dataManager, dispatchDataGraph.bindings);
        data.pushConstants = dispatchDataGraph.pushConstants;
        data.shaderSubstitutions = dispatchDataGraph.shaderSubstitutions;
        data.implicitBarrier = dispatchDataGraph.implicitBarrier;
        return data;
    }

    DispatchSpirvGraphData createData(const DispatchSpirvGraphDesc &dispatchSpirvGraph) {
        DispatchSpirvGraphData data;
        data.dataGraphRef = dispatchSpirvGraph.dataGraphRef;
        data.debugName = dispatchSpirvGraph.debugName;
        data.bindings = convertBindings(_dataManager, dispatchSpirvGraph.bindings);
        data.graphConstants = dispatchSpirvGraph.graphConstants;
        data.implicitBarrier = dispatchSpirvGraph.implicitBarrier;
        return data;
    }

    DispatchOpticalFlowData createData(const DispatchOpticalFlowDesc &dispatchOpticalFlow) {
        DispatchOpticalFlowData data;
        data.debugName = dispatchOpticalFlow.debugName;

        data.searchImage = convertBinding(_dataManager, dispatchOpticalFlow.searchImage);
        data.templateImage = convertBinding(_dataManager, dispatchOpticalFlow.templateImage);
        data.outputImage = convertBinding(_dataManager, dispatchOpticalFlow.outputImage);
        if (dispatchOpticalFlow.hintMotionVectors.has_value()) {
            data.hintMotionVectors = convertBinding(_dataManager, dispatchOpticalFlow.hintMotionVectors.value());
        }
        if (dispatchOpticalFlow.outputCost.has_value()) {
            data.outputCost = convertBinding(_dataManager, dispatchOpticalFlow.outputCost.value());
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
            if (_dataManager.hasBuffer(guid)) {
                data.buffers.emplace_back(guid);
            } else if (_dataManager.hasImage(guid)) {
                data.images.emplace_back(guid);
            } else if (_dataManager.hasTensor(guid)) {
                data.tensors.emplace_back(guid);
            } else {
                throw std::runtime_error("Unsupported resource");
            }
        }
        return data;
    }
};

ShaderInfo convert(const ShaderDesc &shaderDesc) {
    ShaderInfo info{shaderDesc.guidStr,
                    shaderDesc.entry,
                    shaderDesc.pushConstantsSize,
                    shaderDesc.specializationConstants,
                    shaderDesc.src.value_or(std::string{}),
                    shaderDesc.shaderType,
                    shaderDesc.stage,
                    shaderDesc.buildOpts,
                    shaderDesc.includeDirs};
    return info;
}

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

void verifyOpticalFlowConfig(const DataManager &dataManager, const DispatchOpticalFlowData &dispatchOpticalFlow) {
    const auto &searchImage = dataManager.getImage(dispatchOpticalFlow.searchImage.resourceRef);
    const auto &templateImage = dataManager.getImage(dispatchOpticalFlow.templateImage.resourceRef);
    const auto &outputFlowImage = dataManager.getImage(dispatchOpticalFlow.outputImage.resourceRef);

    if (searchImage.shape().size() < 3 || outputFlowImage.shape().size() < 3) {
        throw std::runtime_error("Optical flow search and output flow images must have at least 3 dimensions");
    }
    const auto &searchImageWidth = searchImage.shape()[1];
    const auto &searchImageHeight = searchImage.shape()[2];
    const auto &outputFlowImageWidth = outputFlowImage.shape()[1];
    const auto &outputFlowImageHeight = outputFlowImage.shape()[2];

    if (dispatchOpticalFlow.width != searchImageWidth || dispatchOpticalFlow.height != searchImageHeight) {
        throw std::runtime_error("Optical flow search image dimensions do not match specified input width/height");
    }
    if (searchImage.shape() != templateImage.shape()) {
        throw std::runtime_error("Optical flow search and template images must have the same dimensions");
    }
    if (searchImage.dataType() != templateImage.dataType()) {
        throw std::runtime_error("Optical flow search and template images must have the same data type");
    }
    if (dispatchOpticalFlow.outputCost.has_value()) {
        const auto &costImage = dataManager.getImage(dispatchOpticalFlow.outputCost->resourceRef);
        if (costImage.shape() != outputFlowImage.shape()) {
            throw std::runtime_error(
                "Optical flow output cost image must have the same dimensions as the output flow vector image");
        }
    }
    if (dispatchOpticalFlow.hintMotionVectors.has_value()) {
        const auto &hintMVImage = dataManager.getImage(dispatchOpticalFlow.hintMotionVectors->resourceRef);
        if (hintMVImage.shape() != outputFlowImage.shape()) {
            throw std::runtime_error(
                "Optical flow hint motion vector image must have the same dimensions as the output flow vector image");
        }
    }

    // Check ratio of input/output/grid size is correct
    if (dispatchOpticalFlow.gridSize == OpticalFlowGridSize::e1x1) {
        if (outputFlowImage.shape() != searchImage.shape()) {
            throw std::runtime_error(
                "Optical flow output flow vector image must have the same dimensions as the input for 1x1 grid size");
        }
    } else if (dispatchOpticalFlow.gridSize == OpticalFlowGridSize::e2x2) {
        if (outputFlowImage.shape().size() < 2 || outputFlowImageWidth != (searchImageWidth + 1) / 2 ||
            outputFlowImageHeight != (searchImageHeight + 1) / 2) {
            throw std::runtime_error("Optical flow output flow vector image dimensions are incompatible with input "
                                     "dimensions for 2x2 grid size");
        }
    } else if (dispatchOpticalFlow.gridSize == OpticalFlowGridSize::e4x4) {
        if (outputFlowImage.shape().size() < 2 || outputFlowImageWidth != (searchImageWidth + 3) / 4 ||
            outputFlowImageHeight != (searchImageHeight + 3) / 4) {
            throw std::runtime_error("Optical flow output flow vector image dimensions are incompatible with input "
                                     "dimensions for 4x4 grid size");
        }
    } else if (dispatchOpticalFlow.gridSize == OpticalFlowGridSize::e8x8) {
        if (outputFlowImage.shape().size() < 2 || outputFlowImageWidth != (searchImageWidth + 7) / 8 ||
            outputFlowImageHeight != (searchImageHeight + 7) / 8) {
            throw std::runtime_error("Optical flow output flow vector image dimensions are incompatible with input "
                                     "dimensions for 8x8 grid size");
        }
    } else {
        throw std::runtime_error("Unsupported optical flow grid size");
    }
}

} // namespace

Scenario::Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec)
    : _opts{opts}, _ctx{opts, getFamilyQueue(scenarioSpec)}, _scenarioSpec(scenarioSpec), _compute(_ctx) {
    setupResources();
}

void Scenario::run(int repeatCount, bool dryRun) {
    std::unique_ptr<FrameCapturer> frameCapturer;

    if (_opts.captureFrame) {
        frameCapturer = std::make_unique<FrameCapturer>();
    }

    setupCommands();

    for (int i = 0; i < repeatCount; ++i) {
        mlsdk::logging::debug("Iteration: " + std::to_string(i));
        if (i > 0) {
            _compute.reset();
        }
        if (frameCapturer) {
            frameCapturer->begin();
        }

        if (!dryRun) {
            if (hasAliasedOptimalTensors()) {
                _compute.prepareCommandBuffer();
                handleAliasedLayoutTransitions();
            }
            _compute.submitAndWaitOnFence(_perfCounters, i);
            saveProfilingData(i, repeatCount);
        }

        // Skip reset after final run
        if (i + 1 < repeatCount) {
            for (const auto &resource : _scenarioSpec.resources) {
                if (resource->resourceType == ResourceType::Image) {
                    const auto &imageDesc = static_cast<const ImageDesc &>(*resource);
                    if (imageDesc.tiling.has_value() && imageDesc.tiling.value() == Tiling::Optimal) {
                        auto &image = _dataManager.getImageMut(imageDesc.guid);
                        image.resetLayout();
                    }
                }
            }
        }

        if (frameCapturer) {
            frameCapturer->end();
        }
    }
    saveResults(dryRun);
}

void Scenario::setupResources() {
    mlsdk::logging::info("Setup resources, count: " + std::to_string(_scenarioSpec.resources.size()));
    // Handle memory groups
    for (const auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::Buffer): {
            const auto &buffer = reinterpret_cast<const std::unique_ptr<BufferDesc> &>(resource);
            if (buffer->memoryGroup.has_value()) {
                _groupManager.addResourceToGroup(buffer->memoryGroup->memoryUid, buffer->guid, ResourceIdType::Buffer);
            }
        } break;
        case (ResourceType::Image): {
            const auto &image = reinterpret_cast<const std::unique_ptr<ImageDesc> &>(resource);
            if (image->memoryGroup.has_value()) {
                _groupManager.addResourceToGroup(image->memoryGroup->memoryUid, image->guid, ResourceIdType::Image);
            }
        } break;
        case (ResourceType::Tensor): {
            const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource);
            if (tensor->memoryGroup.has_value()) {
                _groupManager.addResourceToGroup(tensor->memoryGroup->memoryUid, tensor->guid, ResourceIdType::Tensor);
            }
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
    }

    // Setup resource info
    // (Memory for Tensors and Images is allocated in next pass)
    ResourceInfoFactory resourceInfoFactory{_groupManager};
    for (const auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::Buffer): {
            const auto &buffer = reinterpret_cast<const std::unique_ptr<BufferDesc> &>(resource);
            const auto info = resourceInfoFactory.createInfo(*buffer);
            _dataManager.createBuffer(resource->guid, info);
        } break;
        case (ResourceType::RawData): {
            const auto &rawData = reinterpret_cast<const std::unique_ptr<RawDataDesc> &>(resource);
            _dataManager.createRawData(resource->guid, rawData->guidStr, rawData->src.value());
        } break;
        case (ResourceType::Image): {
            const auto &image = reinterpret_cast<const std::unique_ptr<ImageDesc> &>(resource);
            const auto info = resourceInfoFactory.createInfo(*image);
            _dataManager.createImage(resource->guid, info);
        } break;
        case (ResourceType::DataGraph): {
            const auto &dataGraph = reinterpret_cast<const std::unique_ptr<DataGraphDesc> &>(resource);
            PerfCounterGuard guard(_perfCounters, "Parse VGF: " + dataGraph->guidStr, "Scenario Setup");
            _dataManager.createVgfView(resource->guid, dataGraph->src.value());
        } break;
        case (ResourceType::Tensor): {
            const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource);
            const auto info = resourceInfoFactory.createInfo(*tensor, _opts.captureFrame);
            _dataManager.createTensor(resource->guid, info);
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }

    // Setup aliasing resources, foundation before accessing tensors
    for (const auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::Buffer): {
            auto &bufferRef = _dataManager.getBufferMut(resource->guid);
            bufferRef.setup(_ctx, _groupManager.getMemoryManager(resource->guid));
        } break;
        case (ResourceType::Image): {
            auto &imageRef = _dataManager.getImageMut(resource->guid);
            imageRef.setup(_ctx, _groupManager.getMemoryManager(resource->guid));
        } break;
        default:
            break; // No action
        }
    }

    // Setup tensors, aliasing tensors are dependent on other resources having been constructed
    for (const auto &resource : _scenarioSpec.resources) {
        if (resource->resourceType == ResourceType::Tensor) {
            auto &tensorRef = _dataManager.getTensorMut(resource->guid);
            tensorRef.setup(_ctx, _groupManager.getMemoryManager(resource->guid));
        }
    }

    // Setup barrier resource info, these depend on other resources
    BarrierDataFactory barrierDataFactory{_dataManager};
    for (const auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::ImageBarrier): {
            const auto &imageBarrier = reinterpret_cast<const std::unique_ptr<ImageBarrierDesc> &>(resource);
            const auto data = barrierDataFactory.createInfo(*imageBarrier);
            _dataManager.createImageBarrier(resource->guid, data);
        } break;
        case (ResourceType::MemoryBarrier): {
            const auto &memoryBarrier = reinterpret_cast<const std::unique_ptr<MemoryBarrierDesc> &>(resource);
            const auto data = barrierDataFactory.createInfo(*memoryBarrier);
            _dataManager.createMemoryBarrier(resource->guid, data);
        } break;
        case (ResourceType::TensorBarrier): {
            const auto &tensorBarrier = reinterpret_cast<const std::unique_ptr<TensorBarrierDesc> &>(resource);
            const auto data = barrierDataFactory.createInfo(*tensorBarrier);
            _dataManager.createTensorBarrier(resource->guid, data);
        } break;
        case (ResourceType::BufferBarrier): {
            const auto &bufferBarrier = reinterpret_cast<const std::unique_ptr<BufferBarrierDesc> &>(resource);
            const auto data = barrierDataFactory.createInfo(*bufferBarrier);
            _dataManager.createBufferBarrier(resource->guid, data);
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }

    // Allocate and fill resource memory
    for (auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::Tensor): {
            const auto &tensor = reinterpret_cast<std::unique_ptr<TensorDesc> &>(resource);
            auto &tensorRec = _dataManager.getTensorMut(tensor->guid);
            tensorRec.allocateMemory(_ctx);
            PerfCounterGuard guard(_perfCounters, "Load Tensor: " + tensor->guidStr, "Scenario Setup");
            if (tensor->src || !_groupManager.isAliased(tensor->guid)) {
                tensorRec.fillFromDescription(_ctx, *tensor);
            }
        } break;
        case (ResourceType::Image): {
            const auto &image = reinterpret_cast<std::unique_ptr<ImageDesc> &>(resource);
            auto &imageRec = _dataManager.getImageMut(image->guid);
            imageRec.allocateMemory(_ctx);
            PerfCounterGuard guard(_perfCounters, "Load Image: " + image->guidStr, "Scenario Setup");
            if (image->src || !_groupManager.isAliased(image->guid)) {
                imageRec.fillFromDescription(_ctx, *image);
            } else {
                imageRec.transitionLayout(_ctx, vk::ImageLayout::eGeneral);
            }
        } break;
        case (ResourceType::Buffer): {
            const auto &buffer = reinterpret_cast<std::unique_ptr<BufferDesc> &>(resource);
            auto &bufferRec = _dataManager.getBufferMut(buffer->guid);
            bufferRec.allocateMemory(_ctx);
            PerfCounterGuard guard(_perfCounters, "Load Buffer: " + buffer->guidStr, "Scenario Setup");
            if (buffer->src || !_groupManager.isAliased(buffer->guid)) {
                bufferRec.fillFromDescription(_ctx, *buffer);
            }
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }
}

void Scenario::setupCommands() {
    if (_opts.enablePipelineCaching) {
        mlsdk::logging::info("Load Pipeline Cache");
        PerfCounterGuard guard(_perfCounters, "Load Pipeline Cache.", "Load Pipeline Cache");
        _pipelineCache = std::make_shared<PipelineCache>(_ctx, _opts.pipelineCachePath, _opts.clearPipelineCache,
                                                         _opts.failOnPipelineCacheMiss);
    }
    // Setup commands
    mlsdk::logging::info("Setup commands");

    CommandDataFactory factory{_dataManager};
    uint32_t nQueries = 0;
    for (const auto &command : _scenarioSpec.commands) {
        switch (command->commandType) {
        case (CommandType::DispatchCompute): {
            const auto &dispatchCompute = reinterpret_cast<DispatchComputeDesc &>(*command);
            const auto data = factory.createData(dispatchCompute);
            createComputePipeline(data, nQueries);
        } break;
        case (CommandType::DispatchBarrier): {
            const auto &dispatchBarrier = reinterpret_cast<DispatchBarrierDesc &>(*command);
            const auto data = factory.createData(dispatchBarrier);
            _compute.registerPipelineBarrier(data, _dataManager);
        } break;
        case (CommandType::DispatchDataGraph): {
            const auto &dispatchDataGraph = reinterpret_cast<DispatchDataGraphDesc &>(*command);
            const auto data = factory.createData(dispatchDataGraph);
            createDataGraphPipeline(data, nQueries);
        } break;
        case (CommandType::DispatchSpirvGraph): {
            const auto &dispatchSpirvGraph = reinterpret_cast<DispatchSpirvGraphDesc &>(*command);
            const auto data = factory.createData(dispatchSpirvGraph);
            createSpirvGraphPipeline(data, nQueries);
        } break;
        case (CommandType::DispatchFragment): {
            const auto &dispatchFragment = reinterpret_cast<DispatchFragmentDesc &>(*command);
            const auto data = factory.createData(dispatchFragment);
            createFragmentPipeline(data, nQueries);
        } break;
        case (CommandType::DispatchOpticalFlow): {
            const auto &dispatchOpticalFlow = reinterpret_cast<DispatchOpticalFlowDesc &>(*command);
            const auto data = factory.createData(dispatchOpticalFlow);
            verifyOpticalFlowConfig(_dataManager, data);
            createOpticalFlowPipeline(data, nQueries);
        } break;
        case (CommandType::MarkBoundary): {
            const auto &markBoundary = reinterpret_cast<MarkBoundaryDesc &>(*command);
            const auto data = factory.createData(markBoundary);
            if (_ctx._optionals.mark_boundary) {
                _compute.registerMarkBoundary(data, _dataManager);
            } else {
                mlsdk::logging::warning("Frame boundary extension not present");
            }
        } break;
        default:
            throw std::runtime_error("Unknown CommandType in commands");
        }
    }
    // Setup profiling
    if (!_opts.profilingPath.empty() && nQueries != 0) {
        mlsdk::logging::info("Setup profiling");
        _compute.setupQueryPool(nQueries);
    }
}

bool Scenario::hasAliasedOptimalTensors() const {
    // If any tensors in any memgroup have optimal tiling
    for ([[maybe_unused]] const auto &[_, resources] : _groupManager.getGroupResources()) {
        for ([[maybe_unused]] const auto &[resource, type] : resources) {
            if (_dataManager.hasTensor(resource) &&
                _dataManager.getTensor(resource).tiling() == vk::TensorTilingARM::eOptimal) {
                return true;
            }
        }
    }
    return false;
}

void Scenario::handleAliasedLayoutTransitions() {

    // Validation pass: ensure all resources in a group have the same tiling type
    for ([[maybe_unused]] const auto &[_, resources] : _groupManager.getGroupResources()) {
        bool allLinear = true;
        bool allOptimal = true;
        for ([[maybe_unused]] const auto &[resource, type] : resources) {
            if (_dataManager.hasTensor(resource)) {
                if (_dataManager.getTensor(resource).tiling() == vk::TensorTilingARM::eLinear) {
                    allOptimal = false;
                } else {
                    allLinear = false;
                }
            } else if (_dataManager.hasImage(resource)) {
                if (_dataManager.getImage(resource).tiling() == vk::ImageTiling::eLinear) {
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

    //  Usage tracking: find tensors and images used in current dispatches
    std::unordered_set<Guid> usedResources;
    for (const auto &cmd : _scenarioSpec.commands) {
        if (cmd->commandType == CommandType::DispatchCompute) {
            const auto &compute = static_cast<const DispatchComputeDesc &>(*cmd);
            for (const auto &binding : compute.bindings) {
                usedResources.insert(binding.resourceRef);
            }
        } else if (cmd->commandType == CommandType::DispatchDataGraph) {
            const auto &graph = static_cast<const DispatchDataGraphDesc &>(*cmd);
            for (const auto &binding : graph.bindings) {
                usedResources.insert(binding.resourceRef);
            }
        }
    }

    //  Transition pass: for used tensors/images that alias each other
    for (const auto &resource : _scenarioSpec.resources) {
        if (!usedResources.count(resource->guid)) {
            continue;
        }

        //  Tensor → requires image to be in eTensorAliasingARM
        if (resource->resourceType == ResourceType::Tensor) {
            const auto &tensorDesc = static_cast<const TensorDesc &>(*resource);
            if (_groupManager.getAliasCount(tensorDesc.guid) != 2) {
                continue;
            }
            if (!tensorDesc.tiling.has_value()) {
                continue;
            }
            if (tensorDesc.tiling.value() != Tiling::Optimal) {
                continue;
            }

            for (const auto &imageResource : _scenarioSpec.resources) {
                if (imageResource->resourceType != ResourceType::Image) {
                    continue;
                }
                const auto &imageDesc = static_cast<const ImageDesc &>(*imageResource);

                if (_groupManager.getAliasCount(imageDesc.guid) != 2) {
                    continue;
                }
                if (!imageDesc.tiling.has_value()) {
                    continue;
                }

                auto &image = _dataManager.getImageMut(imageDesc.guid);
                if (image.getImageLayout() != vk::ImageLayout::eTensorAliasingARM) {
                    image.addTransitionLayoutCommand(_compute.getCommandBuffer(), vk::ImageLayout::eTensorAliasingARM);
                }
            }

            //  Image → transition back from alias layout
        } else if (resource->resourceType == ResourceType::Image) {
            const auto &imageDesc = static_cast<const ImageDesc &>(*resource);
            if (!imageDesc.tiling.has_value() || imageDesc.tiling.value() != Tiling::Optimal) {
                continue;
            }

            for (const auto &tensorResource : _scenarioSpec.resources) {
                if (tensorResource->resourceType != ResourceType::Tensor) {
                    continue;
                }
                const auto &tensorDesc = static_cast<const TensorDesc &>(*tensorResource);

                if (_groupManager.getAliasCount(tensorDesc.guid) != 2) {
                    continue;
                }
                if (!tensorDesc.tiling.has_value()) {
                    continue;
                }
                if (!usedResources.count(imageDesc.guid)) {
                    continue;
                }

                auto &image = _dataManager.getImageMut(imageDesc.guid);
                vk::ImageLayout targetLayout = vk::ImageLayout::eGeneral;
                if (imageDesc.shaderAccess == ShaderAccessType::ReadOnly) {
                    targetLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                }

                if (image.getImageLayout() != targetLayout) {
                    image.addTransitionLayoutCommand(_compute.getCommandBuffer(), targetLayout);
                }
            }
        }
    }
}

std::pair<const char *, size_t> getPushConstantData(const std::optional<Guid> &pushDataRef,
                                                    const DataManager &dataManager) {
    if (pushDataRef) {
        const auto &rawData = dataManager.getRawData(pushDataRef.value());
        return std::make_pair(rawData.data(), rawData.size());
    }
    return std::make_pair(nullptr, 0U);
}

void Scenario::createComputePipeline(const DispatchComputeData &dispatchCompute, uint32_t &nQueries) {
    // Create Compute shader pipeline
    const auto shaderInfo = convert(_scenarioSpec.getShaderResource(dispatchCompute.shaderRef));
    if (!(shaderInfo.stage == ShaderStage::Compute || shaderInfo.stage == ShaderStage::Unknown)) {
        throw std::runtime_error("DispatchCompute requires a compute shader stage");
    }
    const Compute::PipelineCreateArguments args{dispatchCompute.debugName, dispatchCompute.bindings, _pipelineCache};

    PerfCounterGuard guard(_perfCounters, "Create Pipeline: " + shaderInfo.debugName, "Pipeline Setup");
    _compute.createPipeline(args, shaderInfo);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
    const auto [pushConstantData, pushConstantSize] = getPushConstantData(dispatchCompute.pushDataRef, _dataManager);
    _compute.registerPipelineFenced(_dataManager, dispatchCompute.bindings, pushConstantData, pushConstantSize,
                                    dispatchCompute.implicitBarrier, dispatchCompute.computeDispatch);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
    mlsdk::logging::debug("Shader Pipeline: " + shaderInfo.debugName + " created");
}

void Scenario::createFragmentPipeline(const DispatchFragmentData &dispatchFragment, uint32_t &nQueries) {
    const auto vertexShaderInfo = convert(_scenarioSpec.getShaderResource(dispatchFragment.vertexShaderRef));
    const auto fragmentShaderInfo = convert(_scenarioSpec.getShaderResource(dispatchFragment.fragmentShaderRef));
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
        const auto &colorImage = _dataManager.getImage(attachmentSpec.resourceRef);
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
    const auto [pushConstantData, pushConstantSize] = getPushConstantData(dispatchFragment.pushDataRef, _dataManager);

    GraphicsDispatchInfo dispatchInfo{};
    dispatchInfo.colorAttachments = std::move(attachmentInfos);
    dispatchInfo.extent = targetExtent.value();

    _compute.registerPipelineFenced(_dataManager, dispatchFragment.bindings, pushConstantData, pushConstantSize,
                                    dispatchFragment.implicitBarrier, dispatchInfo);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    mlsdk::logging::debug("Graphics Pipeline: " + fragmentShaderInfo.debugName + " created");
}

void Scenario::createDataGraphPipeline(const DispatchDataGraphData &dispatchDataGraph, uint32_t &nQueries) {
    const VgfView &vgfView = _dataManager.getVgfView(dispatchDataGraph.dataGraphRef);
    Creator creator{_ctx, _dataManager};
    vgfView.createIntermediateResources(creator);
    for (uint32_t segmentIndex = 0; segmentIndex < vgfView.getNumSegments(); ++segmentIndex) {
        const auto &sequenceBindings = vgfView.resolveBindings(segmentIndex, _dataManager, dispatchDataGraph.bindings);
        auto moduleName = vgfView.getModuleName(segmentIndex);
        PerfCounterGuard guard(_perfCounters, "Create Pipeline: " + moduleName, "Pipeline Setup");
        createPipeline(segmentIndex, sequenceBindings, vgfView, dispatchDataGraph, nQueries);
    }
}

void Scenario::createSpirvGraphPipeline(const DispatchSpirvGraphData &dispatchSpirvGraph, uint32_t &nQueries) {
    const auto it =
        std::find_if(_scenarioSpec.resources.begin(), _scenarioSpec.resources.end(),
                     [&](const auto &resource) { return resource->guid == dispatchSpirvGraph.dataGraphRef; });
    if (it == _scenarioSpec.resources.end()) {
        throw std::runtime_error("Shader resource not found.");
    }
    const auto &shaderRes = *it;
    if (shaderRes->resourceType != ResourceType::Shader) {
        throw std::runtime_error("GUID does not reference a shader resource: " + shaderRes->guidStr);
    }
    const auto &shaderDesc = static_cast<const ShaderDesc &>(*shaderRes);

    if (shaderDesc.shaderType != ShaderType::SPIR_V) {
        throw std::runtime_error("Shader resource used to create Graph Pipeline must be of type SPIR-V");
    }

    if (!shaderDesc.src.has_value()) {
        throw std::runtime_error("Shader resource missing src: " + shaderDesc.guidStr);
    }

    const auto shaderInfo = convert(shaderDesc);

    // Validate the bindings
    const auto &sequenceBindings = dispatchSpirvGraph.bindings;
    for (const auto &binding : sequenceBindings) {
        if (_dataManager.hasTensor(binding.resourceRef)) {
            if (binding.vkDescriptorType != vk::DescriptorType::eTensorARM) {
                throw std::runtime_error("DataGraph tensor binding must use a tensor descriptor");
            }
            continue;
        }
        if (_dataManager.hasImage(binding.resourceRef)) {
            if ((binding.vkDescriptorType != vk::DescriptorType::eStorageImage) &&
                (binding.vkDescriptorType != vk::DescriptorType::eCombinedImageSampler)) {
                throw std::runtime_error("DataGraph image binding must use an image descriptor");
            }
            continue;
        }
        throw std::runtime_error("No resource with this guid found");
    }

    const auto graphConstants = collectGraphConstants(dispatchSpirvGraph.graphConstants, _scenarioSpec.resources);

    // Create pipeline and record DataGraph dispatch
    PerfCounterGuard guard(_perfCounters, "Create Pipeline: " + shaderInfo.debugName, "Pipeline Setup");
    const Compute::PipelineCreateArguments args{dispatchSpirvGraph.debugName, sequenceBindings, _pipelineCache};
    _compute.createPipeline(args, shaderInfo, _dataManager, graphConstants);
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
    const Compute::PipelineCreateArguments args{dispatchDataGraph.debugName, sequenceBindings, _pipelineCache};
    switch (vgfView.getSegmentType(segmentIndex)) {
    case ModuleType::GRAPH: {
        _compute.createPipeline(args, segmentIndex, vgfView, _dataManager);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
        _compute.registerPipelineFenced(_dataManager, sequenceBindings, nullptr, 0, dispatchDataGraph.implicitBarrier);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
        mlsdk::logging::debug("Graph Pipeline: " + vgfView.getModuleName(segmentIndex) + " created");
    } break;
    case ModuleType::SHADER: {
        bool hasSPVModule = vgfView.hasSPVModule(segmentIndex);
        bool hasGLSLModule = vgfView.hasGLSLModule(segmentIndex);
        bool hasHLSLModule = vgfView.hasHLSLModule(segmentIndex);

        if (!dispatchDataGraph.shaderSubstitutions.empty()) {
            auto moduleName = vgfView.getModuleName(segmentIndex);
            const auto shaderInfo =
                convert(_scenarioSpec.getSubstitionShader(dispatchDataGraph.shaderSubstitutions, moduleName));
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
        _compute.registerPipelineFenced(_dataManager, sequenceBindings, nullptr, 0, dispatchDataGraph.implicitBarrier,
                                        {dispatchShape[0], dispatchShape[1], dispatchShape[2]});
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
        mlsdk::logging::debug("Shader Pipeline: " + vgfView.getModuleName(segmentIndex) + " created");
    } break;
    default:
        throw std::runtime_error("Unknown module type");
    }
}

void Scenario::saveProfilingData(int iteration, int repeatCount) {
    // Save profiling data
    if (!_opts.profilingPath.empty()) {
        _compute.writeProfilingFile(_opts.profilingPath, iteration, repeatCount);
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
                const auto &guid = resourceDesc->guid;
                switch (resourceDesc->resourceType) {
                case ResourceType::Buffer:
                    _dataManager.getBuffer(guid).store(_ctx, dst.value());
                    break;
                case ResourceType::Tensor:
                    _dataManager.getTensor(guid).store(_ctx, dst.value());
                    break;
                case ResourceType::Image:
                    _dataManager.getImageMut(guid).store(_ctx, dst.value());
                    break;
                default:
                    throw std::runtime_error("Resource not found");
                }
                mlsdk::logging::debug(resourceType(resourceDesc) + " " + resourceDesc->guidStr + " output stored");
            }
        }
    }
    mlsdk::logging::info("Results stored");

    // Hexdump the session ram for debugging
    if (!_opts.sessionRAMsDumpDir.empty()) {
        _compute.sessionRAMsDump(_opts.sessionRAMsDumpDir);
    }
}

} // namespace mlsdk::scenariorunner
