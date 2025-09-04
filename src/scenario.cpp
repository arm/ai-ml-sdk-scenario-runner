/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "scenario.hpp"
#include "dds_reader.hpp"
#include "frame_capturer.hpp"
#include "guid.hpp"
#include "json_writer.hpp"
#include "logging.hpp"
#include "vgf-utils/numpy.hpp"
#include <unordered_set>

namespace mlsdk::scenariorunner {

namespace // unnamed namespace
{
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
    }
    throw std::runtime_error("Unknown resource type in ScenarioSpec");
}

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

} // namespace

Scenario::Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec)
    : _opts{opts}, _ctx{opts, scenarioSpec.useComputeFamilyQueue ? FamilyQueue::Compute : FamilyQueue::DataGraph},
      _dataManager(_ctx), _scenarioSpec(scenarioSpec), _compute(_ctx) {
    setupResources();
}

void Scenario::run(int repeatCount, bool dryRun, bool captureFrame) {
    std::unique_ptr<FrameCapturer> frameCapturer;

    if (captureFrame) {
        frameCapturer = std::make_unique<FrameCapturer>();
    }

    for (int i = 0; i < repeatCount; ++i) {
        mlsdk::logging::debug("Iteration: " + std::to_string(i));
        setupCommands(i);

        if (captureFrame) {
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
            _pipelines.clear();
            _compute.reset();
            _compute.setup();

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

        if (captureFrame) {
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
                _dataManager.addResourceToGroup(buffer->memoryGroup->memoryUid, buffer->guid);
            } else {
                _dataManager.addResourceToGroup(buffer->guid, buffer->guid);
            }
        } break;
        case (ResourceType::Image): {
            const auto &image = reinterpret_cast<const std::unique_ptr<ImageDesc> &>(resource);
            if (image->memoryGroup.has_value()) {
                _dataManager.addResourceToGroup(image->memoryGroup->memoryUid, image->guid);
            } else {
                // Check not old aliasing here
                bool aliasTarget = false;
                for (const auto &resource2 : _scenarioSpec.resources) {
                    if (resource2->resourceType == ResourceType::Tensor) {
                        const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource2);
                        if (tensor->memoryGroup.has_value()) {
                            if (tensor->memoryGroup->memoryUid == image->guid) {
                                aliasTarget = true;
                                break;
                            }
                        }
                    }
                }
                if (!aliasTarget) {
                    _dataManager.addResourceToGroup(image->guid, image->guid);
                }
            }
        } break;
        case (ResourceType::Tensor): {
            const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource);
            if (tensor->memoryGroup.has_value()) {
                _dataManager.addResourceToGroup(tensor->memoryGroup->memoryUid, tensor->guid);
            } else {
                _dataManager.addResourceToGroup(tensor->guid, tensor->guid);
            }
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
    }

    // Needed for old-style memory aliasing
    for (const auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::Tensor): {
            const auto &tensor = reinterpret_cast<const std::unique_ptr<TensorDesc> &>(resource);
            if (tensor->memoryGroup.has_value()) {
                for (const auto &image : _scenarioSpec.resources) {
                    if (image->resourceType == ResourceType::Image && image->guid == tensor->memoryGroup->memoryUid) {
                        _dataManager.addResourceToGroup(tensor->memoryGroup->memoryUid, image->guid);
                    }
                }
            }
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
    }

    // Setup resource info
    // (Memory for Tensors and Images is allocated in next pass)
    for (auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::Buffer): {
            const auto &buffer = reinterpret_cast<std::unique_ptr<BufferDesc> &>(resource);
            BufferInfo info;
            info.debugName = buffer->guidStr;
            info.size = buffer->size;
            if (buffer->memoryGroup.has_value()) {
                info.memoryOffset = buffer->memoryGroup->offset;
            }
            _dataManager.createBuffer(resource->guid, info);
        } break;
        case (ResourceType::RawData): {
            const auto &rawData = reinterpret_cast<std::unique_ptr<RawDataDesc> &>(resource);
            _dataManager.createRawData(resource->guid, rawData->guidStr, rawData->src.value());
        } break;
        case (ResourceType::Image): {
            const auto &image = reinterpret_cast<std::unique_ptr<ImageDesc> &>(resource);

            ImageInfo info;
            info.debugName = image->guidStr;
            info.targetFormat = getVkFormatFromString(image->format);
            info.shape.resize(image->dims.size());
            std::copy(image->dims.begin(), image->dims.end(), info.shape.begin());
            info.mips = image->mips;
            // Image sampler settings
            if (image->minFilter) {
                info.samplerSettings.minFilter = image->minFilter.value();
            }
            if (image->magFilter) {
                info.samplerSettings.magFilter = image->magFilter.value();
            }
            if (image->mipFilter) {
                info.samplerSettings.mipFilter = image->mipFilter.value();
            }
            if (image->borderAddressMode) {
                info.samplerSettings.borderAddressMode = image->borderAddressMode.value();
            }
            if (image->borderColor) {
                info.samplerSettings.borderColor = image->borderColor.value();
            }
            if (image->customBorderColor) {
                if (info.samplerSettings.borderColor == BorderColor::FloatCustomEXT) {
                    info.samplerSettings.customBorderColor =
                        std::get<std::array<float, 4>>(image->customBorderColor.value());
                } else {
                    info.samplerSettings.customBorderColor =
                        std::get<std::array<int, 4>>(image->customBorderColor.value());
                }
            }

            if (image->src) {
                info.isInput = true;
                info.format = getVkFormatFromDDS(image->src.value());
            } else {
                info.format = info.targetFormat; // Output dds does not change type
                info.isInput = false;
            }

            if (image->tiling) {
                info.tiling = image->tiling;
            }

            switch (image->shaderAccess) {
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

            if (image->memoryGroup.has_value()) {
                info.memoryOffset = image->memoryGroup->offset;
            }

            for ([[maybe_unused]] const auto &[group, resources] : _dataManager.getResourceMemoryGroups()) {
                if (resources.find(resource->guid) != resources.end() && resources.size() != 1) {
                    info.isAliased = true;
                }
            }

            _dataManager.createImage(image->guid, info);
        } break;
        case (ResourceType::DataGraph): {
            const auto &dataGraph = reinterpret_cast<std::unique_ptr<DataGraphDesc> &>(resource);
            _perfCounters.emplace_back("Parse VGF: " + dataGraph->guidStr, "Scenario Setup", true).start();
            _dataManager.createVgfView(resource->guid, *dataGraph);
            _perfCounters.back().stop();
        } break;
        case (ResourceType::Tensor): {
            const auto &tensor = reinterpret_cast<std::unique_ptr<TensorDesc> &>(resource);

            TensorInfo info;
            info.debugName = tensor->guidStr;
            if (tensor->memoryGroup.has_value()) {
                info.memoryOffset = tensor->memoryGroup->offset;
            }
            for ([[maybe_unused]] const auto &[group, resources] : _dataManager.getResourceMemoryGroups()) {
                if (resources.find(tensor->guid) != resources.end() && resources.size() != 1) {
                    for (const auto &maybeImage : resources) {
                        if (_dataManager.hasImage(maybeImage)) {
                            info.isAliasedWithImage = true;
                        }
                    }
                }
            }
            info.format = getVkFormatFromString(tensor->format);
            info.shape.resize(tensor->dims.size());
            std::copy(tensor->dims.begin(), tensor->dims.end(), info.shape.begin());
            if (tensor->tiling) {
                info.tiling = tensor->tiling.value();
            }

            _dataManager.createTensor(resource->guid, info);
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
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
            _perfCounters.emplace_back("Load Tensor: " + tensor->guidStr, "Scenario Setup").start();
            if (tensor->src || _dataManager.isSingleMemoryGroup(tensor->guid)) {
                tensorRec.fillFromDescription(*tensor);
            }
            _perfCounters.back().stop();
        } break;
        case (ResourceType::Image): {
            const auto &image = reinterpret_cast<std::unique_ptr<ImageDesc> &>(resource);
            auto &imageRec = _dataManager.getImageMut(image->guid);
            imageRec.allocateMemory(_ctx);
            _perfCounters.emplace_back("Load Image: " + image->guidStr, "Scenario Setup").start();
            if (image->src || _dataManager.isSingleMemoryGroup(image->guid)) {
                imageRec.fillFromDescription(_ctx, *image);
            }
            _perfCounters.back().stop();
        } break;
        case (ResourceType::Buffer): {
            const auto &buffer = reinterpret_cast<std::unique_ptr<BufferDesc> &>(resource);
            auto &bufferRec = _dataManager.getBufferMut(buffer->guid);
            bufferRec.allocateMemory(_ctx);
            _perfCounters.emplace_back("Load Buffer: " + buffer->guidStr, "Scenario Setup").start();
            if (buffer->src) {
                MemoryMap mapped(buffer->src.value());
                auto dataPtr = vgfutils::numpy::parse(mapped);
                bufferRec.fill(dataPtr.ptr, dataPtr.size());
            } else if (_dataManager.isSingleMemoryGroup(buffer->guid)) {
                bufferRec.fillZero();
            }
            _perfCounters.back().stop();
        } break;
        default:
            // Skip the other types of resources
            continue;
        }
        mlsdk::logging::debug(resourceType(resource) + ": " + resource->guidStr + " loaded");
    }
}

void Scenario::setupCommands(int iteration) {
    if (_opts.enablePipelineCaching) {
        mlsdk::logging::info("Load Pipeline Cache");
        _perfCounters
            .emplace_back("Load Pipeline Cache. Iteration: " + std::to_string(iteration + 1), "Load Pipeline Cache",
                          true)
            .start();
        _pipelineCache =
            PipelineCache(_ctx, _opts.pipelineCachePath, _opts.clearPipelineCache, _opts.failOnPipelineCacheMiss);
        _perfCounters.back().stop();
    }
    // Setup commands
    mlsdk::logging::info("Setup commands");
    uint64_t numBoundaries = _scenarioSpec.commandCount(CommandType::MarkBoundary);
    // Check if first mark boundary shall be skipped
    const auto skipFirstMarkBoundary = _scenarioSpec.isFirstAndLastCommand(CommandType::MarkBoundary);
    if (skipFirstMarkBoundary) {
        numBoundaries--;
    }

    uint32_t nQueries = 0;
    for (auto &command : _scenarioSpec.commands) {
        switch (command->commandType) {
        case (CommandType::DispatchCompute): {
            const auto &dispatchCompute = reinterpret_cast<DispatchComputeDesc &>(*command);
            createComputePipeline(dispatchCompute, iteration, nQueries);
        } break;
        case (CommandType::DispatchBarrier): {
            const auto &dispatchBarrier = reinterpret_cast<DispatchBarrierDesc &>(*command);
            _compute.registerPipelineBarrier(dispatchBarrier, _dataManager);
        } break;
        case (CommandType::DispatchDataGraph): {
            const auto &dispatchDataGraph = reinterpret_cast<DispatchDataGraphDesc &>(*command);
            createDataGraphPipeline(dispatchDataGraph, iteration, nQueries);
        } break;
        case (CommandType::MarkBoundary): {
            auto &markBoundary = reinterpret_cast<MarkBoundaryDesc &>(*command);
            if (_ctx._optionals.mark_boundary) {
                if ((iteration > 0) && skipFirstMarkBoundary) {
                    continue;
                }
                markBoundary.frameId += uint64_t(iteration) * numBoundaries;
                _compute.registerMarkBoundary(markBoundary, _dataManager);
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
    for ([[maybe_unused]] const auto &[group, resources] : _dataManager.getResourceMemoryGroups()) {
        if (resources.size() <= 1)
            continue;
        for (auto resource : resources) {
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
    for (const auto &[group, resources] : _dataManager.getResourceMemoryGroups()) {
        bool allLinear = true;
        bool allOptimal = true;
        for (auto resource : resources) {
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
        if (!usedResources.count(resource->guid))
            continue;

        //  Tensor → requires image to be in eTensorAliasingARM
        if (resource->resourceType == ResourceType::Tensor) {
            const auto &tensorDesc = static_cast<const TensorDesc &>(*resource);
            auto aliasing = false;
            for ([[maybe_unused]] const auto &[group, resources] : _dataManager.getResourceMemoryGroups()) {
                if (resources.find(tensorDesc.guid) != resources.end() && resources.size() == 2) {
                    aliasing = true;
                }
            }
            if (!aliasing)
                continue;
            if (!tensorDesc.tiling.has_value())
                continue;
            if (tensorDesc.tiling.value() != Tiling::Optimal)
                continue;

            for (const auto &imageResource : _scenarioSpec.resources) {
                if (imageResource->resourceType != ResourceType::Image)
                    continue;
                const auto &imageDesc = static_cast<const ImageDesc &>(*imageResource);

                auto aliasing1 = false;
                for ([[maybe_unused]] const auto &[group, resources] : _dataManager.getResourceMemoryGroups()) {
                    if (resources.find(imageDesc.guid) != resources.end() && resources.size() == 2) {
                        aliasing1 = true;
                    }
                }
                if (!aliasing1)
                    continue;
                if (!imageDesc.tiling.has_value())
                    continue;

                auto &image = _dataManager.getImageMut(imageDesc.guid);
                if (image.getImageLayout() != vk::ImageLayout::eTensorAliasingARM) {
                    image.transitionLayout(_compute.getCommandBuffer(), vk::ImageLayout::eTensorAliasingARM);
                }
            }

            //  Image → transition back from alias layout
        } else if (resource->resourceType == ResourceType::Image) {
            const auto &imageDesc = static_cast<const ImageDesc &>(*resource);
            if (!imageDesc.tiling.has_value() || imageDesc.tiling.value() != Tiling::Optimal)
                continue;

            for (const auto &tensorResource : _scenarioSpec.resources) {
                if (tensorResource->resourceType != ResourceType::Tensor)
                    continue;
                const auto &tensorDesc = static_cast<const TensorDesc &>(*tensorResource);

                auto aliasing = false;
                for ([[maybe_unused]] const auto &[group, resources] : _dataManager.getResourceMemoryGroups()) {
                    if (resources.find(tensorDesc.guid) != resources.end() && resources.size() == 2) {
                        aliasing = true;
                    }
                }
                if (!aliasing)
                    continue;
                if (!tensorDesc.tiling.has_value())
                    continue;
                if (!usedResources.count(imageDesc.guid))
                    continue;

                auto &image = _dataManager.getImageMut(imageDesc.guid);
                vk::ImageLayout targetLayout = vk::ImageLayout::eGeneral;
                if (imageDesc.shaderAccess == ShaderAccessType::ReadOnly) {
                    targetLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                }

                if (image.getImageLayout() != targetLayout) {
                    image.transitionLayout(_compute.getCommandBuffer(), targetLayout);
                }
            }
        }
    }
}

void Scenario::createComputePipeline(const DispatchComputeDesc &dispatchCompute, int iteration, uint32_t &nQueries) {
    // Create Compute shader pipeline
    const auto &shaderDesc = _scenarioSpec.getShaderResource(dispatchCompute.shaderRef);
    _perfCounters
        .emplace_back("Create Pipeline: " + shaderDesc.guidStr + ". Iteration: " + std::to_string(iteration + 1),
                      "Pipeline Setup", true)
        .start();
    _pipelines.emplace_back(_ctx, dispatchCompute.debugName, dispatchCompute.bindings, shaderDesc, _dataManager,
                            _pipelineCache);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
    const char *pushConstantData = nullptr;
    size_t pushConstantSize = 0;
    if (dispatchCompute.pushDataRef) {
        const auto &rawData = _dataManager.getRawData(dispatchCompute.pushDataRef.value());
        pushConstantData = rawData.data();
        pushConstantSize = rawData.size();
    }
    _compute.registerPipelineFenced(_pipelines.back(), _dataManager, dispatchCompute.bindings, pushConstantData,
                                    pushConstantSize, dispatchCompute.implicitBarrier, dispatchCompute.rangeND[0],
                                    dispatchCompute.rangeND[1], dispatchCompute.rangeND[2]);
    _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
    _perfCounters.back().stop();
    mlsdk::logging::debug("Shader Pipeline: " + shaderDesc.guidStr + " created");
}

void Scenario::createDataGraphPipeline(const DispatchDataGraphDesc &dispatchDataGraph, int iteration,
                                       uint32_t &nQueries) {
    const VgfView &vgfView = _dataManager.getVgfView(dispatchDataGraph.dataGraphRef);
    vgfView.createIntermediateResources(_ctx, _dataManager);
    for (uint32_t segmentIndex = 0; segmentIndex < vgfView.getNumSegments(); ++segmentIndex) {
        std::vector<BindingDesc> sequenceBindings =
            vgfView.resolveBindings(segmentIndex, _dataManager, dispatchDataGraph.bindings);
        auto moduleName = vgfView.getSPVModuleName(segmentIndex);
        _perfCounters
            .emplace_back("Create Pipeline: " + moduleName + ". Iteration: " + std::to_string(iteration + 1),
                          "Pipeline Setup", true)
            .start();
        createPipeline(segmentIndex, sequenceBindings, vgfView, dispatchDataGraph, nQueries);
        _perfCounters.back().stop();
    }
}

void Scenario::createPipeline(const uint32_t segmentIndex, const std::vector<BindingDesc> &sequenceBindings,
                              const VgfView &vgfView, const DispatchDataGraphDesc &dispatchDataGraph,
                              uint32_t &nQueries) {
    switch (vgfView.getSegmentType(segmentIndex)) {
    case ModuleType::GRAPH: {
        _pipelines.emplace_back(_ctx, dispatchDataGraph.debugName, segmentIndex, sequenceBindings, vgfView,
                                _dataManager, _pipelineCache);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
        _compute.registerPipelineFenced(_pipelines.back(), _dataManager, sequenceBindings, nullptr, 0,
                                        dispatchDataGraph.implicitBarrier);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eDataGraphARM);
        mlsdk::logging::debug("Graph Pipeline: " + vgfView.getSPVModuleName(segmentIndex) + " created");
    } break;
    case ModuleType::SHADER: {
        bool hasSPVModule = vgfView.hasSPVModule(segmentIndex);
        if (!dispatchDataGraph.shaderSubstitutions.empty()) {
            auto moduleName = vgfView.getSPVModuleName(segmentIndex);
            const auto &shaderDesc =
                _scenarioSpec.getSubstitionShader(dispatchDataGraph.shaderSubstitutions, moduleName);
            _pipelines.emplace_back(_ctx, dispatchDataGraph.debugName, sequenceBindings, shaderDesc, _dataManager,
                                    _pipelineCache);
            if (hasSPVModule) {
                mlsdk::logging::warning("Performing shader substitution despite shader module containing code");
            }
        } else {
            if (!hasSPVModule) {
                throw std::runtime_error("No SPIR-V module present and no shader substituion defined.");
            }

            auto moduleName = vgfView.getSPVModuleName(segmentIndex);
            auto entryPoint = vgfView.getSPVModuleEntryPoint(segmentIndex);
            auto spv = vgfView.getSPVModule(segmentIndex);
            auto shaderDesc = ShaderDesc(Guid(moduleName), moduleName, {}, std::move(entryPoint), ShaderType::SPIR_V);
            _pipelines.emplace_back(_ctx, dispatchDataGraph.debugName, spv.begin(), spv.size(), sequenceBindings,
                                    shaderDesc, _dataManager, _pipelineCache);
        }

        auto dispatchShape = vgfView.getDispatchShape(segmentIndex);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
        _compute.registerPipelineFenced(_pipelines.back(), _dataManager, sequenceBindings, nullptr, 0,
                                        dispatchDataGraph.implicitBarrier, dispatchShape[0], dispatchShape[1],
                                        dispatchShape[2]);
        _compute.registerWriteTimestamp(nQueries++, vk::PipelineStageFlagBits2::eComputeShader);
        mlsdk::logging::debug("Shader Pipeline: " + vgfView.getSPVModuleName(segmentIndex) + " created");
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
    if (_pipelineCache.has_value()) {
        _perfCounters.emplace_back("Save Pipeline Cache", "Save Pipeline Cache").start();
        _pipelineCache.value().save();
        _perfCounters.back().stop();
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
    _perfCounters.emplace_back("Save Resources", "Save Results").start();
    for (auto &resourceDesc : _scenarioSpec.resources) {
        const auto &dst = resourceDesc->getDestination();
        if (dst.has_value()) {
            const auto &guid = resourceDesc->guid;
            switch (resourceDesc->resourceType) {
            case ResourceType::Buffer:
                _dataManager.getBufferMut(guid).store(_ctx, dst.value());
                break;
            case ResourceType::Tensor:
                _dataManager.getTensorMut(guid).store(_ctx, dst.value());
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
    _perfCounters.back().stop();
    mlsdk::logging::info("Results stored");

    // Hexdump the session ram for debugging
    if (!_opts.sessionRAMsDumpDir.empty()) {
        uint32_t graphPipelineIdx = 0;
        for (auto &pipeline : _pipelines) {
            auto &sessionMemory = pipeline.sessionMemory();
            auto &sessionMemoryDataSizes = pipeline.sessionMemoryDataSizes();

            for (size_t i = 0; i < sessionMemory.size(); i++) {
                const std::string neStatsFileName = "Graph_Pipeline_" + std::to_string(graphPipelineIdx++) +
                                                    "_Session_RAM_" + std::to_string(i) + ".txt";
                std::ofstream fs;
                fs.open(_opts.sessionRAMsDumpDir / neStatsFileName);

                const vk::raii::DeviceMemory &deviceMemory = sessionMemory.at(i);
                uint64_t dataSize = sessionMemoryDataSizes.at(i);

                auto *dst = reinterpret_cast<unsigned char *>(deviceMemory.mapMemory(0, vk::WholeSize));

                fs << std::hex << std::uppercase;
                fs.fill('0');

                for (size_t j = 0; j < dataSize; j++) {
                    if ((j % 16) == 0) {
                        fs << std::endl << std::setw(8) << j << ":   ";
                    }
                    fs << std::setw(2) << static_cast<unsigned>(dst[j]) << " ";
                }

                deviceMemory.unmapMemory();
                fs.close();
            }
            mlsdk::logging::info("Session RAM dump stored");
        }
    }
}

} // namespace mlsdk::scenariorunner
