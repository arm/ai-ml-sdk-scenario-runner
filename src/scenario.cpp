/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "scenario.hpp"
#include "dds_reader.hpp"
#include "frame_capturer.hpp"
#include "guid.hpp"
#include "json_reader.hpp"
#include "logging.hpp"
#include "memory_map.hpp"
#include "numpy.hpp"
#include <unordered_set>

namespace mlsdk::scenariorunner {

namespace // unnamed namespace
{
std::string resourceType(std::unique_ptr<ResourceDesc> &resource) {
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

uint32_t shaderSubstitution(const std::vector<ShaderSubstitutionDesc> &shaderSubs, const std::string &moduleName,
                            std::unordered_map<Guid, uint32_t> &_resourceRefs) {
    for (const auto &shaderSub : shaderSubs) {
        if (shaderSub.target == moduleName) {
            return _resourceRefs[shaderSub.shaderRef];
        }
    }
    throw std::runtime_error("Could not perform shader substitution");
}
} // namespace

ScenarioSpec::ScenarioSpec(std::istream *is, const std::filesystem::path &workDir,
                           const std::filesystem::path &outputDir)
    : workDir(workDir), outputDir(outputDir) {
    readJson(*this, is);
}

void ScenarioSpec::addResource(std::unique_ptr<ResourceDesc> resource) {
    this->resourceRefs[resource->guid] = static_cast<uint32_t>(this->resources.size());
    this->resources.emplace_back(std::move(resource));
}

void ScenarioSpec::addCommand(std::unique_ptr<CommandDesc> command) { this->commands.emplace_back(std::move(command)); }

Scenario::Scenario(const ScenarioOptions &opts, ScenarioSpec &scenarioSpec)
    : _opts{opts}, _ctx{opts}, _dataManager(_ctx), _scenarioSpec(scenarioSpec), _compute(_ctx) {
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
    // Setup resource info
    // (Memory for Tensors and Images is allocated in next pass)
    mlsdk::logging::info("Setup resources");
    for (auto &resource : _scenarioSpec.resources) {
        switch (resource->resourceType) {
        case (ResourceType::Buffer): {
            auto &buffer = reinterpret_cast<std::unique_ptr<BufferDesc> &>(resource);
            BufferInfo info;
            info.debugName = buffer->guidStr;
            info.size = buffer->size;
            if (buffer->src) {
                MemoryMap mapped(buffer->src.value());
                mlsdk::numpy::data_ptr dataPtr;
                mlsdk::numpy::parse(mapped, dataPtr);
                _dataManager.createBuffer(resource->guid, info, dataPtr);
            } else {
                _dataManager.createZeroedBuffer(resource->guid, info);
            }
        } break;
        case (ResourceType::RawData): {
            auto &raw_data = reinterpret_cast<std::unique_ptr<RawDataDesc> &>(resource);
            _dataManager.createRawData(resource->guid, raw_data->guidStr, raw_data->src);
        } break;
        case (ResourceType::Image): {
            ImageInfo info;
            auto &image = reinterpret_cast<std::unique_ptr<ImageDesc> &>(resource);

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

            for (auto &resource_aliased : _scenarioSpec.resources) {
                switch (resource_aliased->resourceType) {
                case (ResourceType::Tensor): {
                    auto &tensor = reinterpret_cast<std::unique_ptr<TensorDesc> &>(resource_aliased);
                    if (tensor->aliasTarget.resourceRef == resource->guid) {
                        info.isAliased = true;
                        break;
                    } else {
                        continue;
                    }
                }
                default:
                    continue;
                }
                break;
            }

            _dataManager.createImage(image->guid, info);
        } break;
        case (ResourceType::DataGraph): {
            auto &vgf = reinterpret_cast<std::unique_ptr<DataGraphDesc> &>(resource);
            _perfCounters.emplace_back("Parse VGF: " + vgf->guidStr, "Scenario Setup", true).start();
            _dataManager.createVgfView(resource->guid, *vgf.get());
            _perfCounters.back().stop();
        } break;
        case (ResourceType::ImageBarrier): {
            auto &imageBarrier = reinterpret_cast<std::unique_ptr<ImageBarrierDesc> &>(resource);

            // check the image affected by this barrier exists
            if (!_dataManager.hasImage(imageBarrier->imageResource)) {
                throw std::runtime_error("Unknown image ID for image barrier");
            }

            ImageBarrierData data;
            data.debugName = imageBarrier->guidStr;
            data.srcAccess = imageBarrier->srcAccess;
            data.dstAccess = imageBarrier->dstAccess;
            data.srcStages = imageBarrier->srcStages;
            data.dstStages = imageBarrier->dstStages;
            data.oldLayout = imageBarrier->oldLayout;
            data.newLayout = imageBarrier->newLayout;
            data.image = _dataManager.getImage(imageBarrier->imageResource).image();
            data.imageRange = imageBarrier->imageRange;
            _dataManager.createImageBarrier(resource->guid, data);
        } break;
        case (ResourceType::MemoryBarrier): {
            auto &memoryBarrier = reinterpret_cast<std::unique_ptr<MemoryBarrierDesc> &>(resource);

            MemoryBarrierData data;
            data.debugName = memoryBarrier->guidStr;
            data.srcAccess = memoryBarrier->srcAccess;
            data.dstAccess = memoryBarrier->dstAccess;
            data.srcStages = memoryBarrier->srcStages;
            data.dstStages = memoryBarrier->dstStages;
            _dataManager.createMemoryBarrier(resource->guid, data);
        } break;
        case (ResourceType::TensorBarrier): {
            auto &tensorBarrier = reinterpret_cast<std::unique_ptr<TensorBarrierDesc> &>(resource);

            TensorBarrierData data;
            data.debugName = tensorBarrier->guidStr;
            data.srcAccess = tensorBarrier->srcAccess;
            data.dstAccess = tensorBarrier->dstAccess;
            data.srcStages = tensorBarrier->srcStages;
            data.dstStages = tensorBarrier->dstStages;

            data.tensor = _dataManager.getTensor(tensorBarrier->tensorResource).tensor();
            _dataManager.createTensorBarrier(resource->guid, data);
        } break;
        case (ResourceType::BufferBarrier): {
            auto &bufferBarrier = reinterpret_cast<std::unique_ptr<BufferBarrierDesc> &>(resource);

            BufferBarrierData data;
            data.debugName = bufferBarrier->guidStr;
            data.srcAccess = bufferBarrier->srcAccess;
            data.dstAccess = bufferBarrier->dstAccess;
            data.srcStages = bufferBarrier->srcStages;
            data.dstStages = bufferBarrier->dstStages;
            data.offset = bufferBarrier->offset;
            data.size = bufferBarrier->size;
            data.buffer = _dataManager.getBuffer(bufferBarrier->bufferResource).buffer();
            _dataManager.createBufferBarrier(resource->guid, data);
        } break;
        case (ResourceType::Tensor): {
            auto &tensor = reinterpret_cast<std::unique_ptr<TensorDesc> &>(resource);

            TensorInfo info;
            info.debugName = tensor->guidStr;
            info.isAliased = tensor->aliasTarget.resourceRef.isValid();
            info.format = getVkFormatFromString(tensor->format);
            info.shape.resize(tensor->dims.size());
            std::copy(tensor->dims.begin(), tensor->dims.end(), info.shape.begin());
            if (tensor->tiling) {
                info.tiling = tensor->tiling.value();
            }

            if (_dataManager.getMemoryManager(tensor->aliasTarget.resourceRef) == nullptr) {
                _dataManager.createTensor(resource->guid, info);
                continue;
            }
            if (tensor->src) {
                throw std::runtime_error("Tensor cannot have src file and alias other resource");
            }
            _dataManager.createTensor(resource->guid, info, tensor->aliasTarget.resourceRef);
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
            auto &tensorDesc = reinterpret_cast<std::unique_ptr<TensorDesc> &>(resource);
            auto &tensor = _dataManager.getTensorMut(tensorDesc->guid);
            tensor.allocateMemory(_ctx);
            if (!tensorDesc->aliasTarget.resourceRef.isValid()) {
                _perfCounters.emplace_back("Load Tensor: " + tensorDesc->guidStr, "Scenario Setup").start();
                tensor.fillFromDescription(*tensorDesc);
                _perfCounters.back().stop();
            }
        } break;
        case (ResourceType::Image): {
            auto &image = reinterpret_cast<std::unique_ptr<ImageDesc> &>(resource);
            auto &imageRec = _dataManager.getImageMut(image->guid);
            imageRec.allocateMemory(_ctx);
            _perfCounters.emplace_back("Load Image: " + image->guidStr, "Scenario Setup").start();
            imageRec.fillFromDescription(_ctx, *image);
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
    uint32_t nQueries = 0;
    for (auto &command : _scenarioSpec.commands) {
        switch (command->commandType) {
        case (CommandType::DispatchCompute): {
            auto &dispatchCompute = reinterpret_cast<DispatchComputeDesc &>(*command);

            // Create Compute shader pipeline
            uint32_t shaderIndex = _scenarioSpec.resourceRefs[dispatchCompute.shaderRef];
            auto &shaderDesc = reinterpret_cast<std::unique_ptr<ShaderDesc> &>(_scenarioSpec.resources[shaderIndex]);
            _perfCounters
                .emplace_back("Create Pipeline: " + shaderDesc->guidStr +
                                  ". Iteration: " + std::to_string(iteration + 1),
                              "Pipeline Setup", true)
                .start();
            // Read shader file
            _pipelines.emplace_back(_ctx, dispatchCompute.debugName, dispatchCompute.bindings, *shaderDesc,
                                    &_dataManager, _pipelineCache);
            _compute.registerWriteTimestamp(nQueries++);
            if (dispatchCompute.pushDataRef) {
                const RawData &pushConstantData = _dataManager.getRawData(dispatchCompute.pushDataRef.value());
                _compute.registerPipelineFenced(_pipelines.back(), &_dataManager, dispatchCompute.bindings,
                                                pushConstantData.data(), pushConstantData.size(),
                                                dispatchCompute.implicitBarrier, dispatchCompute.rangeND[0],
                                                dispatchCompute.rangeND[1], dispatchCompute.rangeND[2]);
            } else {
                _compute.registerPipelineFenced(_pipelines.back(), &_dataManager, dispatchCompute.bindings, nullptr, 0,
                                                dispatchCompute.implicitBarrier, dispatchCompute.rangeND[0],
                                                dispatchCompute.rangeND[1], dispatchCompute.rangeND[2]);
            }
            _compute.registerWriteTimestamp(nQueries++);
            _perfCounters.back().stop();
            mlsdk::logging::debug("Shader Pipeline: " + shaderDesc->guidStr + " created");
        } break;
        case (CommandType::DispatchBarrier): {
            auto &dispatchBarrier = reinterpret_cast<DispatchBarrierDesc &>(*command);
            _compute.registerPipelineBarrier(dispatchBarrier, &_dataManager);
        } break;
        case (CommandType::DispatchDataGraph): {
            DispatchDataGraphDesc &dispatchDataGraph = reinterpret_cast<DispatchDataGraphDesc &>(*command);
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
                createPipeline(segmentIndex, sequenceBindings, vgfView, dispatchDataGraph, _pipelineCache, nQueries);
                _perfCounters.back().stop();
            }
        } break;
        case (CommandType::MarkBoundary): {
            MarkBoundaryDesc &markBoundary = reinterpret_cast<MarkBoundaryDesc &>(*command);
            if (_ctx._optionals.mark_boundary == true) {
                _compute.registerMarkBoundary(markBoundary, &_dataManager);
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
    for (const auto &res : _scenarioSpec.resources) {
        if (res->resourceType == ResourceType::Tensor) {
            const auto &tensorDesc = static_cast<const TensorDesc &>(*res);
            if (tensorDesc.aliasTarget.resourceRef.isValid() && tensorDesc.tiling.has_value() &&
                tensorDesc.tiling.value() == Tiling::Optimal)
                return {true};
        }
    }
    return false;
}

void Scenario::handleAliasedLayoutTransitions() {
    auto &cmdBuf = _compute.getCommandBuffer();

    // Validation pass: ensure all aliased tensor-image pairs have same tiling
    for (const auto &resource : _scenarioSpec.resources) {
        if (resource->resourceType != ResourceType::Tensor)
            continue;

        const auto &tensorDesc = static_cast<const TensorDesc &>(*resource);
        if (!tensorDesc.aliasTarget.resourceRef.isValid() || !tensorDesc.tiling.has_value())
            continue;

        for (const auto &imageResource : _scenarioSpec.resources) {
            if (imageResource->resourceType != ResourceType::Image)
                continue;
            const auto &imageDesc = static_cast<const ImageDesc &>(*imageResource);

            if (imageDesc.guid != tensorDesc.aliasTarget.resourceRef || !imageDesc.tiling.has_value())
                continue;

            if (tensorDesc.tiling.value() != imageDesc.tiling.value()) {
                throw std::runtime_error("Aliased resources must have identical tiling.");
            }
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
            if (!tensorDesc.aliasTarget.resourceRef.isValid() || !tensorDesc.tiling.has_value())
                continue;
            if (tensorDesc.tiling.value() != Tiling::Optimal)
                continue;

            for (const auto &imageResource : _scenarioSpec.resources) {
                if (imageResource->resourceType != ResourceType::Image)
                    continue;
                const auto &imageDesc = static_cast<const ImageDesc &>(*imageResource);

                if (imageDesc.guid != tensorDesc.aliasTarget.resourceRef || !imageDesc.tiling.has_value())
                    continue;

                auto &image = _dataManager.getImageMut(imageDesc.guid);
                if (image.getImageLayout() != vk::ImageLayout::eTensorAliasingARM) {
                    image.transitionLayout(cmdBuf, vk::ImageLayout::eTensorAliasingARM);
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

                if (tensorDesc.aliasTarget.resourceRef != imageDesc.guid || !tensorDesc.tiling.has_value())
                    continue;
                if (!usedResources.count(imageDesc.guid))
                    continue;

                auto &image = _dataManager.getImageMut(imageDesc.guid);
                vk::ImageLayout targetLayout = vk::ImageLayout::eGeneral;
                if (imageDesc.shaderAccess == ShaderAccessType::ReadOnly) {
                    targetLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                }

                if (image.getImageLayout() != targetLayout) {
                    image.transitionLayout(cmdBuf, targetLayout);
                }
            }
        }
    }
}

void Scenario::createPipeline(const uint32_t segmentIndex, std::vector<BindingDesc> &sequenceBindings,
                              const VgfView &vgfView, DispatchDataGraphDesc &dispatchDataGraph,
                              std::optional<PipelineCache> &pipelineCache, uint32_t &nQueries) {
    switch (vgfView.getSegmentType(segmentIndex)) {
    case ModuleType::GRAPH: {
        _pipelines.emplace_back(_ctx, dispatchDataGraph.debugName, segmentIndex, sequenceBindings, vgfView,
                                &_dataManager, pipelineCache);
        _compute.registerWriteTimestamp(nQueries++);
        _compute.registerPipelineFenced(_pipelines.back(), &_dataManager, sequenceBindings, nullptr, 0,
                                        dispatchDataGraph.implicitBarrier);
        _compute.registerWriteTimestamp(nQueries++);
        mlsdk::logging::debug("Graph Pipeline: " + vgfView.getSPVModuleName(segmentIndex) + " created");
    } break;
    case ModuleType::SHADER: {
        bool hasSPVModule = vgfView.hasSPVModule(segmentIndex);
        if (!dispatchDataGraph.shaderSubstitutions.empty()) {
            auto moduleName = vgfView.getSPVModuleName(segmentIndex);
            uint32_t substitutedShaderIdx =
                shaderSubstitution(dispatchDataGraph.shaderSubstitutions, moduleName, _scenarioSpec.resourceRefs);
            auto &shaderDesc =
                reinterpret_cast<std::unique_ptr<ShaderDesc> &>(_scenarioSpec.resources[substitutedShaderIdx]);
            _pipelines.emplace_back(_ctx, dispatchDataGraph.debugName, sequenceBindings, *shaderDesc, &_dataManager,
                                    pipelineCache);
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
                                    shaderDesc, &_dataManager, pipelineCache);
        }

        auto dispatchShape = vgfView.getDispatchShape(segmentIndex);
        _compute.registerWriteTimestamp(nQueries++);
        _compute.registerPipelineFenced(_pipelines.back(), &_dataManager, sequenceBindings, nullptr, 0,
                                        dispatchDataGraph.implicitBarrier, dispatchShape[0], dispatchShape[1],
                                        dispatchShape[2]);
        _compute.registerWriteTimestamp(nQueries++);
        mlsdk::logging::debug("Shader Pipeline: " + vgfView.getSPVModuleName(segmentIndex) + " created");
    } break;
    default:
        throw std::runtime_error("Unknown module type");
    }
}

void Scenario::saveProfilingData(int iteration, int repeatCount) {
    // Save profiling data
    if (!_opts.profilingPath.empty()) {
        std::vector<uint64_t> timestamps = _compute.queryTimestamps();
        std::vector<std::string> profiledCommands;
        VkPhysicalDeviceLimits physicalDeviceLimits = _ctx.physicalDevice().getProperties().limits;
        float timestampPeriod = physicalDeviceLimits.timestampPeriod;
        for (const auto &command : _compute.getCommands()) {
            if (std::holds_alternative<ComputeDispatch>(command)) {
                profiledCommands.push_back("ComputeDispatch");
            } else if (std::holds_alternative<DataGraphDispatch>(command)) {
                profiledCommands.push_back("DataGraphDispatch");
            }
        }
        writeProfilingData(timestamps, timestampPeriod, profiledCommands, _opts.profilingPath, iteration, repeatCount);
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
        // Save performance conunters
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
        if (resourceDesc->getDestination().has_value()) {
            _dataManager.storeResource(*resourceDesc);
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
