/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_builder_impl.hpp"
#include <stdexcept>
#include <utility>

namespace mlsdk::scenariorunner {
namespace {
template <typename Id> void requireResource(const ResourceManager &resources, Id id, const char *type) {
    try {
        static_cast<void>(resources.get(id));
    } catch (const std::out_of_range &) {
        throw std::runtime_error(std::string(type) + " resource does not exist");
    }
}

template <typename Info, typename Add>
auto addResource(detail::ScenarioBuildData &data, bool built, Info &&info, Add add) {
    if (built) {
        throw std::runtime_error("ScenarioBuilder cannot be modified after build");
    }
    return (data.resources.*add)(std::forward<Info>(info));
}
} // namespace

std::unique_ptr<ScenarioBuilder> createScenarioBuilder() { return std::make_unique<ScenarioBuilderImpl>(); }

void ScenarioBuilderImpl::ensureMutable() const {
    if (_built) {
        throw std::runtime_error("ScenarioBuilder cannot be modified after build");
    }
}

BufferId ScenarioBuilderImpl::addBuffer(const BufferInfo &info) {
    return addResource(_data, _built, info,
                       static_cast<BufferId (ResourceManager::*)(const BufferInfo &)>(&ResourceManager::addBuffer));
}
BufferId ScenarioBuilderImpl::addBuffer(BufferInfo &&info) {
    return addResource(_data, _built, std::move(info),
                       static_cast<BufferId (ResourceManager::*)(BufferInfo &&)>(&ResourceManager::addBuffer));
}
ImageId ScenarioBuilderImpl::addImage(const ImageInfo &info) {
    return addResource(_data, _built, info,
                       static_cast<ImageId (ResourceManager::*)(const ImageInfo &)>(&ResourceManager::addImage));
}
ImageId ScenarioBuilderImpl::addImage(ImageInfo &&info) {
    return addResource(_data, _built, std::move(info),
                       static_cast<ImageId (ResourceManager::*)(ImageInfo &&)>(&ResourceManager::addImage));
}
TensorId ScenarioBuilderImpl::addTensor(const TensorInfo &info) {
    return addResource(_data, _built, info,
                       static_cast<TensorId (ResourceManager::*)(const TensorInfo &)>(&ResourceManager::addTensor));
}
TensorId ScenarioBuilderImpl::addTensor(TensorInfo &&info) {
    return addResource(_data, _built, std::move(info),
                       static_cast<TensorId (ResourceManager::*)(TensorInfo &&)>(&ResourceManager::addTensor));
}
ShaderId ScenarioBuilderImpl::addShader(const ShaderInfo &info) {
    return addResource(_data, _built, info,
                       static_cast<ShaderId (ResourceManager::*)(const ShaderInfo &)>(&ResourceManager::addShader));
}
ShaderId ScenarioBuilderImpl::addShader(ShaderInfo &&info) {
    return addResource(_data, _built, std::move(info),
                       static_cast<ShaderId (ResourceManager::*)(ShaderInfo &&)>(&ResourceManager::addShader));
}
RawDataId ScenarioBuilderImpl::addRawData(const RawDataInfo &info) {
    return addResource(_data, _built, info,
                       static_cast<RawDataId (ResourceManager::*)(const RawDataInfo &)>(&ResourceManager::addRawData));
}
RawDataId ScenarioBuilderImpl::addRawData(RawDataInfo &&info) {
    return addResource(_data, _built, std::move(info),
                       static_cast<RawDataId (ResourceManager::*)(RawDataInfo &&)>(&ResourceManager::addRawData));
}
VgfId ScenarioBuilderImpl::addVgf(const VgfInfo &info) {
    return addResource(_data, _built, info,
                       static_cast<VgfId (ResourceManager::*)(const VgfInfo &)>(&ResourceManager::addVgf));
}
VgfId ScenarioBuilderImpl::addVgf(VgfInfo &&info) {
    return addResource(_data, _built, std::move(info),
                       static_cast<VgfId (ResourceManager::*)(VgfInfo &&)>(&ResourceManager::addVgf));
}
GraphConstantResourceId ScenarioBuilderImpl::addGraphConstant(const GraphConstantInfo &info) {
    return addResource(_data, _built, info,
                       static_cast<GraphConstantResourceId (ResourceManager::*)(const GraphConstantInfo &)>(
                           &ResourceManager::addGraphConstant));
}
GraphConstantResourceId ScenarioBuilderImpl::addGraphConstant(GraphConstantInfo &&info) {
    return addResource(_data, _built, std::move(info),
                       static_cast<GraphConstantResourceId (ResourceManager::*)(GraphConstantInfo &&)>(
                           &ResourceManager::addGraphConstant));
}

ImageBarrierId ScenarioBuilderImpl::addImageBarrier(const ImageBarrierInfo &info) {
    ensureMutable();
    requireResource(_data.resources, info.image, "Image");
    return _data.resources.addImageBarrier(info);
}
BufferBarrierId ScenarioBuilderImpl::addBufferBarrier(const BufferBarrierInfo &info) {
    ensureMutable();
    requireResource(_data.resources, info.buffer, "Buffer");
    return _data.resources.addBufferBarrier(info);
}
TensorBarrierId ScenarioBuilderImpl::addTensorBarrier(const TensorBarrierInfo &info) {
    ensureMutable();
    requireResource(_data.resources, info.tensor, "Tensor");
    return _data.resources.addTensorBarrier(info);
}
MemoryBarrierId ScenarioBuilderImpl::addMemoryBarrier(const MemoryBarrierInfo &info) {
    ensureMutable();
    return _data.resources.addMemoryBarrier(info);
}

MemoryGroupId ScenarioBuilderImpl::createMemoryGroup() {
    ensureMutable();
    return _data.groupManager.createMemoryGroup();
}

void ScenarioBuilderImpl::validateGroup(MemoryGroupId group) const {
    if (_data.groupManager.getGroups().find(group) == _data.groupManager.getGroups().end()) {
        throw std::runtime_error("Memory group does not exist");
    }
}

void ScenarioBuilderImpl::validateMemoryResource(MemoryResourceId resource) const {
    std::visit([&](auto id) { requireResource(_data.resources, id, "Memory"); }, resource);
}

void ScenarioBuilderImpl::addResourceToMemoryGroup(MemoryGroupId group, MemoryResourceId resource) {
    ensureMutable();
    validateGroup(group);
    validateMemoryResource(resource);
    _data.groupManager.addResourceToGroup(group, resource);
}

void ScenarioBuilderImpl::validateBinding(const TypedBinding &binding) const {
    validateMemoryResource(binding.resource);
}

void ScenarioBuilderImpl::addDispatchCompute(DispatchComputeData command) {
    ensureMutable();
    requireResource(_data.resources, command.shader, "Shader");
    for (const auto &binding : command.bindings) {
        validateBinding(binding);
    }
    if (command.pushData) {
        requireResource(_data.resources, *command.pushData, "Raw data");
    }
    _data.commands.emplace_back(std::move(command));
}

void ScenarioBuilderImpl::addDispatchFragment(DispatchFragmentData command) {
    ensureMutable();
    requireResource(_data.resources, command.vertexShader, "Shader");
    requireResource(_data.resources, command.fragmentShader, "Shader");
    for (const auto &binding : command.bindings) {
        validateBinding(binding);
    }
    for (const auto &attachment : command.colorAttachments) {
        requireResource(_data.resources, attachment.resource, "Image");
    }
    if (command.pushData) {
        requireResource(_data.resources, *command.pushData, "Raw data");
    }
    _data.commands.emplace_back(std::move(command));
}

void ScenarioBuilderImpl::addDispatchVgf(DispatchVgfData command) {
    ensureMutable();
    requireResource(_data.resources, command.vgf, "VGF");
    for (const auto &binding : command.bindings) {
        validateBinding(binding);
    }
    for (const auto &push : command.pushConstants) {
        requireResource(_data.resources, push.pushData, "Raw data");
    }
    for (const auto &shader : command.shaderSubstitutions) {
        requireResource(_data.resources, shader.shader, "Shader");
    }
    _data.commands.emplace_back(std::move(command));
}

void ScenarioBuilderImpl::addDispatchDataGraph(DispatchDataGraphData command) {
    ensureMutable();
    requireResource(_data.resources, command.graphShader, "Shader");
    for (const auto &binding : command.bindings) {
        validateBinding(binding);
    }
    for (const auto constant : command.graphConstants) {
        requireResource(_data.resources, constant, "Graph constant");
    }
    _data.commands.emplace_back(std::move(command));
}

void ScenarioBuilderImpl::addDispatchOpticalFlow(DispatchOpticalFlowData command) {
    ensureMutable();
    validateBinding(command.searchImage);
    validateBinding(command.templateImage);
    validateBinding(command.outputImage);
    if (command.hintMotionVectors) {
        validateBinding(*command.hintMotionVectors);
    }
    if (command.outputCost) {
        validateBinding(*command.outputCost);
    }
    _data.commands.emplace_back(std::move(command));
}

void ScenarioBuilderImpl::addPipelineBarrier(PipelineBarrierData command) {
    ensureMutable();
    for (const auto id : command.memoryBarriers) {
        requireResource(_data.resources, id, "Memory barrier");
    }
    for (const auto id : command.imageBarriers) {
        requireResource(_data.resources, id, "Image barrier");
    }
    for (const auto id : command.tensorBarriers) {
        requireResource(_data.resources, id, "Tensor barrier");
    }
    for (const auto id : command.bufferBarriers) {
        requireResource(_data.resources, id, "Buffer barrier");
    }
    _data.commands.emplace_back(std::move(command));
}

void ScenarioBuilderImpl::addFrameBoundary(FrameBoundaryData command) {
    ensureMutable();
    for (const auto resource : command.buffers) {
        requireResource(_data.resources, resource, "Buffer");
    }
    for (const auto resource : command.images) {
        requireResource(_data.resources, resource, "Image");
    }
    for (const auto resource : command.tensors) {
        requireResource(_data.resources, resource, "Tensor");
    }
    _data.commands.emplace_back(std::move(command));
}

std::unique_ptr<Scenario> ScenarioBuilderImpl::build(const ScenarioOptions &options) {
    return detail::createScenario(options, detail::ScenarioBuilderAccess::takeBuildData(*this));
}

detail::ScenarioBuildData detail::ScenarioBuilderAccess::takeBuildData(ScenarioBuilderImpl &builder) {
    builder.ensureMutable();
    builder._built = true;
    return std::move(builder._data);
}

} // namespace mlsdk::scenariorunner
