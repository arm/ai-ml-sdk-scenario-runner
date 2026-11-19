/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "workload_data_graph.hpp"

#include "data_manager.hpp"
#include "resource_manager.hpp"
#include "workload_bindings.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace mlsdk::scenariorunner {
namespace wl = workloadlib;
namespace {

wl::DataGraphResource describeTensorBinding(const TypedBinding &binding, const DataManager &data) {
    const auto *id = std::get_if<TensorId>(&binding.resource);
    if (!id || binding.vkDescriptorType != vk::DescriptorType::eTensorARM) {
        throw std::runtime_error("Standalone data graph shaders require tensor bindings");
    }
    const auto &tensor = data.getTensor(*id);
    wl::DataGraphResource result;
    result.set = binding.set;
    result.binding = binding.id;
    result.name = tensor.debugName();
    result.resource.descriptorType = binding.vkDescriptorType;
    result.resource.kind = wl::ResourceKind::Tensor;
    result.resource.format = tensor.dataType();
    result.resource.tensor.usage = vk::TensorUsageFlagBitsARM::eDataGraph;
    result.resource.tensor.shape = tensor.shape();
    result.resource.tensor.stride = tensor.dimStrides();
    return result;
}

} // namespace

wl::DataGraphDescription describeStandaloneDataGraph(const DispatchDataGraphData &dispatch,
                                                     const ResourceManager &resources, const DataManager &data) {
    const auto &shader = resources.get(dispatch.graphShader);
    if (shader.shaderType != ShaderType::SPIR_V) {
        throw std::runtime_error("Shader resource used to create Graph Pipeline must be of type SPIR-V");
    }
    if (!shader.src || shader.src->empty()) {
        throw std::runtime_error("Shader resource missing src: " + shader.debugName);
    }

    wl::DataGraphDescription description;
    description.module.codeKind = wl::ModuleCodeKind::Spirv;
    description.module.spirv = *shader.src;
    description.entryPoint = shader.entry;
    description.implicitBarrier = dispatch.implicitBarrier;
    for (const auto &binding : dispatch.bindings) {
        description.resources.push_back(describeTensorBinding(binding, data));
    }
    for (const auto id : dispatch.graphConstants) {
        // Borrow scenario-owned bytes, never a temporary GraphConstantInfo copy.
        const auto &constant = resources.get(id);
        wl::DataGraphConstant result;
        result.name = constant.debugName;
        result.resource.kind = wl::ResourceKind::Tensor;
        result.resource.format = constant.format;
        result.resource.tensor.usage = vk::TensorUsageFlagBitsARM::eDataGraph;
        result.resource.tensor.shape = constant.dims;
        result.data = constant.data.data();
        result.size = constant.data.size();
        description.constants.push_back(std::move(result));
    }

    return description;
}

std::shared_ptr<WorkloadExecution> prepareStandaloneDataGraph(std::shared_ptr<WorkloadRuntime> runtime,
                                                              const DispatchDataGraphData &dispatch,
                                                              const ResourceManager &resources, const DataManager &data,
                                                              wl::SessionOptions options) {
    auto description = describeStandaloneDataGraph(dispatch, resources, data);
    const WorkloadBindings binder(data);
    auto execution = std::make_shared<WorkloadExecution>(
        std::move(runtime), [&] { return wl::Workload::fromDataGraph(std::move(description)); }, options);
    const auto &workload = execution->workload();
    const auto executable = workload.executable(0);
    std::vector<std::pair<wl::ResourceView, const TypedBinding *>> mapping;
    for (uint32_t i = 0; i < executable.interfaceDescriptorBindingCount(); ++i) {
        const auto descriptor = executable.interfaceDescriptorBinding(i);
        const auto binding = std::find_if(dispatch.bindings.begin(), dispatch.bindings.end(), [&](const auto &entry) {
            return entry.set == descriptor.set && entry.id == descriptor.binding;
        });
        if (binding == dispatch.bindings.end()) {
            throw std::logic_error("Workload descriptor has no scenario binding");
        }
        const auto resource = workload.resource(descriptor.resourceIndex);
        mapping.emplace_back(resource, &*binding);
        execution->session().setTensorTiling(resource, data.getTensor(std::get<TensorId>(binding->resource)).tiling());
    }
    execution->session().configure();
    auto bindings = execution->session().createBindingSet();
    for (const auto &[resource, binding] : mapping) {
        binder.bind(bindings, resource, *binding);
    }
    execution->prepare(bindings);
    return execution;
}

} // namespace mlsdk::scenariorunner
