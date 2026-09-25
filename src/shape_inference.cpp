/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shape_inference.hpp"

#include "commands.hpp"
#include "logging.hpp"
#include "resource_desc.hpp"
#include "scenario_desc.hpp"
#include "utils.hpp"
#include "vgf_view.hpp"

#include "spirv-tools/optimizer.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlsdk::scenariorunner {
namespace {

using DescriptorBinding = std::pair<uint32_t, uint32_t>;

struct GraphShapePassResult {
    std::vector<uint32_t> shapedSpirv;
    std::map<DescriptorBinding, std::vector<int64_t>> outputShapesByBinding;
};

template <typename Shape> bool isConcreteShape(const Shape &shape) {
    return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim > 0; });
}

template <typename ResourceDescType>
ResourceDescType *findResource(ScenarioSpec &scenarioSpec, const Guid &guid, ResourceType resourceType) {
    const auto resource =
        std::find_if(scenarioSpec.resources.begin(), scenarioSpec.resources.end(), [&](const auto &candidate) {
            return candidate->guid == guid && candidate->resourceType == resourceType;
        });
    return resource == scenarioSpec.resources.end() ? nullptr : static_cast<ResourceDescType *>(resource->get());
}

GraphShapePassResult runGraphShapePass(vgflib::DataView<uint32_t> unshapedSpirv,
                                       const std::map<DescriptorBinding, std::vector<int64_t>> &interfaceShapes,
                                       const std::string &debugName) {
    logging::debug("Running SPIR-V graph shape inference for " + debugName);

    std::vector<spvtools::GraphInterfaceShape> graphInterfaceShapes;
    graphInterfaceShapes.reserve(interfaceShapes.size());
    for (const auto &[descriptorBinding, shape] : interfaceShapes) {
        graphInterfaceShapes.push_back({descriptorBinding.first, descriptorBinding.second, shape});
    }

    std::map<DescriptorBinding, std::vector<int64_t>> outputShapesByBinding;
    spvtools::Optimizer optimizer(SPV_ENV_UNIVERSAL_1_6);
    optimizer.SetMessageConsumer(SPIRVMessageConsumer);
    optimizer.RegisterPass(spvtools::CreateGraphShapePass(
        graphInterfaceShapes, [&outputShapesByBinding](const std::vector<spvtools::GraphInterfaceShape> &outputShapes) {
            for (const auto &outputShape : outputShapes) {
                outputShapesByBinding[{outputShape.descriptor_set, outputShape.binding}] = outputShape.shape;
            }
        }));

    std::vector<uint32_t> shapedSpirv;
    if (!optimizer.Run(unshapedSpirv.data(), unshapedSpirv.size(), &shapedSpirv)) {
        throw std::runtime_error("SPIR-V graph shape inference failed for: " + debugName);
    }

    return {std::move(shapedSpirv), std::move(outputShapesByBinding)};
}

bool resolveVgfDispatchShapes(ScenarioSpec &scenarioSpec, const DispatchVgfDesc &dispatchVgf) {
    // Resolve and validate the referenced VGF resource.
    auto *vgf = findResource<VgfDesc>(scenarioSpec, dispatchVgf.vgfRef, ResourceType::Vgf);
    if (vgf == nullptr) {
        throw std::runtime_error("Shape inference graph_ref does not reference a VGF graph resource");
    }
    if (!vgf->src.has_value()) {
        throw std::runtime_error("Shape inference VGF graph resource is missing src: " + vgf->guidStr);
    }

    // Apply known external tensor shapes to the VGF interface.
    auto vgfView = VgfView::createVgfView(vgf->src.value());
    std::map<uint32_t, TensorDesc *> externalTensorsByMrt;

    for (const auto &binding : dispatchVgf.bindings) {
        auto *tensor = findResource<TensorDesc>(scenarioSpec, binding.resourceRef, ResourceType::Tensor);
        if (tensor == nullptr) {
            continue;
        }

        const auto mrtIndex = vgfView.getModelInterfaceMrtIndex(binding.id);
        if (!mrtIndex.has_value()) {
            continue;
        }

        externalTensorsByMrt[*mrtIndex] = tensor;

        if (isConcreteShape(tensor->dims)) {
            vgfView.setTensorShape(*mrtIndex, tensor->dims);
        }
    }

    // Resolve each SPIR-V graph segment from its concrete interface shapes.
    bool shaped = false;
    for (uint32_t segmentIndex = 0; segmentIndex < vgfView.getNumSegments(); ++segmentIndex) {
        if (vgfView.getSegmentType(segmentIndex) != ModuleType::GRAPH || !vgfView.hasSPVModule(segmentIndex)) {
            continue;
        }

        const auto mrtIndexes = vgfView.getSegmentMrtIndexes(segmentIndex);
        std::map<DescriptorBinding, std::vector<int64_t>> concreteInterfaceShapes;
        for (const auto &[descriptorBinding, mrtIndex] : mrtIndexes) {
            const auto vgfShape = vgfView.getTensorShape(mrtIndex);
            if (isConcreteShape(vgfShape)) {
                concreteInterfaceShapes[descriptorBinding] = {vgfShape.begin(), vgfShape.end()};
            }
        }

        if (concreteInterfaceShapes.empty()) {
            continue;
        }

        auto shapeResult = runGraphShapePass(vgfView.getSPVModuleCode(segmentIndex), concreteInterfaceShapes,
                                             vgf->guidStr + " segment " + std::to_string(segmentIndex));

        // Propagate inferred shapes and retain the rewritten module.
        for (const auto &[descriptorBinding, shape] : shapeResult.outputShapesByBinding) {
            const auto mrtIndex = mrtIndexes.find(descriptorBinding);
            if (mrtIndex == mrtIndexes.end()) {
                continue;
            }

            vgfView.setTensorShape(mrtIndex->second, shape);
            if (const auto tensor = externalTensorsByMrt.find(mrtIndex->second); tensor != externalTensorsByMrt.end()) {
                tensor->second->dims = shape;
            }
        }

        vgfView.setSPVModuleCode(segmentIndex, std::move(shapeResult.shapedSpirv));
        shaped = true;
    }

    if (shaped) {
        vgf->resolvedSrc = std::make_shared<const VgfView>(std::move(vgfView));
    }
    return shaped;
}

bool resolveDataGraphDispatchShapes(ScenarioSpec &scenarioSpec, const DispatchDataGraphDesc &dispatchDataGraph) {
    // Resolve and validate the referenced SPIR-V shader resource.
    auto *shaderDesc = findResource<ShaderDesc>(scenarioSpec, dispatchDataGraph.dataGraphRef, ResourceType::Shader);
    if (shaderDesc == nullptr) {
        throw std::runtime_error("Shape inference graph_ref does not reference a shader resource");
    }
    if (shaderDesc->shaderType != ShaderType::SPIR_V) {
        throw std::runtime_error("Shape inference currently requires a SPIR-V shader resource");
    }
    if (!shaderDesc->src.has_value()) {
        throw std::runtime_error("Shape inference shader resource is missing src: " + shaderDesc->guidStr);
    }

    // Collect graph bindings and their concrete tensor shapes.
    std::map<DescriptorBinding, std::vector<int64_t>> concreteInterfaceShapes;
    std::map<DescriptorBinding, TensorDesc *> tensorsByBinding;
    for (const auto &binding : dispatchDataGraph.bindings) {
        auto *tensor = findResource<TensorDesc>(scenarioSpec, binding.resourceRef, ResourceType::Tensor);
        if (tensor == nullptr) {
            continue;
        }

        const DescriptorBinding descriptorBinding{binding.set, binding.id};
        tensorsByBinding[descriptorBinding] = tensor;

        if (isConcreteShape(tensor->dims)) {
            concreteInterfaceShapes[descriptorBinding] = tensor->dims;
        }
    }

    // The shape pass needs at least one concrete interface shape.
    if (concreteInterfaceShapes.empty()) {
        return false;
    }

    // Load and rewrite the graph module.
    ShaderInfo shaderInfo;
    shaderInfo.shaderType = shaderDesc->shaderType;
    const auto unshapedSpirv = readShaderCode(shaderDesc->src.value(), shaderInfo);
    auto shapeResult =
        runGraphShapePass({unshapedSpirv->data(), unshapedSpirv->size()}, concreteInterfaceShapes, shaderDesc->guidStr);

    // Propagate inferred shapes and retain the rewritten module.
    for (const auto &[descriptorBinding, shape] : shapeResult.outputShapesByBinding) {
        if (const auto tensor = tensorsByBinding.find(descriptorBinding); tensor != tensorsByBinding.end()) {
            tensor->second->dims = shape;
        }
    }

    shaderDesc->resolvedSrc = std::make_shared<const std::vector<uint32_t>>(std::move(shapeResult.shapedSpirv));
    return true;
}

} // namespace

void resolveScenarioShapes(ScenarioSpec &scenarioSpec) {
    // Detect whether the scenario requests inference.
    bool requiresShapeInference = false;
    for (const auto &resource : scenarioSpec.resources) {
        if (resource->resourceType != ResourceType::Tensor) {
            continue;
        }

        const auto &tensor = static_cast<const TensorDesc &>(*resource);
        if (!isConcreteShape(tensor.dims)) {
            requiresShapeInference = true;
        }
    }

    // Fully concrete scenarios need no graph processing.
    if (!requiresShapeInference) {
        return;
    }

    // Resolve shapes and rewrite graph modules in dispatch order.
    bool hasResolvedGraph = false;
    for (const auto &command : scenarioSpec.commands) {
        if (command->commandType == CommandType::DispatchVgf) {
            const auto &dispatchVgf = static_cast<const DispatchVgfDesc &>(*command);
            hasResolvedGraph = resolveVgfDispatchShapes(scenarioSpec, dispatchVgf) || hasResolvedGraph;
        } else if (command->commandType == CommandType::DispatchDataGraph) {
            const auto &dispatchDataGraph = static_cast<const DispatchDataGraphDesc &>(*command);
            hasResolvedGraph = resolveDataGraphDispatchShapes(scenarioSpec, dispatchDataGraph) || hasResolvedGraph;
        }
    }

    // Inference requires at least one supported graph dispatch.
    if (!hasResolvedGraph) {
        throw std::runtime_error("Shape inference was requested, but no supported graph dispatch was found");
    }
}

} // namespace mlsdk::scenariorunner
