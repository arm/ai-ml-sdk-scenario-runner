/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf_bindings.hpp"
#include "vgf_binding_map.hpp"
#include "vgf_view.hpp"

#include <stdexcept>

namespace mlsdk::scenariorunner {

VgfBindings::VgfBindings(const VgfView &source, const workloadlib::Workload &workload, const DataManager &resources,
                         const std::vector<TypedBinding> &externalBindings)
    : _workload(workload), _resources(resources) {
    if (source.getNumSegments() != workload.executableCount()) {
        throw std::invalid_argument("VGF source and workload executable counts differ");
    }
    VgfBindingMap externalResources(source.getModelInterface());
    for (const auto &binding : externalBindings) {
        if (!externalResources.hasInterfaceBinding(binding.id)) {
            continue;
        }
        // Retain SR's contextual validation; VGF-utils supplies binding identity.
        if (!source.resolveExternalBinding(resources, binding)) {
            throw std::invalid_argument("VGF source and model interface differ");
        }
        externalResources.add(binding);
    }

    for (const auto resource : workload.resources()) {
        // Bind supplied resources; workload-lib rejects missing resources used by segments.
        if (!externalResources.hasResourceBinding(resource.index())) {
            continue;
        }
        auto binding = externalResources.at(resource.index());
        // The original VGF resolver uses whole-image views for external resources.
        binding.lod = std::nullopt;
        binding.vkDescriptorType = resource.requirements().descriptorType();
        _bindings.push_back({resource, binding, source.getModelResourceAliasGroup(binding.id)});
    }
}

void VgfBindings::setResourceTilings(workloadlib::Session &session) const {
    if (&session.workload() != &_workload) {
        throw std::invalid_argument("VgfBindings::setResourceTilings: session belongs to a different workload");
    }
    for (const auto &entry : _bindings) {
        if (const auto *image = std::get_if<ImageId>(&entry.binding.resource)) {
            session.setImageTiling(entry.resource, _resources.getImage(*image).tiling());
        } else if (const auto *tensor = std::get_if<TensorId>(&entry.binding.resource)) {
            session.setTensorTiling(entry.resource, _resources.getTensor(*tensor).tiling());
        }
    }
}

std::vector<VgfBindings::AliasMemoryRequirement>
VgfBindings::aliasMemoryRequirements(const workloadlib::Session &session) const {
    if (&session.workload() != &_workload) {
        throw std::invalid_argument("VgfBindings::aliasMemoryRequirements: session belongs to a different workload");
    }
    std::vector<AliasMemoryRequirement> requirements;
    for (const auto &entry : _bindings) {
        for (uint32_t i = 0; i < entry.resource.aliasPeerCount(); ++i) {
            const auto peer = entry.resource.aliasPeer(i);
            requirements.push_back({entry.binding.resource, peer, session.aliasPeerMemoryRequirements(peer)});
        }
    }
    return requirements;
}

void VgfBindings::bind(workloadlib::BindingSet &bindings, const MemoryResolver &memory) const {
    if (&bindings.workload() != &_workload) {
        throw std::invalid_argument("VgfBindings::bind: binding set belongs to a different workload");
    }
    const WorkloadBindings adapter(_resources);
    for (const auto &entry : _bindings) {
        workloadlib::BoundMemoryInfo info;
        if (entry.resource.requirements().requiresBoundMemoryInfo()) {
            if (!memory) {
                throw std::invalid_argument("VGF alias binding requires allocation metadata");
            }
            info = memory(entry.binding.resource);
        }
        adapter.bind(bindings, entry.resource, entry.binding, info);
    }
}

} // namespace mlsdk::scenariorunner
