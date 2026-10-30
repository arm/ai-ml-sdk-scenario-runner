/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf_binding_map.hpp"

#include "vgf-utils/model_interface.hpp"

#include <stdexcept>

namespace mlsdk::scenariorunner {

VgfBindingMap::VgfBindingMap(const vgfutils::ModelInterface &modelInterface) {
    // Preserve SR's first-match lookup, with inputs taking precedence over outputs.
    for (const auto &input : modelInterface.inputs) {
        _interfaceResources.emplace(input.bindingSlot.mBinding, input.inputOutputIndex);
    }
    for (const auto &output : modelInterface.outputs) {
        _interfaceResources.emplace(output.bindingSlot.mBinding, output.inputOutputIndex);
    }
}

bool VgfBindingMap::hasInterfaceBinding(uint32_t binding) const { return _interfaceResources.count(binding) != 0; }

bool VgfBindingMap::hasResourceBinding(uint32_t publicResourceIndex) const {
    return _externalResources.count(publicResourceIndex) != 0;
}

void VgfBindingMap::add(const TypedBinding &binding) {
    const auto target = _interfaceResources.find(binding.id);
    if (target == _interfaceResources.end()) {
        return;
    }
    const auto [entry, inserted] = _externalResources.emplace(target->second, binding);
    if (!inserted && entry->second.resource != binding.resource) {
        throw std::runtime_error("Conflicting scenario bindings for one VGF resource");
    }
}

const TypedBinding &VgfBindingMap::at(uint32_t publicResourceIndex) const {
    const auto external = _externalResources.find(publicResourceIndex);
    if (external == _externalResources.end()) {
        throw std::runtime_error("No resource found for VGF binding");
    }
    return external->second;
}

} // namespace mlsdk::scenariorunner
