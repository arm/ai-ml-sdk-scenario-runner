/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "scenario_runner/types.hpp"

#include <cstdint>
#include <map>

namespace mlsdk::vgfutils {
struct ModelInterface;
}

namespace mlsdk::scenariorunner {

/// Matches SR bindings to public resource indices using VGF interface metadata only.
/// Resource validation and workload views remain in the VGF binding adapter.
class VgfBindingMap {
  public:
    explicit VgfBindingMap(const vgfutils::ModelInterface &modelInterface);
    ~VgfBindingMap() = default;

    bool hasInterfaceBinding(uint32_t binding) const;

    bool hasResourceBinding(uint32_t publicResourceIndex) const;

    /// Ignore unknown IDs; retain the first binding and reject conflicting resources.
    void add(const TypedBinding &binding);

    /// Require a scenario binding for the public workload resource index.
    const TypedBinding &at(uint32_t publicResourceIndex) const;

  private:
    std::map<uint32_t, uint32_t> _interfaceResources;
    std::map<uint32_t, TypedBinding> _externalResources;
};

} // namespace mlsdk::scenariorunner
