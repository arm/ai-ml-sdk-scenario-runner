/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "scenario_runner/types.hpp"

#include "data_manager.hpp"

#include "mlworkloadlib/binding_set.hpp"

namespace mlsdk::scenariorunner {

/// Binds existing SR objects without allocating or transferring ownership.
/// Each feature adapter resolves its own ResourceView explicitly: VGF public
/// indices and standalone descriptor locations are different namespaces.
class WorkloadBindings {
  public:
    explicit WorkloadBindings(const DataManager &resources) : _resources(resources) {}
    ~WorkloadBindings() = default;

    /// Returns the range of the existing SR image view, including an optional
    /// single-mip selection. Use the same range in workload resource metadata.
    vk::ImageSubresourceRange imageSubresourceRange(const TypedBinding &binding) const;

    /// SR owns image transitions and layout tracking. The adapter must ensure
    /// SR establishes the workload's required layout before each recorded use.
    /// Standalone descriptions can explicitly require SR's General layout,
    /// including for sampled images. Supply memory information when required.
    void bind(workloadlib::BindingSet &bindings, workloadlib::ResourceView resource, const TypedBinding &binding,
              workloadlib::BoundMemoryInfo memory = {}) const;

  private:
    const DataManager &_resources;
};

} // namespace mlsdk::scenariorunner
