/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "workload_bindings.hpp"

#include "mlworkloadlib/session.hpp"

#include <functional>
#include <optional>
#include <vector>

namespace mlsdk::scenariorunner {

/// Resolves one dispatch's external resources without creating SR intermediates.
/// Source metadata must describe the workload. Workload and resources are
/// borrowed and must outlive this adapter and its returned views.
class VgfBindings {
  public:
    struct ResourceBinding {
        workloadlib::ResourceView resource;
        TypedBinding binding;
        std::optional<uint32_t> aliasGroup; // VGF-local identity, not an SR memory group ID.
    };
    struct AliasMemoryRequirement {
        MemoryResourceId resource;
        workloadlib::AliasPeerView peer;
        workloadlib::AliasPeerMemoryRequirements requirements;
    };
    struct AliasPeerPlacement {
        workloadlib::AliasPeerView peer;
        vk::DeviceSize byteOffset;
    };
    using MemoryResolver = std::function<workloadlib::BoundMemoryInfo(MemoryResourceId)>;

    VgfBindings(const VgfView &source, const workloadlib::Workload &workload, const DataManager &resources,
                const std::vector<TypedBinding> &externalBindings);
    ~VgfBindings() = default;

    const std::vector<ResourceBinding> &resources() const { return _bindings; }

    /// Apply actual SR tilings before Session::configure(). Peer strides, where
    /// needed, must also be supplied by the allocation planner before configure().
    void setResourceTilings(workloadlib::Session &session) const;

    /// Query every internal alias peer after configure(), before allocating SR
    /// backing memory. The caller combines these with external Vulkan requirements.
    std::vector<AliasMemoryRequirement> aliasMemoryRequirements(const workloadlib::Session &session) const;

    /// Bind allocated objects. Alias bindings require allocation metadata from
    /// the caller; peer offsets are relative to the group's BoundMemoryInfo::offset.
    /// Omitted placements leave existing offsets unchanged (zero in a new binding
    /// set). Placements and backing regions must come from the same allocation
    /// plan. WL validates them during prepare(). Image layout transitions remain
    /// in SR's command flow.
    void bind(workloadlib::BindingSet &bindings, const MemoryResolver &memory = {},
              const std::vector<AliasPeerPlacement> &placements = {}) const;

  private:
    const workloadlib::Workload &_workload;
    const DataManager &_resources;
    std::vector<ResourceBinding> _bindings;
};

} // namespace mlsdk::scenariorunner
