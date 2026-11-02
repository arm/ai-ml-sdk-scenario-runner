/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "workload_runtime.hpp"

#include "mlworkloadlib/session.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace mlsdk::scenariorunner {

/// Owns one migrated command's workload, session and prepared binding snapshot.
/// Adapters configure session-specific inputs before configure(), then bind and
/// prepare once before registering with Compute. Recording never submits work.
/// Bound SR resources must outlive this object and GPU completion.
class WorkloadExecution {
  public:
    /// Construct the immovable workload directly in its final location.
    /// Invoke the factory directly: older Android libc++ versions reject
    /// std::function factories with an immovable return type.
    template <typename WorkloadFactory>
    WorkloadExecution(std::shared_ptr<WorkloadRuntime> runtime, WorkloadFactory &&createWorkload,
                      workloadlib::SessionOptions options = {})
        : _runtime(std::move(runtime)), _workload(std::forward<WorkloadFactory>(createWorkload)()),
          _session(requireRuntime(_runtime).context(), _workload, options) {}
    ~WorkloadExecution() = default;

    // Keep a stable address: prepared executions borrow the session.
    WorkloadExecution(const WorkloadExecution &) = delete;
    WorkloadExecution &operator=(const WorkloadExecution &) = delete;
    WorkloadExecution(WorkloadExecution &&) = delete;
    WorkloadExecution &operator=(WorkloadExecution &&) = delete;

    const workloadlib::Workload &workload() const { return _workload; }
    workloadlib::Session &session() { return _session; }
    bool isPrepared() const { return _prepared.has_value(); }

    /// Freezes the bindings for all subsequent recordings. Replacing a prepared
    /// snapshot could invalidate command buffers, so it requires a new owner.
    void prepare(const workloadlib::BindingSet &bindings);
    void record(vk::CommandBuffer commandBuffer);

  private:
    static WorkloadRuntime &requireRuntime(const std::shared_ptr<WorkloadRuntime> &owner);

    // Declaration order is intentional: borrowed objects are destroyed last.
    std::shared_ptr<WorkloadRuntime> _runtime;
    workloadlib::Workload _workload;
    workloadlib::Session _session;
    std::optional<workloadlib::PreparedExecution> _prepared;
};

} // namespace mlsdk::scenariorunner
