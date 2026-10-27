/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "context.hpp"

#include "mlworkloadlib/context.hpp"

namespace mlsdk::scenariorunner {

/// Scenario-scoped bridge to workload-lib. Borrows the existing device and
/// queue zero from SR's selected family; does not select or create a device.
/// The SR Context must outlive this object and all dependent executions.
class WorkloadRuntime {
  public:
    explicit WorkloadRuntime(const Context &context);
    ~WorkloadRuntime() = default;

    // Keep a stable address: sessions borrow the context and its queue wrapper.
    WorkloadRuntime(const WorkloadRuntime &) = delete;
    WorkloadRuntime &operator=(const WorkloadRuntime &) = delete;
    WorkloadRuntime(WorkloadRuntime &&) = delete;
    WorkloadRuntime &operator=(WorkloadRuntime &&) = delete;

    workloadlib::Context &context() { return _context; }

  private:
    // ContextView borrows the queue wrapper as well as its Vulkan handle.
    vk::raii::Queue _queue;
    workloadlib::Context _context;
};

} // namespace mlsdk::scenariorunner
