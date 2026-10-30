/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "workload_execution.hpp"

#include <stdexcept>
#include <utility>

namespace mlsdk::scenariorunner {
namespace {

WorkloadRuntime &requireRuntime(const std::shared_ptr<WorkloadRuntime> &owner) {
    if (!owner) {
        throw std::invalid_argument("WorkloadExecution requires a runtime");
    }
    return *owner;
}

} // namespace

WorkloadExecution::WorkloadExecution(std::shared_ptr<WorkloadRuntime> runtime, const WorkloadFactory &createWorkload,
                                     workloadlib::SessionOptions options)
    : _runtime(std::move(runtime)), _workload(createWorkload()),
      _session(requireRuntime(_runtime).context(), _workload, options) {}

void WorkloadExecution::prepare(const workloadlib::BindingSet &bindings) {
    if (isPrepared()) {
        throw std::logic_error("WorkloadExecution is already prepared");
    }
    _prepared.emplace(_session.prepare(bindings));
}

void WorkloadExecution::record(vk::CommandBuffer commandBuffer) {
    if (!isPrepared()) {
        throw std::logic_error("WorkloadExecution must be prepared before recording");
    }
    _prepared->record(commandBuffer);
}

} // namespace mlsdk::scenariorunner
