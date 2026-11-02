/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "workload_execution.hpp"

#include <stdexcept>

namespace mlsdk::scenariorunner {

WorkloadRuntime &WorkloadExecution::requireRuntime(const std::shared_ptr<WorkloadRuntime> &owner) {
    if (!owner) {
        throw std::invalid_argument("WorkloadExecution requires a runtime");
    }
    return *owner;
}

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
