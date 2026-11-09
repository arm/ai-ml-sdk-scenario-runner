/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "workload_execution.hpp"
#include "vgf_view.hpp"

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

std::shared_ptr<WorkloadExecution> WorkloadExecution::fromVgf(std::shared_ptr<WorkloadRuntime> runtime,
                                                              std::shared_ptr<const VgfView> source,
                                                              workloadlib::SessionOptions options) {
    if (!source) {
        throw std::invalid_argument("VGF execution requires a VGF source");
    }
    auto execution =
        std::make_shared<WorkloadExecution>(std::move(runtime), [source] { return source->createWorkload(); }, options);
    execution->_vgfSource = std::move(source);
    return execution;
}

std::vector<workloadlib::ExecutableView> WorkloadExecution::executableTargets(std::string_view moduleName) const {
    std::vector<workloadlib::ExecutableView> targets;
    for (const auto executable : _workload.executables()) {
        if (executable.module().name() == moduleName) {
            targets.push_back(executable);
        }
    }
    if (targets.empty()) {
        throw std::runtime_error("No VGF executable found for module target: " + std::string(moduleName));
    }
    return targets;
}

void WorkloadExecution::record(vk::CommandBuffer commandBuffer, const workloadlib::RecordOptions &options) {
    if (!isPrepared()) {
        throw std::logic_error("WorkloadExecution must be prepared before recording");
    }
    _prepared->record(commandBuffer, options);
}

} // namespace mlsdk::scenariorunner
