/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "workload_execution.hpp"

#include "scenario_runner/command_types.hpp"

namespace mlsdk::scenariorunner {

class DataManager;

/// Prepare a standalone compute dispatch using existing SR resources and
/// compiled shader bytes. SR retains image layouts and command synchronization.
/// Push payloads update the supplied prefix; absent data records no update.
/// The caller supplies cache and robustness session options.
std::shared_ptr<WorkloadExecution> prepareComputeExecution(std::shared_ptr<WorkloadRuntime> runtime,
                                                           const DataManager &resources, const ShaderInfo &shader,
                                                           const DispatchComputeData &dispatch,
                                                           workloadlib::SessionOptions options = {});

} // namespace mlsdk::scenariorunner
