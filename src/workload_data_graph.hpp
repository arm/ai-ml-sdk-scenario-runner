/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "scenario_runner/command_types.hpp"

#include "workload_execution.hpp"

#include <memory>

namespace mlsdk::scenariorunner {

class DataManager;
class ResourceManager;

/// Describe a standalone data graph shader using resolved SR tensor metadata.
/// Constant bytes are borrowed from ResourceManager and must remain valid and
/// unchanged for the WL workload lifetime.
workloadlib::DataGraphDescription describeStandaloneDataGraph(const DispatchDataGraphData &dispatch,
                                                              const ResourceManager &resources,
                                                              const DataManager &data);

/// Prepare a standalone data graph shader using existing SR tensors. Constant
/// payloads must remain valid and unchanged, and bound tensors must outlive the
/// execution and GPU completion. SR handles synchronization between commands.
std::shared_ptr<WorkloadExecution> prepareStandaloneDataGraph(std::shared_ptr<WorkloadRuntime> runtime,
                                                              const DispatchDataGraphData &dispatch,
                                                              const ResourceManager &resources, const DataManager &data,
                                                              workloadlib::SessionOptions options = {});

} // namespace mlsdk::scenariorunner
