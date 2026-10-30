/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "workload_runtime.hpp"

namespace mlsdk::scenariorunner {

WorkloadRuntime::WorkloadRuntime(const Context &context)
    : _queue(context.device().getQueue(context.familyQueueIdx(), 0)),
      _context(workloadlib::Context::wrap(
          {context.instance(), context.physicalDevice(), context.device(), context.familyQueueIdx(), _queue})) {}

} // namespace mlsdk::scenariorunner
