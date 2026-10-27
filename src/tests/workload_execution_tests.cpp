/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_options.hpp"

#include "context.hpp"
#include "device_selection.hpp"
#include "workload_execution.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

namespace {
using namespace mlsdk::scenariorunner;
namespace wl = mlsdk::workloadlib;

// A placeholder workload needs no shader compiler or pipeline configuration.
wl::Workload makeWorkload() {
    wl::ComputeShaderDescription description;
    description.pushConstantSize = 4;
    return wl::Workload::fromComputeShader(description);
}

TEST(WorkloadExecution, PropagatesFactoryFailure) {
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    auto runtime = std::make_shared<WorkloadRuntime>(context);
    struct FactoryFailure {};
    int calls = 0;
    const auto factory = [&]() -> wl::Workload {
        ++calls;
        throw FactoryFailure{};
    };

    EXPECT_THROW((WorkloadExecution{runtime, factory}), FactoryFailure);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(runtime.use_count(), 1);
}

TEST(WorkloadExecution, RejectsMissingRuntime) {
    EXPECT_THROW((WorkloadExecution{nullptr, makeWorkload}), std::invalid_argument);
}

TEST(WorkloadExecution, ConstructsOnceAndRetainsWrappedRuntime) {
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    auto runtime = std::make_shared<WorkloadRuntime>(context);
    const auto wrapped = runtime->context().contextView();
    EXPECT_EQ(*wrapped.instance.get(), *context.instance());
    EXPECT_EQ(*wrapped.physicalDevice.get(), *context.physicalDevice());
    EXPECT_EQ(*wrapped.device.get(), *context.device());
    EXPECT_EQ(wrapped.queueFamilyIndex, context.familyQueueIdx());
    EXPECT_EQ(*wrapped.queue.get(), *context.device().getQueue(context.familyQueueIdx(), 0));

    std::weak_ptr<WorkloadRuntime> retained = runtime;
    {
        int calls = 0;
        WorkloadExecution execution{runtime, [&] {
                                        ++calls;
                                        return makeWorkload();
                                    }};
        runtime.reset();

        EXPECT_EQ(calls, 1);
        EXPECT_FALSE(retained.expired());
        EXPECT_EQ(&execution.session().workload(), &execution.workload());
        EXPECT_EQ(execution.workload().executableCount(), 1u);
        EXPECT_EQ(execution.workload().executable(0).pushConstantSize(), 4u);
        EXPECT_FALSE(execution.isPrepared());
        EXPECT_THROW(execution.record(vk::CommandBuffer{}), std::logic_error);
        EXPECT_FALSE(execution.isPrepared());
    }
    EXPECT_TRUE(retained.expired());
}

} // namespace
