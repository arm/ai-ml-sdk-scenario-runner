/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario.hpp"
#include "scenario_runner/scenario_builder.hpp"
#include "scenario_runner/scenario_json_factory.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace mlsdk::scenariorunner;

TEST(PublicApi, PublicHeadersAreSelfContained) {
    static_assert(std::is_abstract_v<ScenarioBuilder>);

    auto builder = createScenarioBuilder();
    EXPECT_EQ(builder->addBuffer(BufferInfo{"buffer", 16}), BufferId{0});
}

TEST(PublicApi, ExposesScenarioJsonFactory) {
    using BuildResult = decltype(ScenarioJsonFactory::make("scenario.json"));
    static_assert(std::is_same_v<BuildResult, std::unique_ptr<Scenario>>);
}
