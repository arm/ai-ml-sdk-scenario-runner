/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_builder.hpp"
#include "scenario_desc.hpp"

namespace mlsdk::scenariorunner {

class ScenarioJsonFactory {
  public:
    static std::unique_ptr<IScenario> make(const ScenarioOptions &options, const ScenarioSpec &scenarioSpec);

  private:
    static void populate(const ScenarioOptions &options, const ScenarioSpec &scenarioSpec, ScenarioBuilder &builder);
    static void resolveCommands(ScenarioBuilder &builder, const ScenarioSpec &scenarioSpec,
                                const std::unordered_map<Guid, TypedResourceId> &resourceIds);
};

} // namespace mlsdk::scenariorunner
