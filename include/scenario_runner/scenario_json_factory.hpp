/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario.hpp"
#include "scenario_options.hpp"

#include <filesystem>
#include <memory>

namespace mlsdk::scenariorunner {

class ScenarioJsonFactory {
  public:
    static std::unique_ptr<Scenario> make(const std::filesystem::path &scenarioFile,
                                          const std::filesystem::path &workDir = {},
                                          const std::filesystem::path &outputDir = {},
                                          const ScenarioOptions &options = {});
};

} // namespace mlsdk::scenariorunner
