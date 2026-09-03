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

/// @brief Construct a scenario from a JSON description.
///
/// The factory preserves the standalone Scenario Runner JSON workflow while
/// returning the same Scenario interface used by programmatically built
/// scenarios. The returned scenario owns all parsed construction data.
class ScenarioJsonFactory {
  public:
    /// @brief Parse a JSON scenario file and return a ready-to-run scenario.
    /// @param scenarioFile Path to the JSON scenario description.
    /// @param workDir Base directory for relative input paths. The scenario
    /// file's parent directory is used when this is empty.
    /// @param outputDir Base directory for relative output paths.
    /// @param options Runtime and diagnostic options for the scenario.
    /// @return A ready-to-run scenario that owns the parsed resources and commands.
    /// @throws std::runtime_error If parsing, validation, or runtime setup fails.
    static std::unique_ptr<Scenario> make(const std::filesystem::path &scenarioFile,
                                          const std::filesystem::path &workDir = {},
                                          const std::filesystem::path &outputDir = {},
                                          const ScenarioOptions &options = {});
};

} // namespace mlsdk::scenariorunner
