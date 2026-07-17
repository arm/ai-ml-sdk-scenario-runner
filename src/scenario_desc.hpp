/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "commands.hpp"
#include "resource_desc.hpp"

#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>

namespace mlsdk::scenariorunner {

struct ScenarioSpec {
    explicit ScenarioSpec(const std::string &jsonStr, const std::filesystem::path &workDir = {},
                          const std::filesystem::path &outputDir = {});
    ScenarioSpec(const std::filesystem::path &jsonFile, const std::filesystem::path &workDir,
                 const std::filesystem::path &outputDir = {});

    /// \brief Add resource and resolve paths
    void addResource(std::unique_ptr<ResourceDesc> resource);

    void addCommand(std::unique_ptr<CommandDesc> command);

    std::vector<std::unique_ptr<ResourceDesc>> resources;
    std::vector<std::unique_ptr<CommandDesc>> commands;
    // Mark scenario to have compute commands, default is dataGraph
    bool useComputeFamilyQueue{};
    bool requiresGraphicsFamilyQueue{};

  private:
    std::unordered_set<Guid> _resourceGuids;
    std::filesystem::path _workDir;
    std::filesystem::path _outputDir;
};

} // namespace mlsdk::scenariorunner
