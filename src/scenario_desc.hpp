/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "commands.hpp"
#include "resource_desc.hpp"

#include <filesystem>
#include <memory>
#include <unordered_map>
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

    /// Assumes existence of guid.
    const ShaderDesc &getShaderResource(const Guid &guid) const;

    /// Assumes existence of module
    const ShaderDesc &getSubstitionShader(const std::vector<ShaderSubstitution> &shaderSubstitutions,
                                          const std::string &moduleName) const;

    std::vector<std::unique_ptr<ResourceDesc>> resources;
    std::vector<std::unique_ptr<CommandDesc>> commands;
    // Mark scenario to have compute commands, default is dataGraph
    bool useComputeFamilyQueue{};
    bool requiresGraphicsFamilyQueue{};

  private:
    std::unordered_map<Guid, uint32_t> _resourceRefs;
    std::filesystem::path _workDir;
    std::filesystem::path _outputDir;

    uint32_t shaderSubstitution(const std::vector<ShaderSubstitution> &shaderSubs, const std::string &moduleName) const;
};

} // namespace mlsdk::scenariorunner
