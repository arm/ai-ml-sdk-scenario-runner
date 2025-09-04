/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "commands.hpp"
#include "resource_desc.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mlsdk::scenariorunner {

struct ScenarioSpec {
    ScenarioSpec(std::istream *is, const std::filesystem::path &workDir, const std::filesystem::path &outputDir = {});

    /// \brief Add resource and resolve paths
    void addResource(std::unique_ptr<ResourceDesc> resource);

    void addCommand(std::unique_ptr<CommandDesc> command);
    void addCommand(std::unique_ptr<DispatchComputeDesc> command);

    bool isFirstAndLastCommand(CommandType type) const;

    uint64_t commandCount(CommandType type) const;

    /// Assumes existence of guid.
    const ShaderDesc &getShaderResource(const Guid &guid) const;

    /// Assumes existence of module
    const ShaderDesc &getSubstitionShader(const std::vector<ShaderSubstitutionDesc> &shaderSubstitutions,
                                          const std::string &moduleName) const;

    std::vector<std::unique_ptr<ResourceDesc>> resources;
    std::vector<std::unique_ptr<CommandDesc>> commands;
    // Mark scenario to have compute commands, default is dataGraph
    bool useComputeFamilyQueue{};

  private:
    std::unordered_map<Guid, uint32_t> _resourceRefs;
    std::filesystem::path _workDir;
    std::filesystem::path _outputDir;

    uint32_t shaderSubstitution(const std::vector<ShaderSubstitutionDesc> &shaderSubs,
                                const std::string &moduleName) const;
};

} // namespace mlsdk::scenariorunner
