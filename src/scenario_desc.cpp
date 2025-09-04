/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_desc.hpp"
#include "json_reader.hpp"

namespace mlsdk::scenariorunner {

ScenarioSpec::ScenarioSpec(std::istream *is, const std::filesystem::path &workDir,
                           const std::filesystem::path &outputDir)
    : _workDir(workDir), _outputDir(outputDir) {
    readJson(*this, is);
}

void ScenarioSpec::addResource(std::unique_ptr<ResourceDesc> resource) {
    if (_resourceRefs.find(resource->guid) != _resourceRefs.end()) {
        throw std::runtime_error("Not unique uid: " + resource->guidStr);
    }
    if (resource->src.has_value()) {
        auto resolvedPath = _workDir / std::filesystem::path(resource->src.value());
        resource->src = resolvedPath.string();
    }
    if (resource->dst.has_value()) {
        auto resolvedPath = _outputDir / std::filesystem::path(resource->dst.value());
        resource->dst = resolvedPath.string();
    }

    _resourceRefs[resource->guid] = static_cast<uint32_t>(resources.size());
    resources.emplace_back(std::move(resource));
}

void ScenarioSpec::addCommand(std::unique_ptr<CommandDesc> command) { commands.emplace_back(std::move(command)); }

void ScenarioSpec::addCommand(std::unique_ptr<DispatchComputeDesc> command) {
    commands.emplace_back(std::move(command));
    useComputeFamilyQueue = true;
}

bool ScenarioSpec::isFirstAndLastCommand(CommandType type) const {
    return !commands.empty() && (commands.front()->commandType == type) && (commands.back()->commandType == type);
}

uint64_t ScenarioSpec::commandCount(CommandType type) const {
    return static_cast<uint64_t>(
        std::count_if(commands.begin(), commands.end(),
                      [type](const std::unique_ptr<CommandDesc> &cmd) { return cmd->commandType == type; }));
}

const ShaderDesc &ScenarioSpec::getShaderResource(const Guid &guid) const {
    uint32_t shaderIndex = _resourceRefs.at(guid);
    const auto *ptr = dynamic_cast<const ShaderDesc *>(resources.at(shaderIndex).get());
    return *ptr;
}

const ShaderDesc &ScenarioSpec::getSubstitionShader(const std::vector<ShaderSubstitutionDesc> &shaderSubstitutions,
                                                    const std::string &moduleName) const {
    uint32_t substitutedShaderIdx = shaderSubstitution(shaderSubstitutions, moduleName);
    const auto *ptr = dynamic_cast<const ShaderDesc *>(resources.at(substitutedShaderIdx).get());
    return *ptr;
}

uint32_t ScenarioSpec::shaderSubstitution(const std::vector<ShaderSubstitutionDesc> &shaderSubs,
                                          const std::string &moduleName) const {
    for (const auto &shaderSub : shaderSubs) {
        if (shaderSub.target == moduleName) {
            return _resourceRefs.at(shaderSub.shaderRef);
        }
    }
    throw std::runtime_error("Could not perform shader substitution");
}

} // namespace mlsdk::scenariorunner
