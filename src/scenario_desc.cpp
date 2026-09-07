/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_desc.hpp"
#include "json_reader.hpp"

#include <fstream>
#include <iostream>
#include <utility>

namespace mlsdk::scenariorunner {

ScenarioSpec::ScenarioSpec(const std::string &jsonStr, std::filesystem::path workDir, std::filesystem::path outputDir)
    : _workDir(std::move(workDir)), _outputDir(std::move(outputDir)) {
    readJson(*this, jsonStr);
}

ScenarioSpec::ScenarioSpec(const std::filesystem::path &jsonFile, std::filesystem::path workDir,
                           std::filesystem::path outputDir)
    : _workDir(std::move(workDir)), _outputDir(std::move(outputDir)) {
    std::ifstream is(jsonFile);
    if (!is) {
        throw std::runtime_error("Error while opening scenario file " + jsonFile.string());
    }

    readJson(*this, is);
}

void ScenarioSpec::addResource(std::unique_ptr<ResourceDesc> resource) {
    if (!_resourceGuids.insert(resource->guid).second) {
        throw std::runtime_error("Not unique uid: " + resource->guidStr);
    }
    if (resource->src.has_value()) {
        auto resolvedPath = _workDir / std::filesystem::path(resource->src.value());
        resource->src = resolvedPath.string();
        if (!std::filesystem::exists(resource->src.value())) {
            std::cout << "Source file does not exist: " + resource->src.value() << "\n";
        }
    }
    if (resource->resourceType == ResourceType::Shader) {
        auto &shader = static_cast<ShaderDesc &>(*resource);
        for (auto &includeDir : shader.includeDirs) {
            includeDir = (_workDir / std::filesystem::path(includeDir)).string();
        }
    }
    if (resource->dst.has_value()) {
        auto resolvedPath = _outputDir / std::filesystem::path(resource->dst.value());
        resource->dst = resolvedPath.string();
    }

    resources.emplace_back(std::move(resource));
}

void ScenarioSpec::addCommand(std::unique_ptr<CommandDesc> command) { commands.emplace_back(std::move(command)); }

} // namespace mlsdk::scenariorunner
