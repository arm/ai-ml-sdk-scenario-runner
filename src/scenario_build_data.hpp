/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_runner/command_types.hpp"
#include "scenario_runner/resource_data.hpp"

#include "group_manager.hpp"
#include "guid.hpp"
#include "resource_manager.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace mlsdk::scenariorunner {

class Scenario;
struct ScenarioOptions;

using TypedResourceId = std::variant<BufferId, ImageId, TensorId, ShaderId, RawDataId, VgfId, GraphConstantResourceId,
                                     ImageBarrierId, BufferBarrierId, TensorBarrierId, MemoryBarrierId>;

namespace detail {

using ScenarioCommand = std::variant<DispatchComputeData, DispatchFragmentData, DispatchVgfData, DispatchDataGraphData,
                                     DispatchOpticalFlowData, PipelineBarrierData, FrameBoundaryData>;

struct InitializationBase {
    explicit InitializationBase(std::string debugName) : debugName{std::move(debugName)} {}
    std::string debugName;
};

struct BufferInitialization : InitializationBase {
    BufferInitialization(BufferId id, BufferData data, std::string debugName)
        : InitializationBase{std::move(debugName)}, id{id}, data{std::move(data)} {}
    BufferId id;
    BufferData data;
};

struct ImageInitialization : InitializationBase {
    ImageInitialization(ImageId id, std::optional<ImageData> data, std::string debugName)
        : InitializationBase{std::move(debugName)}, id{id}, data{std::move(data)} {}
    ImageId id;
    std::optional<ImageData> data;
};

struct TensorInitialization : InitializationBase {
    TensorInitialization(TensorId id, TensorData data, std::string debugName)
        : InitializationBase{std::move(debugName)}, id{id}, data{std::move(data)} {}
    TensorId id;
    TensorData data;
};

using ResourceInitialization = std::variant<BufferInitialization, ImageInitialization, TensorInitialization>;

struct ResourceOutput {
    TypedResourceId id;
    std::string destination;
    std::string debugName;
};

struct ScenarioBuildData {
    ResourceManager resources;
    GroupManager groupManager;
    std::vector<ScenarioCommand> commands;
    std::unordered_map<Guid, TypedResourceId> resourceIds;
    std::vector<ResourceInitialization> initializations;
    std::vector<ResourceOutput> outputs;
};

std::unique_ptr<Scenario> createScenario(const ScenarioOptions &options, ScenarioBuildData buildData);

} // namespace detail
} // namespace mlsdk::scenariorunner
