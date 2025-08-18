/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "guid.hpp"

#include <optional>
#include <vector>

namespace mlsdk::scenariorunner {

enum class CommandType { Unknown, DispatchCompute, DispatchDataGraph, DispatchBarrier, MarkBoundary };

enum class DescriptorType { Unknown, Auto, StorageImage };

/**
 * @brief Commands are executed by the ScenarioRunner. CommandDesc describes a Command.
 *
 */
struct CommandDesc {
    CommandDesc() = default;
    explicit CommandDesc(CommandType commandType);
    virtual ~CommandDesc() = default;

    CommandType commandType = CommandType::Unknown;
};

/**
 * @brief A Binding maps a resource reference to a Vulkan Descriptor Set and ID. BindingDesc describes a Binding.
 *
 */
struct BindingDesc {
    BindingDesc() = default;
    BindingDesc(uint32_t set, uint32_t id, Guid resourceRef);

    uint32_t set;
    uint32_t id;
    Guid resourceRef;
    std::optional<uint32_t> lod;
    DescriptorType descriptorType = DescriptorType::Auto;
};

/**
 * @brief Maps the raw data resource containing push constants data
 * to the shader node in the graph which consumes these
 *
 */
struct PushConstantMap {
    PushConstantMap() = default;
    explicit PushConstantMap(Guid pushDataRef, Guid shaderTarget);

    Guid pushDataRef;
    Guid shaderTarget;
};

/**
 * @brief Describes a placeholder shader node in the graph that will be substituted with an actual shader
 * implementation
 *
 */
struct ShaderSubstitutionDesc {
    ShaderSubstitutionDesc() = default;
    explicit ShaderSubstitutionDesc(Guid shaderRef, const std::string &target);

    Guid shaderRef;
    std::string target;
};

/**
 * @brief The DispatchCompute command dispatches a compute shader to execute. DispatchComputeDesc describes a
 * DispatchCompute Command.
 *
 */
struct DispatchComputeDesc : CommandDesc {
    DispatchComputeDesc();

    std::string debugName;
    std::vector<BindingDesc> bindings;
    std::vector<uint32_t> rangeND;
    Guid shaderRef;
    bool implicitBarrier{true};
    std::optional<Guid> pushDataRef;
};

/**
 * @brief The DispatchDataGraph command dispatches a data graph shader to execute. DispatchDataGraphDesc describes a
 * DispatchDataGraph.
 *
 */
struct DispatchDataGraphDesc : CommandDesc {
    DispatchDataGraphDesc();

    Guid dataGraphRef;
    std::string debugName;
    std::vector<BindingDesc> bindings;
    std::vector<PushConstantMap> pushConstants = {};
    std::vector<ShaderSubstitutionDesc> shaderSubstitutions = {};
    bool implicitBarrier = true;
};

/**
 * @brief The DispatchBarrier command dispatches a barrier to execute. DispatchBarrierDesc describes a
 * DispatchBarrier Command.
 *
 */
struct DispatchBarrierDesc : CommandDesc {
    DispatchBarrierDesc();

    std::vector<std::string> memoryBarriersRef;
    std::vector<std::string> imageBarriersRef;
    std::vector<std::string> tensorBarriersRef;
    std::vector<std::string> bufferBarriersRef;
};

struct MarkBoundaryDesc : CommandDesc {
    MarkBoundaryDesc();

    std::vector<std::string> resources;
    uint64_t frameId{};
};

} // namespace mlsdk::scenariorunner
