/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "commands.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace mlsdk::scenariorunner {

/**
 * @brief Construct a new Command base class object
 *
 */
CommandDesc::CommandDesc(CommandType commandType) : commandType(commandType) {}

/**
 * @brief Construct a new Dispatch Compute object
 *
 */
DispatchComputeDesc::DispatchComputeDesc() : CommandDesc(CommandType::DispatchCompute) {}

/**
 * @brief Construct a new Dispatch Graph object
 *
 */
DispatchDataGraphDesc::DispatchDataGraphDesc() : CommandDesc(CommandType::DispatchDataGraph) {}

/**
 * @brief Construct a new Dispatch Barrier object
 *
 */
DispatchBarrierDesc::DispatchBarrierDesc() : CommandDesc(CommandType::DispatchBarrier) {}

/**
 * @brief Construct a new Mark Boundary object
 *
 */
MarkBoundaryDesc::MarkBoundaryDesc() : CommandDesc(CommandType::MarkBoundary) {}

/**
 * @brief Construct a new BindingDesc object
 *
 * @param set
 * @param id
 * @param resourceRef
 */
BindingDesc::BindingDesc(uint32_t set, uint32_t id, Guid resourceRef) : set(set), id(id), resourceRef(resourceRef) {}

/**
 * @brief Construct a new PushConstantMap object
 *
 * @param pushDataRef
 * @param shaderTarget
 */
PushConstantMap::PushConstantMap(Guid pushDataRef, Guid shaderTarget)
    : pushDataRef(pushDataRef), shaderTarget(shaderTarget) {}

/**
 * @brief Construct a new ShaderSubstitutionDesc object
 *
 * @param shaderRef
 * @param target
 */
ShaderSubstitutionDesc::ShaderSubstitutionDesc(Guid shaderRef, const std::string &target)
    : shaderRef(shaderRef), target(target) {}

} // namespace mlsdk::scenariorunner
