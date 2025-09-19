/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "commands.hpp"

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

} // namespace mlsdk::scenariorunner
