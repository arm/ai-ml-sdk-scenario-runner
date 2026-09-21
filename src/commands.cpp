/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
DispatchVgfDesc::DispatchVgfDesc() : CommandDesc(CommandType::DispatchVgf) {}

/**
 * @brief Construct a new Dispatch Spirv Graph object
 *
 */
DispatchDataGraphDesc::DispatchDataGraphDesc() : CommandDesc(CommandType::DispatchDataGraph) {}

/**
 * @brief Construct a new Dispatch Fragment object
 *
 */
DispatchFragmentDesc::DispatchFragmentDesc() : CommandDesc(CommandType::DispatchFragment) {}

/**
 * @brief Construct a new Dispatch Optical Flow object
 *
 */
DispatchOpticalFlowDesc::DispatchOpticalFlowDesc() : CommandDesc(CommandType::DispatchOpticalFlow) {}

/**
 * @brief Construct a new Dispatch Barrier object
 *
 */
PipelineBarrierDesc::PipelineBarrierDesc() : CommandDesc(CommandType::PipelineBarrier) {}

/**
 * @brief Construct a new Mark Boundary object
 *
 */
FrameBoundaryDesc::FrameBoundaryDesc() : CommandDesc(CommandType::FrameBoundary) {}

} // namespace mlsdk::scenariorunner
