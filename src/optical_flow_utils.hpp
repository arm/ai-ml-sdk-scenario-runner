/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "types.hpp"

#include <optional>

namespace mlsdk::scenariorunner {
class DataManager;
enum class OpticalFlowGridSize : uint32_t;

void verifyOpticalFlowConfig(const DataManager &dataManager, const TypedBinding &searchImageBinding,
                             const TypedBinding &templateImageBinding, const TypedBinding &outputImageBinding,
                             const std::optional<TypedBinding> &hintMotionVectorsBinding,
                             const std::optional<TypedBinding> &outputCostBinding, uint32_t width, uint32_t height,
                             OpticalFlowGridSize gridSize);

} // namespace mlsdk::scenariorunner
