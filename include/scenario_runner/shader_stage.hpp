/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace mlsdk::scenariorunner {

/// \brief Pipeline stage in which a shader executes
enum class ShaderStage {
    Unknown, ///< Shader stage has not been specified
    Compute, ///< Compute shader stage
    Vertex,  ///< Vertex shader stage
    Fragment ///< Fragment shader stage
};

} // namespace mlsdk::scenariorunner
