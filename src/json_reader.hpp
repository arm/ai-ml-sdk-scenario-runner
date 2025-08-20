/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "commands.hpp"
#include "resource_desc.hpp"

#include "nlohmann/json.hpp"

#include <istream>

namespace mlsdk::scenariorunner {

struct ScenarioSpec;

using json = nlohmann::json;

void readJson(ScenarioSpec &scenarioSpec, std::istream *is);

//==============
// Commands
// Function to de-serialize DispatchComputeDesc from JSON
void from_json(const json &j, DispatchComputeDesc &dispatchCompute);

// Function to de-serialize DispatchGraphDesc from JSON
void from_json(const json &j, DispatchDataGraphDesc &dispatchDataGraph);

// Function to de-serialize a BindingDesc from JSON
void from_json(const json &j, BindingDesc &binding);

// Function to de-serialize a PushConstantMap from JSON
void from_json(const json &j, PushConstantMap &pushConstantMap);

//==============
// Resources
// Function to de-serialize BufferDesc from JSON
void from_json(const json &j, BufferDesc &buffer);

// Function to de-serialize DataGraphDesc from JSON
void from_json(const json &j, DataGraphDesc &graph);

// Function to de-serialize ShaderDesc from JSON
void from_json(const json &j, ShaderDesc &shader);

// Function to de-serialize MemoryBarrierDesc from JSON
void from_json(const json &j, MemoryBarrierDesc &memoryBarrier);

// Function to de-serialize TensorBarrierDesc from JSON
void from_json(const json &j, TensorBarrierDesc &memoryBarrier);

// Function to de-serialize ImageBarrierDesc from JSON
void from_json(const json &j, ImageBarrierDesc &imageBarrier);

// Function to de-serialize BufferBarrierDesc from JSON
void from_json(const json &j, BufferBarrierDesc &imageBarrier);

// Function to de-serialize RawDataDesc from JSON
void from_json(const json &j, RawDataDesc &raw_data);

// Function to de-serialize TensorDesc from JSON
void from_json(const json &j, TensorDesc &tensor);

// Function to de-serialize ImageDesc from JSON
void from_json(const json &j, ImageDesc &image);

} // namespace mlsdk::scenariorunner
