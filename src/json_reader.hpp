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
// Function to de-serialize CommandDesc from JSON
void from_json(const json &j, CommandDesc &command);

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
// Function to de-serialize a ResourceDesc from JSON
void from_json(const json &j, ResourceDesc &resource);

// Function to de-serialize BufferDesc from JSON
void from_json(const json &j, BufferDesc &buffer);

// Function to de-serialize a SpecializationConstant from JSON
void from_json(const json &j, SpecializationConstant &specializationConstant);

// Function to de-serialize a SpecializationConstantMap from JSON
void from_json(const json &j, SpecializationConstantMap &pushConstantMap);

// Function to de-serialize a ShaderSubstitutionDesc from JSON
void from_json(const json &j, ShaderSubstitutionDesc &shaderSubstitution);

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

// Function to de-serialize SubresourceRange from JSON
void from_json(const json &j, SubresourceRange &subresourceRange);

// Function to de-serialize BufferBarrierDesc from JSON
void from_json(const json &j, BufferBarrierDesc &imageBarrier);

// Map ShaderType values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ShaderType, {{ShaderType::Unknown, nullptr},
                                          {ShaderType::SPIR_V, "SPIR-V"},
                                          {ShaderType::GLSL, "GLSL"}})

// Map ShaderAccessType values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ShaderAccessType, {
                                                   {ShaderAccessType::Unknown, nullptr},
                                                   {ShaderAccessType::ReadOnly, "readonly"},
                                                   {ShaderAccessType::WriteOnly, "writeonly"},
                                                   {ShaderAccessType::ReadWrite, "readwrite"},
                                                   {ShaderAccessType::ImageRead, "image_read"},
                                               })
// Map MemoryAccess values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(MemoryAccess, {{MemoryAccess::Unknown, nullptr},
                                            {MemoryAccess::MemoryWrite, "memory_write"},
                                            {MemoryAccess::MemoryRead, "memory_read"},
                                            {MemoryAccess::GraphWrite, "graph_write"},
                                            {MemoryAccess::GraphRead, "graph_read"},
                                            {MemoryAccess::ComputeShaderWrite, "compute_shader_write"},
                                            {MemoryAccess::ComputeShaderRead, "compute_shader_read"}})

NLOHMANN_JSON_SERIALIZE_ENUM(PipelineStage, {{PipelineStage::Unknown, nullptr},
                                             {PipelineStage::Graph, "graph"},
                                             {PipelineStage::Compute, "compute"},
                                             {PipelineStage::All, "all"}})

// Map ImageLayout values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ImageLayout, {{ImageLayout::Unknown, nullptr},
                                           {ImageLayout::TensorAliasing, "tensor_aliasing"},
                                           {ImageLayout::General, "general"},
                                           {ImageLayout::Undefined, "undefined"}})

// Function to de-serialize RawDataDesc from JSON
void from_json(const json &j, RawDataDesc &raw_data);

// Function to de-serialize AliasTarget from JSON
void from_json(const json &j, AliasTarget &target);

// Function to de-serialize TensorDesc from JSON
void from_json(const json &j, TensorDesc &tensor);

// Function to de-serialize ImageDesc from JSON
void from_json(const json &j, ImageDesc &image);

// Map values of sampler settings to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(FilterMode, {{FilterMode::Unknown, nullptr},
                                          {FilterMode::Nearest, "NEAREST"},
                                          {FilterMode::Linear, "LINEAR"}})

NLOHMANN_JSON_SERIALIZE_ENUM(AddressMode, {{AddressMode::Unknown, nullptr},
                                           {AddressMode::ClampBorder, "CLAMP_BORDER"},
                                           {AddressMode::ClampEdge, "CLAMP_EDGE"},
                                           {AddressMode::Repeat, "REPEAT"},
                                           {AddressMode::MirroredRepeat, "MIRRORED_REPEAT"}})

NLOHMANN_JSON_SERIALIZE_ENUM(BorderColor, {{BorderColor::Unknown, nullptr},
                                           {BorderColor::FloatTransparentBlack, "FLOAT_TRANSPARENT_BLACK"},
                                           {BorderColor::FloatOpaqueBlack, "FLOAT_OPAQUE_BLACK"},
                                           {BorderColor::FloatOpaqueWhite, "FLOAT_OPAQUE_WHITE"},
                                           {BorderColor::IntTransparentBlack, "INT_TRANSPARENT_BLACK"},
                                           {BorderColor::IntOpaqueBlack, "INT_OPAQUE_BLACK"},
                                           {BorderColor::IntOpaqueWhite, "INT_OPAQUE_WHITE"},
                                           {BorderColor::FloatCustomEXT, "FLOAT_CUSTOM_EXT"},
                                           {BorderColor::IntCustomEXT, "INT_CUSTOM_EXT"}})

NLOHMANN_JSON_SERIALIZE_ENUM(DescriptorType, {{DescriptorType::Unknown, nullptr},
                                              {DescriptorType::Auto, "VK_DESCRIPTOR_TYPE_AUTO"},
                                              {DescriptorType::StorageImage, "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"}})

NLOHMANN_JSON_SERIALIZE_ENUM(Tiling,
                             {{Tiling::Unknown, nullptr}, {Tiling::Optimal, "OPTIMAL"}, {Tiling::Linear, "LINEAR"}})

} // namespace mlsdk::scenariorunner
