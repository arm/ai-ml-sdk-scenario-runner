/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "json_reader.hpp"

#include "logging.hpp"
#include "scenario_desc.hpp"

#include <string_view>

using namespace std::string_view_literals;

namespace mlsdk::scenariorunner {
namespace {
void parseOptionalPipelineStages(const json &j, std::string_view fieldName, std::vector<PipelineStage> &stages) {
    const auto stagesIter = j.find(fieldName);
    if (stagesIter == j.end()) {
        return;
    }

    stages = stagesIter.value().get<std::vector<PipelineStage>>();
    for (const PipelineStage stage : stages) {
        if (stage == PipelineStage::Unknown) {
            throw std::runtime_error(std::string("Unknown ") + std::string(fieldName) + " value");
        }
    }
}

void parseResourceDescGuid(const json &j, ResourceDesc &resource) {
    resource.guidStr = j.at("uid"sv).get<std::string>();
    resource.guid = resource.guidStr;
}

MemoryAccess parseRequiredMemoryAccess(const json &j, std::string_view fieldName) {
    const MemoryAccess access = j.at(fieldName).get<MemoryAccess>();
    if (access == MemoryAccess::Unknown) {
        throw std::runtime_error(std::string("Unknown ") + std::string(fieldName) + " value");
    }

    return access;
}

void parseBaseBarrierDesc(const json &j, BaseBarrierDesc &barrier) {
    parseResourceDescGuid(j, barrier);
    barrier.srcAccess = parseRequiredMemoryAccess(j, "src_access");
    barrier.dstAccess = parseRequiredMemoryAccess(j, "dst_access");
    parseOptionalPipelineStages(j, "src_stage", barrier.srcStages);
    parseOptionalPipelineStages(j, "dst_stage", barrier.dstStages);
}

template <typename ParsedType, typename ValueType>
void parseOptionalFieldAs(const json &j, std::string_view fieldName, ValueType &value) {
    if (const auto it = j.find(fieldName); it != j.end()) {
        value = it->get<ParsedType>();
    }
}

template <typename ValueType> void parseOptionalField(const json &j, std::string_view fieldName, ValueType &value) {
    parseOptionalFieldAs<ValueType>(j, fieldName, value);
}

} // namespace

//==============
// Command details
void from_json(const json &j, DispatchFragmentDesc &dispatchFragment);
void from_json(const json &j, DispatchBarrierDesc &dispatchBarrier);
void from_json(const json &j, MarkBoundaryDesc &markBoundaryDesc);

//==============
// Resource details
// Function to de-serialize a SpecializationConstant from JSON
void from_json(const json &j, SpecializationConstant &specializationConstant);

// Function to de-serialize a SpecializationConstantMap from JSON
void from_json(const json &j, SpecializationConstantMap &specializationConstantMap);

// Function to de-serialize a ShaderSubstitution from JSON
void from_json(const json &j, ShaderSubstitution &shaderSubstitution);

// Function to de-serialize SubresourceRange from JSON
void from_json(const json &j, SubresourceRange &subresourceRange);

//==============
// Enums
NLOHMANN_JSON_SERIALIZE_ENUM(ResourceType, {{ResourceType::Unknown, nullptr},
                                            {ResourceType::Shader, "shader"},
                                            {ResourceType::Buffer, "buffer"},
                                            {ResourceType::DataGraph, "graph"},
                                            {ResourceType::RawData, "raw_data"},
                                            {ResourceType::Tensor, "tensor"},
                                            {ResourceType::Image, "image"},
                                            {ResourceType::ImageBarrier, "image_barrier"},
                                            {ResourceType::MemoryBarrier, "memory_barrier"},
                                            {ResourceType::TensorBarrier, "tensor_barrier"},
                                            {ResourceType::BufferBarrier, "buffer_barrier"},
                                            {ResourceType::GraphConstant, "graph_constant"}})

NLOHMANN_JSON_SERIALIZE_ENUM(CommandType, {{CommandType::Unknown, nullptr},
                                           {CommandType::DispatchCompute, "dispatch_compute"},
                                           {CommandType::DispatchDataGraph, "dispatch_graph"},
                                           {CommandType::DispatchSpirvGraph, "dispatch_spirv_graph"},
                                           {CommandType::DispatchFragment, "dispatch_fragment"},
                                           {CommandType::DispatchBarrier, "dispatch_barrier"},
                                           {CommandType::DispatchOpticalFlow, "dispatch_optical_flow"},
                                           {CommandType::MarkBoundary, "mark_boundary"}})

// Map ShaderType values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ShaderType, {{ShaderType::Unknown, nullptr},
                                          {ShaderType::SPIR_V, "SPIR-V"},
                                          {ShaderType::GLSL, "GLSL"},
                                          {ShaderType::HLSL, "HLSL"}})

NLOHMANN_JSON_SERIALIZE_ENUM(ShaderStage, {{ShaderStage::Unknown, nullptr},
                                           {ShaderStage::Compute, "compute"},
                                           {ShaderStage::Vertex, "vertex"},
                                           {ShaderStage::Fragment, "fragment"}})

// Map ShaderAccessType values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ShaderAccessType, {{ShaderAccessType::Unknown, nullptr},
                                                {ShaderAccessType::ReadOnly, "readonly"},
                                                {ShaderAccessType::WriteOnly, "writeonly"},
                                                {ShaderAccessType::ReadWrite, "readwrite"},
                                                {ShaderAccessType::ImageRead, "image_read"}})
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
                                             {PipelineStage::Graphics, "graphics"},
                                             {PipelineStage::All, "all"}})

// Map ImageLayout values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ImageLayout, {{ImageLayout::Unknown, nullptr},
                                           {ImageLayout::TensorAliasing, "tensor_aliasing"},
                                           {ImageLayout::General, "general"},
                                           {ImageLayout::Undefined, "undefined"}})

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

NLOHMANN_JSON_SERIALIZE_ENUM(OpticalFlowGridSize, {{OpticalFlowGridSize::Invalid, nullptr},
                                                   {OpticalFlowGridSize::e1x1, "1x1"},
                                                   {OpticalFlowGridSize::e2x2, "2x2"},
                                                   {OpticalFlowGridSize::e4x4, "4x4"},
                                                   {OpticalFlowGridSize::e8x8, "8x8"}})

NLOHMANN_JSON_SERIALIZE_ENUM(OpticalFlowPerformanceLevel, {{OpticalFlowPerformanceLevel::Invalid, nullptr},
                                                           {OpticalFlowPerformanceLevel::Unknown, "unknown"},
                                                           {OpticalFlowPerformanceLevel::Slow, "slow"},
                                                           {OpticalFlowPerformanceLevel::Medium, "medium"},
                                                           {OpticalFlowPerformanceLevel::Fast, "fast"}})

NLOHMANN_JSON_SERIALIZE_ENUM(OpticalFlowExecutionFlag,
                             {{OpticalFlowExecutionFlag::Invalid, nullptr},
                              {OpticalFlowExecutionFlag::DisableTemporalHints, "disable_temporal_hints"},
                              {OpticalFlowExecutionFlag::InputUnchanged, "input_unchanged"},
                              {OpticalFlowExecutionFlag::ReferenceUnchanged, "reference_unchanged"},
                              {OpticalFlowExecutionFlag::InputIsPreviousReference, "input_is_previous_reference"},
                              {OpticalFlowExecutionFlag::ReferenceIsPreviousInput, "reference_is_previous_input"}})

void readJsonImpl(ScenarioSpec &scenarioSpec, const json &j);

void readJson(ScenarioSpec &scenarioSpec, std::istream &is) {
    const auto j = json::parse(is);
    readJsonImpl(scenarioSpec, j);
}

void readJson(ScenarioSpec &scenarioSpec, const std::string &jsonStr) {
    const auto j = json::parse(jsonStr);
    readJsonImpl(scenarioSpec, j);
}

void readJsonImpl(ScenarioSpec &scenarioSpec, const json &j) {
    const json &resourcesJson = j.at("resources"sv);
    for (const auto &resourceJson : resourcesJson) {
        const auto resourceType = json(resourceJson.begin().key()).get<ResourceType>();
        switch (resourceType) {
        case (ResourceType::Shader): {
            auto shader = resourceJson.at("shader"sv).get<ShaderDesc>();
            scenarioSpec.addResource(std::make_unique<ShaderDesc>(std::move(shader)));
        } break;
        case (ResourceType::Buffer): {
            auto buffer = resourceJson.at("buffer"sv).get<BufferDesc>();
            scenarioSpec.addResource(std::make_unique<BufferDesc>(std::move(buffer)));
        } break;
        case (ResourceType::RawData): {
            auto rawData = resourceJson.at("raw_data"sv).get<RawDataDesc>();
            scenarioSpec.addResource(std::make_unique<RawDataDesc>(std::move(rawData)));
        } break;
        case (ResourceType::DataGraph): {
            auto dataGraph = resourceJson.at("graph"sv).get<DataGraphDesc>();
            scenarioSpec.addResource(std::make_unique<DataGraphDesc>(std::move(dataGraph)));
        } break;
        case (ResourceType::Tensor): {
            auto tensor = resourceJson.at("tensor"sv).get<TensorDesc>();
            scenarioSpec.addResource(std::make_unique<TensorDesc>(std::move(tensor)));
        } break;
        case (ResourceType::Image): {
            auto image = resourceJson.at("image"sv).get<ImageDesc>();
            scenarioSpec.addResource(std::make_unique<ImageDesc>(std::move(image)));
        } break;
        case (ResourceType::ImageBarrier): {
            auto imageBarrier = resourceJson.at("image_barrier"sv).get<ImageBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<ImageBarrierDesc>(std::move(imageBarrier)));
        } break;
        case (ResourceType::TensorBarrier): {
            auto tensorBarrier = resourceJson.at("tensor_barrier"sv).get<TensorBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<TensorBarrierDesc>(std::move(tensorBarrier)));
        } break;
        case (ResourceType::MemoryBarrier): {
            auto memoryBarrier = resourceJson.at("memory_barrier"sv).get<MemoryBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<MemoryBarrierDesc>(std::move(memoryBarrier)));
        } break;
        case (ResourceType::BufferBarrier): {
            auto bufferBarrier = resourceJson.at("buffer_barrier"sv).get<BufferBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<BufferBarrierDesc>(std::move(bufferBarrier)));
        } break;
        case (ResourceType::GraphConstant): {
            auto graphConstant = resourceJson.at("graph_constant"sv).get<GraphConstantDesc>();
            scenarioSpec.addResource(std::make_unique<GraphConstantDesc>(std::move(graphConstant)));
        } break;
        default:
            throw std::runtime_error("Unknown Resource type in resources");
        }
    }

    const json &commandsJson = j.at("commands"sv);
    for (const auto &commandJson : commandsJson) {
        const auto commandType = json(commandJson.begin().key()).get<CommandType>();
        switch (commandType) {
        case (CommandType::DispatchCompute): {
            auto dispatchCompute = commandJson.at("dispatch_compute"sv).get<DispatchComputeDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchComputeDesc>(std::move(dispatchCompute)));
        } break;
        case (CommandType::DispatchDataGraph): {
            auto dispatchDataGraph = commandJson.at("dispatch_graph"sv).get<DispatchDataGraphDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchDataGraphDesc>(std::move(dispatchDataGraph)));
        } break;
        case (CommandType::DispatchSpirvGraph): {
            auto dispatchSpirvGraph = commandJson.at("dispatch_spirv_graph"sv).get<DispatchSpirvGraphDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchSpirvGraphDesc>(std::move(dispatchSpirvGraph)));
        } break;
        case (CommandType::DispatchFragment): {
            auto dispatchFragment = commandJson.at("dispatch_fragment"sv).get<DispatchFragmentDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchFragmentDesc>(std::move(dispatchFragment)));
        } break;
        case (CommandType::DispatchBarrier): {
            auto dispatchBarrier = commandJson.at("dispatch_barrier"sv).get<DispatchBarrierDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchBarrierDesc>(std::move(dispatchBarrier)));
        } break;
        case (CommandType::MarkBoundary): {
            auto markBoundary = commandJson.at("mark_boundary"sv).get<MarkBoundaryDesc>();
            scenarioSpec.addCommand(std::make_unique<MarkBoundaryDesc>(std::move(markBoundary)));
        } break;
        case (CommandType::DispatchOpticalFlow): {
            auto dispatchOpticalFlow = commandJson.at("dispatch_optical_flow"sv).get<DispatchOpticalFlowDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchOpticalFlowDesc>(std::move(dispatchOpticalFlow)));
        } break;
        default:
            throw std::runtime_error("Unknown Command type in commands");
        }
    }
}

/**
 * @brief De-serialize DispatchComputeDesc from JSON.
 *
 * @param j
 * @param dispatchCompute
 */
void from_json(const json &j, DispatchComputeDesc &dispatchCompute) {
    dispatchCompute.bindings = j.at("bindings"sv).get<std::vector<BindingDesc>>();
    dispatchCompute.rangeND = j.at("rangeND"sv).get<std::vector<uint32_t>>();
    for (size_t i = dispatchCompute.rangeND.size(); i < 3; ++i) {
        dispatchCompute.rangeND.push_back(1);
    }
    dispatchCompute.debugName = j.at("shader_ref"sv).get<std::string>();
    dispatchCompute.shaderRef = dispatchCompute.debugName;
    parseOptionalFieldAs<std::string>(j, "push_data_ref", dispatchCompute.pushDataRef);
    parseOptionalField(j, "implicit_barrier", dispatchCompute.implicitBarrier);
}

void from_json(const json &j, FragmentAttachmentDesc &fragmentAttachment) {
    if (j.is_string()) {
        fragmentAttachment.resourceRef = j.get<std::string>();
    } else if (j.is_object()) {
        fragmentAttachment.resourceRef = j.at("resource_ref"sv).get<std::string>();
        parseOptionalFieldAs<uint32_t>(j, "lod", fragmentAttachment.lod);
    } else {
        throw std::runtime_error("color_attachment_refs entries must be strings or objects");
    }
}

void from_json(const json &j, DispatchFragmentDesc &dispatchFragment) {
    dispatchFragment.bindings = j.at("bindings"sv).get<std::vector<BindingDesc>>();

    dispatchFragment.vertexShaderRef = j.at("vertex_shader_ref"sv).get<std::string>();
    dispatchFragment.debugName = j.at("fragment_shader_ref"sv).get<std::string>();
    dispatchFragment.fragmentShaderRef = dispatchFragment.debugName;
    parseOptionalField(j, "debug_name", dispatchFragment.debugName);
    parseOptionalField(j, "color_attachment_refs", dispatchFragment.colorAttachments);
    parseOptionalFieldAs<std::array<uint32_t, 2>>(j, "render_extent", dispatchFragment.renderExtent);
    if (dispatchFragment.colorAttachments.empty() && !dispatchFragment.renderExtent.has_value()) {
        throw std::runtime_error("dispatch_fragment requires color_attachment_refs or render_extent");
    }
    parseOptionalFieldAs<std::string>(j, "push_data_ref", dispatchFragment.pushDataRef);
    parseOptionalField(j, "implicit_barrier", dispatchFragment.implicitBarrier);
}

/**
 * @brief De-serialize DispatchDataGraphDesc from JSON.
 *
 * @param j
 * @param dispatchDataGraph
 */
void from_json(const json &j, DispatchDataGraphDesc &dispatchDataGraph) {
    dispatchDataGraph.debugName = j.at("graph_ref"sv).get<std::string>();
    dispatchDataGraph.dataGraphRef = dispatchDataGraph.debugName;
    dispatchDataGraph.bindings = j.at("bindings"sv).get<std::vector<BindingDesc>>();

    parseOptionalField(j, "push_constants", dispatchDataGraph.pushConstants);
    parseOptionalField(j, "shader_substitutions", dispatchDataGraph.shaderSubstitutions);
    parseOptionalField(j, "implicit_barrier", dispatchDataGraph.implicitBarrier);
}

/**
 * @brief De-serialize DispatchSpirvGraphDesc from JSON.
 *
 * @param j
 * @param dispatchSpirvGraph
 */
void from_json(const json &j, DispatchSpirvGraphDesc &dispatchSpirvGraph) {
    dispatchSpirvGraph.debugName = j.at("graph_ref"sv).get<std::string>();
    dispatchSpirvGraph.dataGraphRef = dispatchSpirvGraph.debugName;
    dispatchSpirvGraph.bindings = j.at("bindings"sv).get<std::vector<BindingDesc>>();
    if (const auto it = j.find("graph_constants"sv); it != j.end()) {
        const auto graphConstants = it->get<std::vector<std::string>>();
        dispatchSpirvGraph.graphConstants.assign(graphConstants.begin(), graphConstants.end());
    }
    parseOptionalField(j, "implicit_barrier", dispatchSpirvGraph.implicitBarrier);
}

/**
 * @brief De-serialize DispatchOpticalFlowDesc from JSON.
 *
 * @param j
 * @param dispatchOpticalFlow
 */
void from_json(const json &j, DispatchOpticalFlowDesc &dispatchOpticalFlow) {
    dispatchOpticalFlow.debugName = "dispatch_optical_flow";

    dispatchOpticalFlow.width = j.at("width"sv).get<uint32_t>();
    dispatchOpticalFlow.height = j.at("height"sv).get<uint32_t>();
    if (dispatchOpticalFlow.width == 0 || dispatchOpticalFlow.height == 0) {
        throw std::runtime_error("Optical flow width and height must be > 0");
    }
    dispatchOpticalFlow.gridSize = j.at("grid_size"sv).get<OpticalFlowGridSize>();
    if (dispatchOpticalFlow.gridSize == OpticalFlowGridSize::Invalid) {
        throw std::runtime_error("Invalid optical flow grid_size value. Expected one of: 1x1, 2x2, 4x4, 8x8.");
    }

    parseOptionalField(j, "performance_level", dispatchOpticalFlow.performanceLevel);
    if (dispatchOpticalFlow.performanceLevel == OpticalFlowPerformanceLevel::Invalid) {
        throw std::runtime_error(
            "Invalid optical flow performance_level value. Expected one of: unknown, slow, medium, fast.");
    }
    parseOptionalField(j, "mean_flow_l1_norm_hint", dispatchOpticalFlow.meanFlowL1NormHint);
    if (dispatchOpticalFlow.meanFlowL1NormHint >= std::max(dispatchOpticalFlow.width, dispatchOpticalFlow.height)) {
        throw std::runtime_error(
            "Invalid optical flow mean_flow_l1_norm_hint value. Expected 0 or < max(width, height).");
    }

    parseOptionalField(j, "implicit_barrier", dispatchOpticalFlow.implicitBarrier);

    if (const auto it = j.find("execution_flags"sv); it != j.end()) {
        auto flags = it->get<std::vector<OpticalFlowExecutionFlag>>();
        for (const auto flag : flags) {
            if (flag == OpticalFlowExecutionFlag::Invalid) {
                throw std::runtime_error("Invalid optical flow execution_flags value.");
            }
            dispatchOpticalFlow.executionFlags |= static_cast<uint32_t>(flag);
        }
    }
    constexpr uint32_t validExecutionFlagsMask = 0x1Fu;
    if ((dispatchOpticalFlow.executionFlags & ~validExecutionFlagsMask) != 0u) {
        throw std::runtime_error("Invalid optical flow execution_flags value.");
    }

    const json &bindingsJson = j.at("bindings"sv);
    dispatchOpticalFlow.searchImage = bindingsJson.at("search_image"sv).get<BindingDesc>();
    dispatchOpticalFlow.templateImage = bindingsJson.at("template_image"sv).get<BindingDesc>();
    dispatchOpticalFlow.outputImage = bindingsJson.at("output_image"sv).get<BindingDesc>();

    parseOptionalFieldAs<BindingDesc>(j, "hint_motion_vectors", dispatchOpticalFlow.hintMotionVectors);
    parseOptionalFieldAs<BindingDesc>(j, "output_cost", dispatchOpticalFlow.outputCost);
}

/**
 * @brief De-serialize DispatchBarrierDesc from JSON.
 *
 * @param j
 * @param dispatchBarrier
 */
void from_json(const json &j, DispatchBarrierDesc &dispatchBarrier) {
    dispatchBarrier.imageBarriersRef = j.at("image_barrier_refs"sv).get<std::vector<std::string>>();

    parseOptionalField(j, "tensor_barrier_refs", dispatchBarrier.tensorBarriersRef);
    dispatchBarrier.memoryBarriersRef = j.at("memory_barrier_refs"sv).get<std::vector<std::string>>();
    dispatchBarrier.bufferBarriersRef = j.at("buffer_barrier_refs"sv).get<std::vector<std::string>>();
}

/**
 * @brief De-serialize MarkBoundary from JSON.
 *
 * @param j
 * @param markBoundaryDesc
 */
void from_json(const json &j, MarkBoundaryDesc &markBoundaryDesc) {
    if (j.contains("frame_id"sv)) {
        mlsdk::logging::warning("Manual setting of frameID is deprecated");
    }
    markBoundaryDesc.resources = j.at("resources"sv).get<std::vector<std::string>>();
}

/**
 * @brief De-serialize BindingDesc from JSON.
 *
 * @param j
 * @param binding
 */
void from_json(const json &j, BindingDesc &binding) {
    binding.set = j.at("set"sv).get<uint32_t>();
    binding.id = j.at("id"sv).get<uint32_t>();
    parseOptionalFieldAs<int>(j, "lod", binding.lod);
    binding.resourceRef = j.at("resource_ref"sv).get<std::string>();
    parseOptionalField(j, "descriptor_type", binding.descriptorType);
    if (binding.descriptorType == DescriptorType::Unknown) {
        throw std::runtime_error("Unknown descriptor_type value");
    }
}

/**
 * @brief De-serialize PushConstantMap from JSON.
 *
 * @param j
 * @param pushConstantMap
 */
void from_json(const json &j, PushConstantMap &pushConstantMap) {
    pushConstantMap.pushDataRef = j.at("push_data_ref"sv).get<std::string>();
    pushConstantMap.shaderTarget = j.at("shader_target"sv).get<std::string>();
}

/**
 * @brief De-serialize MemoryGroup from JSON.
 *
 * @param j
 * @param group
 */
void from_json(const json &j, MemoryGroup &group) {
    group.memoryUid = j.at("id"sv);
    parseOptionalField(j, "offset", group.offset);
}

//====================
// Resources
//====================

/**
 * @brief De-serialize BufferDesc from JSON.
 *
 * @param j
 * @param buffer
 */
void from_json(const json &j, BufferDesc &buffer) {
    parseResourceDescGuid(j, buffer);
    buffer.size = j.at("size"sv).get<uint32_t>();
    buffer.shaderAccess = j.at("shader_access"sv).get<ShaderAccessType>();
    if (buffer.shaderAccess == ShaderAccessType::Unknown) {
        throw std::runtime_error("Unknown shader_access value");
    }
    parseOptionalFieldAs<std::string>(j, "src", buffer.src);
    parseOptionalFieldAs<std::string>(j, "dst", buffer.dst);
    parseOptionalFieldAs<MemoryGroup>(j, "memory_group", buffer.memoryGroup);
}

/**
 * @brief De-serialize SpecializationConstant from JSON.
 *
 * @param j
 * @param specializationConstant
 */
void from_json(const json &j, SpecializationConstant &specializationConstant) {
    specializationConstant.id = j.at("id"sv).get<int>();
    const auto &val = j.at("value"sv);
    if (val.is_boolean()) {
        specializationConstant.value.ui = val.get<bool>() ? 1u : 0u;
    } else if (val.is_number_unsigned()) {
        specializationConstant.value.ui = val.get<uint32_t>();
    } else if (val.is_number_integer()) {
        specializationConstant.value.i = val.get<int32_t>();
    } else if (val.is_number_float()) {
        specializationConstant.value.f = val.get<float>();
    } else {
        throw std::runtime_error("Unknown specialization constant value type");
    }
}

/**
 * @brief De-serialize SpecializationConstantMap from JSON.
 *
 * @param j
 * @param specializationConstantMap
 */
void from_json(const json &j, SpecializationConstantMap &specializationConstantMap) {
    specializationConstantMap.shaderTarget = j.at("shader_target"sv).get<std::string>();
    specializationConstantMap.specializationConstants =
        j.at("specialization_constants"sv).get<std::vector<SpecializationConstant>>();
}

/**
 * @brief De-serialize ShaderSubstitution from JSON.
 *
 * @param j
 * @param shaderSubstitution
 */
void from_json(const json &j, ShaderSubstitution &shaderSubstitution) {
    shaderSubstitution.shaderRef = j.at("shader_ref"sv).get<std::string>();
    shaderSubstitution.target = j.at("target"sv).get<std::string>();
}

/**
 * @brief De-serialize GraphDesc from JSON.
 *
 * @param j
 * @param dataGraph
 */
void from_json(const json &j, DataGraphDesc &dataGraph) {
    parseResourceDescGuid(j, dataGraph);
    dataGraph.src = j.at("src"sv).get<std::string>();
    parseOptionalField(j, "shader_substitutions", dataGraph.shaderSubstitutions);
    parseOptionalField(j, "specialization_constants_map", dataGraph.specializationConstantMaps);
    parseOptionalField(j, "push_constants_size", dataGraph.pushConstantsSize);
}

/**
 * @brief De-serialize ShaderDesc from JSON.
 *
 * @param j
 * @param shader
 */
void from_json(const json &j, ShaderDesc &shader) {
    parseResourceDescGuid(j, shader);
    shader.src = j.at("src"sv).get<std::string>();
    shader.shaderType = j.at("type"sv).get<ShaderType>();
    if (shader.shaderType == ShaderType::Unknown) {
        throw std::runtime_error("Unknown shader type value");
    }
    parseOptionalField(j, "stage", shader.stage);
    if (shader.stage == ShaderStage::Unknown) {
        throw std::runtime_error("Unknown shader stage value");
    }
    parseOptionalField(j, "entry", shader.entry);
    if (shader.shaderType == ShaderType::GLSL && shader.entry != "main"sv) {
        throw std::runtime_error("GLSL is required to have an entrypoint of 'main'");
    }
    parseOptionalField(j, "push_constants_size", shader.pushConstantsSize);
    parseOptionalField(j, "specialization_constants", shader.specializationConstants);
    parseOptionalField(j, "build_options", shader.buildOpts);
    parseOptionalField(j, "include_dirs", shader.includeDirs);
}

/**
 * @brief De-serialize RawDataDesc from JSON.
 *
 * @param j
 * @param rawData
 */
void from_json(const json &j, RawDataDesc &rawData) {
    parseResourceDescGuid(j, rawData);
    rawData.src = j.at("src"sv).get<std::string>();
}

/**
 * @brief De-serialize GraphConstantDesc from JSON.
 *
 * @param j
 * @param graphConstant
 */
void from_json(const json &j, GraphConstantDesc &graphConstant) {
    parseResourceDescGuid(j, graphConstant);
    const auto dims = j.at("dims"sv).get<std::vector<int64_t>>();
    if (dims.empty() || dims.size() > 6) {
        throw std::runtime_error("Invalid graph constant dimensions");
    }
    graphConstant.dims = dims;
    graphConstant.src = j.at("src"sv).get<std::string>();
    graphConstant.format = j.at("format"sv).get<std::string>();
}

/**
 * @brief De-serialize TensorDesc from JSON.
 *
 * @param j
 * @param tensor
 */
void from_json(const json &j, TensorDesc &tensor) {
    parseResourceDescGuid(j, tensor);
    tensor.dims = j.at("dims"sv).get<std::vector<int64_t>>();
    tensor.format = j.at("format"sv).get<std::string>();
    tensor.shaderAccess = j.at("shader_access"sv).get<ShaderAccessType>();
    if (tensor.shaderAccess == ShaderAccessType::Unknown) {
        throw std::runtime_error("Unknown shader_access type");
    }
    parseOptionalFieldAs<std::string>(j, "src", tensor.src);
    parseOptionalFieldAs<std::string>(j, "dst", tensor.dst);
    if (j.find("alias_target"sv) != j.end()) {
        throw std::runtime_error(
            "The alias_target field is deprecated, please use the memory_group functionality instead");
    }
    parseOptionalFieldAs<MemoryGroup>(j, "memory_group", tensor.memoryGroup);
    parseOptionalFieldAs<Tiling>(j, "tiling", tensor.tiling);
    if (tensor.tiling.has_value() && tensor.tiling.value() == Tiling::Unknown) {
        throw std::runtime_error("Unknown tiling value");
    }
}

/**
 * @brief De-serialize ImageDesc from JSON.
 *
 * @param j
 * @param image
 */
void from_json(const json &j, ImageDesc &image) {
    parseResourceDescGuid(j, image);
    image.dims = j.at("dims"sv).get<std::vector<uint32_t>>();
    // for compatibility with the old json configs that had this field set as "false"
    if (const auto it = j.find("mips"sv); it == j.end()) {
        image.mips = 1;
    } else if (it.value().is_boolean()) {
        mlsdk::logging::warning("Boolean mips flag is deprecated, defaulting to \"1\". Use integer value instead");
        image.mips = 1;
    } else {
        image.mips = it->get<uint32_t>();
    }

    image.format = j.at("format"sv).get<std::string>();
    image.shaderAccess = j.at("shader_access"sv).get<ShaderAccessType>();
    if (image.shaderAccess == ShaderAccessType::Unknown) {
        throw std::runtime_error("Unknown shader_access type");
    }
    parseOptionalFieldAs<std::string>(j, "src", image.src);
    parseOptionalFieldAs<std::string>(j, "dst", image.dst);
    parseOptionalFieldAs<bool>(j, "color_attachment", image.colorAttachment);
    parseOptionalFieldAs<FilterMode>(j, "min_filter", image.minFilter);
    if (image.minFilter.has_value() && image.minFilter.value() == FilterMode::Unknown) {
        throw std::runtime_error("Unknown min_filter value");
    }
    parseOptionalFieldAs<FilterMode>(j, "mag_filter", image.magFilter);
    if (image.magFilter.has_value() && image.magFilter.value() == FilterMode::Unknown) {
        throw std::runtime_error("Unknown mag_filter value");
    }
    parseOptionalFieldAs<FilterMode>(j, "mip_filter", image.mipFilter);
    if (image.mipFilter.has_value() && image.mipFilter.value() == FilterMode::Unknown) {
        throw std::runtime_error("Unknown mip_filter value");
    }
    parseOptionalFieldAs<AddressMode>(j, "border_address_mode", image.borderAddressMode);
    if (image.borderAddressMode.has_value() && image.borderAddressMode.value() == AddressMode::Unknown) {
        throw std::runtime_error("Unknown border_address_mode value");
    }
    parseOptionalFieldAs<BorderColor>(j, "border_color", image.borderColor);
    if (image.borderColor.has_value() && image.borderColor.value() == BorderColor::Unknown) {
        throw std::runtime_error("Unknown border_color value");
    }
    if (const auto it = j.find("custom_border_color"sv); it != j.end()) {
        if (!image.borderColor.has_value()) {
            throw std::runtime_error("custom_border_color requires border_color");
        }

        const json &customColorJson = it.value();
        if (image.borderColor.value() == BorderColor::FloatCustomEXT) {
            image.customBorderColor = customColorJson.get<std::array<float, 4>>();
        } else {
            image.customBorderColor = customColorJson.get<std::array<int, 4>>();
        }
    }
    parseOptionalFieldAs<Tiling>(j, "tiling", image.tiling);
    if (image.tiling.has_value() && image.tiling.value() == Tiling::Unknown) {
        throw std::runtime_error("Unknown tiling value");
    }
    parseOptionalFieldAs<MemoryGroup>(j, "memory_group", image.memoryGroup);
}

/**
 * @brief De-serialize Memory Barrier from JSON.
 *
 * @param j
 * @param memoryBarrier
 */
void from_json(const json &j, MemoryBarrierDesc &memoryBarrier) { parseBaseBarrierDesc(j, memoryBarrier); }

/**
 * @brief De-serialize Tensor Barrier from JSON.
 *
 * @param j json object
 * @param tensorBarrier tensor barrier description struct
 */
void from_json(const json &j, TensorBarrierDesc &tensorBarrier) {
    parseBaseBarrierDesc(j, tensorBarrier);

    tensorBarrier.tensorResource = j.at("tensor_resource"sv).get<std::string>();
}

/**
 * @brief De-serialize ImageBarrier from JSON.
 *
 * @param j
 * @param imageBarrier
 */
void from_json(const json &j, ImageBarrierDesc &imageBarrier) {
    parseBaseBarrierDesc(j, imageBarrier);
    imageBarrier.oldLayout = j.at("old_layout"sv).get<ImageLayout>();
    if (imageBarrier.oldLayout == ImageLayout::Unknown) {
        throw std::runtime_error("Unknown old_layout value");
    }
    imageBarrier.newLayout = j.at("new_layout"sv).get<ImageLayout>();
    if (imageBarrier.newLayout == ImageLayout::Unknown) {
        throw std::runtime_error("Unknown new_layout value");
    }
    imageBarrier.imageResource = j.at("image_resource"sv).get<std::string>();
    parseOptionalField(j, "subresource_range", imageBarrier.imageRange);
}

/**
 * @brief De-serialize Buffer Barrier from JSON.
 *
 * @param j
 * @param bufferBarrier
 */
void from_json(const json &j, BufferBarrierDesc &bufferBarrier) {
    parseBaseBarrierDesc(j, bufferBarrier);
    bufferBarrier.size = j.at("size"sv).get<uint64_t>();
    bufferBarrier.offset = j.at("offset"sv).get<uint64_t>();
    bufferBarrier.bufferResource = j.at("buffer_resource"sv).get<std::string>();
}

/**
 * @brief De-serialize SubresourceRange from JSON.
 *
 * @param j
 * @param subresourceRange
 */
void from_json(const json &j, SubresourceRange &subresourceRange) {
    subresourceRange.baseMipLevel = j.at("base_mip_level"sv).get<uint32_t>();
    subresourceRange.levelCount = j.at("level_count"sv).get<uint32_t>();
    subresourceRange.baseArrayLayer = j.at("base_array_layer"sv).get<uint32_t>();
    subresourceRange.layerCount = j.at("layer_count"sv).get<uint32_t>();
}

} // namespace mlsdk::scenariorunner
