/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "json_reader.hpp"

#include "logging.hpp"
#include "scenario_desc.hpp"

#include <string_view>
#include <unordered_map>

namespace mlsdk::scenariorunner {
namespace {

const std::unordered_map<std::string_view, CommandType> &getCommandTypeByKey() {
    static const std::unordered_map<std::string_view, CommandType> commandTypeByKey = {
        {"dispatch_compute", CommandType::DispatchCompute},   {"dispatch_graph", CommandType::DispatchDataGraph},
        {"dispatch_fragment", CommandType::DispatchFragment}, {"dispatch_spirv_graph", CommandType::DispatchSpirvGraph},
        {"dispatch_barrier", CommandType::DispatchBarrier},   {"mark_boundary", CommandType::MarkBoundary},
    };

    return commandTypeByKey;
}

const std::unordered_map<std::string_view, ResourceType> &getResourceTypeByKey() {
    static const std::unordered_map<std::string_view, ResourceType> resourceTypeByKey = {
        {"shader", ResourceType::Shader},
        {"buffer", ResourceType::Buffer},
        {"graph", ResourceType::DataGraph},
        {"raw_data", ResourceType::RawData},
        {"tensor", ResourceType::Tensor},
        {"image", ResourceType::Image},
        {"image_barrier", ResourceType::ImageBarrier},
        {"memory_barrier", ResourceType::MemoryBarrier},
        {"tensor_barrier", ResourceType::TensorBarrier},
        {"buffer_barrier", ResourceType::BufferBarrier},
        {"graph_constant", ResourceType::GraphConstant},
    };

    return resourceTypeByKey;
}

template <typename EnumType>
EnumType parseSingleKeyObjectType(const json &j, const std::unordered_map<std::string_view, EnumType> &typeByKey,
                                  std::string_view entryName) {
    if (!j.is_object()) {
        throw std::runtime_error(std::string(entryName) + " entry must be a JSON object");
    }

    if (j.size() != 1) {
        throw std::runtime_error(std::string(entryName) + " entry must contain exactly one top-level key");
    }

    const auto key = std::string_view(j.begin().key());
    const auto it = typeByKey.find(key);
    if (it == typeByKey.end()) {
        throw std::runtime_error(std::string("Unknown ") + std::string(entryName) + " type");
    }

    return it->second;
}

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
    resource.guidStr = j.at("uid").get<std::string>();
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

void readJson(ScenarioSpec &scenarioSpec, std::istream *is) {
    json j;
    *is >> j;

    const json &resourcesJson = j.at("resources");
    for (const auto &resourceJson : resourcesJson) {
        const auto resourceType = parseSingleKeyObjectType(resourceJson, getResourceTypeByKey(), "Resource");
        switch (resourceType) {
        case (ResourceType::Shader): {
            auto shader = resourceJson.at("shader").get<ShaderDesc>();
            scenarioSpec.addResource(std::make_unique<ShaderDesc>(std::move(shader)));
        } break;
        case (ResourceType::Buffer): {
            auto buffer = resourceJson.at("buffer").get<BufferDesc>();
            scenarioSpec.addResource(std::make_unique<BufferDesc>(std::move(buffer)));
        } break;
        case (ResourceType::RawData): {
            auto rawData = resourceJson.at("raw_data").get<RawDataDesc>();
            scenarioSpec.addResource(std::make_unique<RawDataDesc>(std::move(rawData)));
        } break;
        case (ResourceType::DataGraph): {
            auto dataGraph = resourceJson.at("graph").get<DataGraphDesc>();
            scenarioSpec.addResource(std::make_unique<DataGraphDesc>(std::move(dataGraph)));
        } break;
        case (ResourceType::Tensor): {
            auto tensor = resourceJson.at("tensor").get<TensorDesc>();
            scenarioSpec.addResource(std::make_unique<TensorDesc>(std::move(tensor)));
        } break;
        case (ResourceType::Image): {
            auto image = resourceJson.at("image").get<ImageDesc>();
            scenarioSpec.addResource(std::make_unique<ImageDesc>(std::move(image)));
        } break;
        case (ResourceType::ImageBarrier): {
            auto imageBarrier = resourceJson.at("image_barrier").get<ImageBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<ImageBarrierDesc>(std::move(imageBarrier)));
        } break;
        case (ResourceType::TensorBarrier): {
            auto tensorBarrier = resourceJson.at("tensor_barrier").get<TensorBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<TensorBarrierDesc>(std::move(tensorBarrier)));
        } break;
        case (ResourceType::MemoryBarrier): {
            auto memoryBarrier = resourceJson.at("memory_barrier").get<MemoryBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<MemoryBarrierDesc>(std::move(memoryBarrier)));
        } break;
        case (ResourceType::BufferBarrier): {
            auto bufferBarrier = resourceJson.at("buffer_barrier").get<BufferBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<BufferBarrierDesc>(std::move(bufferBarrier)));
        } break;
        case (ResourceType::GraphConstant): {
            auto graphConstant = resourceJson.at("graph_constant").get<GraphConstantDesc>();
            scenarioSpec.addResource(std::make_unique<GraphConstantDesc>(std::move(graphConstant)));
        } break;
        default:
            throw std::runtime_error("Unknown Resource type in resources");
        }
    }

    const json &commandsJson = j.at("commands");
    for (const auto &commandJson : commandsJson) {
        const auto commandType = parseSingleKeyObjectType(commandJson, getCommandTypeByKey(), "Command");
        switch (commandType) {
        case (CommandType::DispatchCompute): {
            auto dispatchCompute = commandJson.at("dispatch_compute").get<DispatchComputeDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchComputeDesc>(std::move(dispatchCompute)));
        } break;
        case (CommandType::DispatchDataGraph): {
            auto dispatchDataGraph = commandJson.at("dispatch_graph").get<DispatchDataGraphDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchDataGraphDesc>(std::move(dispatchDataGraph)));
        } break;
        case (CommandType::DispatchSpirvGraph): {
            auto dispatchSpirvGraph = commandJson.at("dispatch_spirv_graph").get<DispatchSpirvGraphDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchSpirvGraphDesc>(std::move(dispatchSpirvGraph)));
        } break;
        case (CommandType::DispatchFragment): {
            auto dispatchFragment = commandJson.at("dispatch_fragment").get<DispatchFragmentDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchFragmentDesc>(std::move(dispatchFragment)));
        } break;
        case (CommandType::DispatchBarrier): {
            auto dispatchBarrier = commandJson.at("dispatch_barrier").get<DispatchBarrierDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchBarrierDesc>(std::move(dispatchBarrier)));
        } break;
        case (CommandType::MarkBoundary): {
            auto markBoundary = commandJson.at("mark_boundary").get<MarkBoundaryDesc>();
            scenarioSpec.addCommand(std::make_unique<MarkBoundaryDesc>(std::move(markBoundary)));
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
    dispatchCompute.bindings = j.at("bindings").get<std::vector<BindingDesc>>();
    const json &rangeNDJson = j.at("rangeND");
    uint32_t i = 0;
    for (auto dimension = rangeNDJson.begin(); dimension != rangeNDJson.end(); ++dimension) {
        const json &newDimension = dimension.value();
        dispatchCompute.rangeND.push_back(newDimension.get<uint32_t>());
        ++i;
    }
    for (; i < 3; ++i) {
        dispatchCompute.rangeND.push_back(1);
    }
    auto shaderRef = j.at("shader_ref").get<std::string>();
    dispatchCompute.shaderRef = shaderRef;
    dispatchCompute.debugName = shaderRef;
    if (j.contains("push_data_ref")) {
        dispatchCompute.pushDataRef = j.at("push_data_ref").get<std::string>();
    }
    if (j.contains("implicit_barrier")) {
        dispatchCompute.implicitBarrier = j.at("implicit_barrier").get<bool>();
    }
}

void from_json(const json &j, FragmentAttachmentDesc &fragmentAttachment) {
    if (j.is_string()) {
        fragmentAttachment.resourceRef = j.get<std::string>();
    } else if (j.is_object()) {
        fragmentAttachment.resourceRef = j.at("resource_ref").get<std::string>();
        if (j.contains("lod")) {
            fragmentAttachment.lod = j.at("lod").get<uint32_t>();
        }
    } else {
        throw std::runtime_error("color_attachment_refs entries must be strings or objects");
    }
}

void from_json(const json &j, DispatchFragmentDesc &dispatchFragment) {
    dispatchFragment.bindings = j.at("bindings").get<std::vector<BindingDesc>>();

    dispatchFragment.vertexShaderRef = j.at("vertex_shader_ref").get<std::string>();
    const auto fragmentShaderRef = j.at("fragment_shader_ref").get<std::string>();
    dispatchFragment.fragmentShaderRef = fragmentShaderRef;
    dispatchFragment.debugName = fragmentShaderRef;
    if (j.contains("debug_name")) {
        dispatchFragment.debugName = j.at("debug_name").get<std::string>();
    }
    if (j.contains("color_attachment_refs")) {
        dispatchFragment.colorAttachments = j.at("color_attachment_refs").get<std::vector<FragmentAttachmentDesc>>();
    }
    if (j.contains("render_extent")) {
        dispatchFragment.renderExtent = j.at("render_extent").get<std::array<uint32_t, 2>>();
    }
    if (dispatchFragment.colorAttachments.empty() && !dispatchFragment.renderExtent.has_value()) {
        throw std::runtime_error("dispatch_fragment requires color_attachment_refs or render_extent");
    }
    if (j.contains("push_data_ref")) {
        dispatchFragment.pushDataRef = j.at("push_data_ref").get<std::string>();
    }
    if (j.contains("implicit_barrier")) {
        dispatchFragment.implicitBarrier = j.at("implicit_barrier").get<bool>();
    }
}

/**
 * @brief De-serialize DispatchDataGraphDesc from JSON.
 *
 * @param j
 * @param dispatchDataGraph
 */
void from_json(const json &j, DispatchDataGraphDesc &dispatchDataGraph) {
    auto graphRef = j.at("graph_ref").get<std::string>();
    dispatchDataGraph.dataGraphRef = graphRef;
    dispatchDataGraph.debugName = graphRef;
    dispatchDataGraph.bindings = j.at("bindings").get<std::vector<BindingDesc>>();

    if (j.contains("push_constants")) {
        dispatchDataGraph.pushConstants = j.at("push_constants").get<std::vector<PushConstantMap>>();
    }
    if (j.contains("shader_substitutions")) {
        dispatchDataGraph.shaderSubstitutions = j.at("shader_substitutions").get<std::vector<ShaderSubstitution>>();
    }
    if (j.contains("implicit_barrier")) {
        dispatchDataGraph.implicitBarrier = j.at("implicit_barrier").get<bool>();
    }
}

/**
 * @brief De-serialize DispatchSpirvGraphDesc from JSON.
 *
 * @param j
 * @param dispatchSpirvGraph
 */
void from_json(const json &j, DispatchSpirvGraphDesc &dispatchSpirvGraph) {
    auto graphRef = j.at("graph_ref").get<std::string>();
    dispatchSpirvGraph.dataGraphRef = graphRef;
    dispatchSpirvGraph.debugName = graphRef;
    dispatchSpirvGraph.bindings = j.at("bindings").get<std::vector<BindingDesc>>();
    if (j.contains("graph_constants")) {
        const auto graphConstants = j.at("graph_constants").get<std::vector<std::string>>();
        dispatchSpirvGraph.graphConstants.assign(graphConstants.begin(), graphConstants.end());
    }
    if (j.contains("implicit_barrier")) {
        dispatchSpirvGraph.implicitBarrier = j.at("implicit_barrier").get<bool>();
    }
}

/**
 * @brief De-serialize DispatchBarrierDesc from JSON.
 *
 * @param j
 * @param dispatchBarrier
 */
void from_json(const json &j, DispatchBarrierDesc &dispatchBarrier) {
    dispatchBarrier.imageBarriersRef = j.at("image_barrier_refs").get<std::vector<std::string>>();

    auto tensorBarriersIter = j.find("tensor_barrier_refs");
    if (tensorBarriersIter != j.end()) {
        const json &tensorBarriersJson = tensorBarriersIter.value();
        for (auto tensorBarrier = tensorBarriersJson.begin(); tensorBarrier != tensorBarriersJson.end();
             ++tensorBarrier) {
            const json &newTensorBarrier = tensorBarrier.value();
            dispatchBarrier.tensorBarriersRef.push_back(newTensorBarrier.get<std::string>());
        }
    }

    dispatchBarrier.memoryBarriersRef = j.at("memory_barrier_refs").get<std::vector<std::string>>();

    dispatchBarrier.bufferBarriersRef = j.at("buffer_barrier_refs").get<std::vector<std::string>>();
}

/**
 * @brief De-serialize MarkBoundary from JSON.
 *
 * @param j
 * @param markBoundaryDesc
 */
void from_json(const json &j, MarkBoundaryDesc &markBoundaryDesc) {
    if (j.contains("frame_id")) {
        mlsdk::logging::warning("Manual setting of frameID is deprecated");
    }
    markBoundaryDesc.resources = j.at("resources").get<std::vector<std::string>>();
}

/**
 * @brief De-serialize BindingDesc from JSON.
 *
 * @param j
 * @param binding
 */
void from_json(const json &j, BindingDesc &binding) {
    binding.set = j.at("set").get<uint32_t>();
    binding.id = j.at("id").get<uint32_t>();
    if (j.contains("lod")) {
        binding.lod = j.at("lod").get<int>();
    }
    binding.resourceRef = j.at("resource_ref").get<std::string>();
    if (j.contains("descriptor_type")) {
        binding.descriptorType = j.at("descriptor_type").get<DescriptorType>();
        if (binding.descriptorType == DescriptorType::Unknown) {
            throw std::runtime_error("Unknown descriptor_type value");
        }
    }
}

/**
 * @brief De-serialize PushConstantMap from JSON.
 *
 * @param j
 * @param pushConstantMap
 */
void from_json(const json &j, PushConstantMap &pushConstantMap) {
    pushConstantMap.pushDataRef = j.at("push_data_ref").get<std::string>();
    pushConstantMap.shaderTarget = j.at("shader_target").get<std::string>();
}

/**
 * @brief De-serialize MemoryGroup from JSON.
 *
 * @param j
 * @param group
 */
void from_json(const json &j, MemoryGroup &group) {
    group.memoryUid = j.at("id");
    if (j.contains("offset")) {
        group.offset = j.at("offset").get<uint64_t>();
    }
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
    buffer.size = j.at("size").get<uint32_t>();
    buffer.shaderAccess = j.at("shader_access").get<ShaderAccessType>();
    if (buffer.shaderAccess == ShaderAccessType::Unknown) {
        throw std::runtime_error("Unknown shader_access value");
    }
    if (j.contains("src")) {
        buffer.src = j.at("src").get<std::string>();
    }
    if (j.contains("dst")) {
        buffer.dst = j.at("dst").get<std::string>();
    }
    if (j.contains("memory_group")) {
        buffer.memoryGroup = j.at("memory_group").get<MemoryGroup>();
    }
}

/**
 * @brief De-serialize SpecializationConstant from JSON.
 *
 * @param j
 * @param specializationConstant
 */
void from_json(const json &j, SpecializationConstant &specializationConstant) {
    specializationConstant.id = j.at("id").get<int>();
    const auto &val = j.at("value");
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
    specializationConstantMap.shaderTarget = j.at("shader_target").get<std::string>();
    specializationConstantMap.specializationConstants =
        j.at("specialization_constants").get<std::vector<SpecializationConstant>>();
}

/**
 * @brief De-serialize ShaderSubstitution from JSON.
 *
 * @param j
 * @param shaderSubstitution
 */
void from_json(const json &j, ShaderSubstitution &shaderSubstitution) {
    shaderSubstitution.shaderRef = j.at("shader_ref").get<std::string>();
    shaderSubstitution.target = j.at("target").get<std::string>();
}

/**
 * @brief De-serialize GraphDesc from JSON.
 *
 * @param j
 * @param dataGraph
 */
void from_json(const json &j, DataGraphDesc &dataGraph) {
    parseResourceDescGuid(j, dataGraph);
    dataGraph.src = j.at("src").get<std::string>();
    if (j.contains("shader_substitutions")) {
        dataGraph.shaderSubstitutions = j.at("shader_substitutions").get<std::vector<ShaderSubstitution>>();
    }
    if (j.contains("specialization_constants_map")) {
        dataGraph.specializationConstantMaps =
            j.at("specialization_constants_map").get<std::vector<SpecializationConstantMap>>();
    }
    if (j.contains("push_constants_size")) {
        dataGraph.pushConstantsSize = j.at("push_constants_size").get<uint32_t>();
    }
}

/**
 * @brief De-serialize ShaderDesc from JSON.
 *
 * @param j
 * @param shader
 */
void from_json(const json &j, ShaderDesc &shader) {
    parseResourceDescGuid(j, shader);
    shader.src = j.at("src").get<std::string>();
    shader.shaderType = j.at("type").get<ShaderType>();
    if (shader.shaderType == ShaderType::Unknown) {
        throw std::runtime_error("Unknown shader type value");
    }
    shader.stage = ShaderStage::Compute;
    if (j.contains("stage")) {
        shader.stage = j.at("stage").get<ShaderStage>();
        if (shader.stage == ShaderStage::Unknown) {
            throw std::runtime_error("Unknown shader stage value");
        }
    }
    if (j.contains("entry")) {
        shader.entry = j.at("entry").get<std::string>();
    }
    if (shader.shaderType == ShaderType::GLSL && shader.entry != "main") {
        throw std::runtime_error("GLSL is required to have an entrypoint of 'main'");
    }
    if (j.contains("push_constants_size")) {
        shader.pushConstantsSize = j.at("push_constants_size").get<uint32_t>();
    }
    if (j.contains("specialization_constants")) {
        shader.specializationConstants = j.at("specialization_constants").get<std::vector<SpecializationConstant>>();
    }
    if (j.contains("build_options")) {
        shader.buildOpts = j.at("build_options").get<std::string>();
    }
    if (j.contains("include_dirs")) {
        shader.includeDirs = j.at("include_dirs").get<std::vector<std::string>>();
    }
}

/**
 * @brief De-serialize RawDataDesc from JSON.
 *
 * @param j
 * @param rawData
 */
void from_json(const json &j, RawDataDesc &rawData) {
    parseResourceDescGuid(j, rawData);
    rawData.src = j.at("src").get<std::string>();
}

/**
 * @brief De-serialize GraphConstantDesc from JSON.
 *
 * @param j
 * @param graphConstant
 */
void from_json(const json &j, GraphConstantDesc &graphConstant) {
    parseResourceDescGuid(j, graphConstant);
    const auto dims = j.at("dims").get<std::vector<int64_t>>();
    if (dims.empty() || dims.size() > 6) {
        throw std::runtime_error("Invalid graph constant dimensions");
    }
    graphConstant.dims = dims;
    graphConstant.src = j.at("src").get<std::string>();
    graphConstant.format = j.at("format").get<std::string>();
}

/**
 * @brief De-serialize TensorDesc from JSON.
 *
 * @param j
 * @param tensor
 */
void from_json(const json &j, TensorDesc &tensor) {
    parseResourceDescGuid(j, tensor);
    tensor.dims = j.at("dims").get<std::vector<int64_t>>();
    tensor.format = j.at("format").get<std::string>();
    tensor.shaderAccess = j.at("shader_access").get<ShaderAccessType>();
    if (tensor.shaderAccess == ShaderAccessType::Unknown) {
        throw std::runtime_error("Unknown shader_access type");
    }
    if (j.contains("src")) {
        tensor.src = j.at("src").get<std::string>();
    }
    if (j.contains("dst")) {
        tensor.dst = j.at("dst").get<std::string>();
    }
    if (j.contains("alias_target")) {
        throw std::runtime_error(
            "The alias_target field is deprecated, please use the memory_group functionality instead");
    }
    if (j.contains("memory_group")) {
        tensor.memoryGroup = j.at("memory_group").get<MemoryGroup>();
    }
    if (j.contains("tiling")) {
        tensor.tiling = j.at("tiling").get<scenariorunner::Tiling>();
        if (tensor.tiling == Tiling::Unknown) {
            throw std::runtime_error("Unknown tiling value");
        }
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
    image.dims = j.at("dims").get<std::vector<uint32_t>>();
    // for compatibility with the old json configs that had this field set as "false"
    if (!j.contains("mips")) {
        image.mips = 1;
    } else if (j.at("mips").is_boolean()) {
        mlsdk::logging::warning("Boolean mips flag is deprecated, defaulting to \"1\". Use integer value instead");
        image.mips = 1;
    } else {
        image.mips = j.at("mips").get<uint32_t>();
    }

    image.format = j.at("format").get<std::string>();
    image.shaderAccess = j.at("shader_access").get<ShaderAccessType>();
    if (image.shaderAccess == ShaderAccessType::Unknown) {
        throw std::runtime_error("Unknown shader_access type");
    }
    if (j.contains("src")) {
        image.src = j.at("src").get<std::string>();
    }
    if (j.contains("dst")) {
        image.dst = j.at("dst").get<std::string>();
    }
    if (j.contains("color_attachment")) {
        image.colorAttachment = j.at("color_attachment").get<bool>();
    }
    if (j.contains("min_filter")) {
        image.minFilter = j.at("min_filter").get<FilterMode>();
        if (image.minFilter == FilterMode::Unknown) {
            throw std::runtime_error("Unknown min_filter value");
        }
    }
    if (j.contains("mag_filter")) {
        image.magFilter = j.at("mag_filter").get<FilterMode>();
        if (image.magFilter == FilterMode::Unknown) {
            throw std::runtime_error("Unknown mag_filter value");
        }
    }
    if (j.contains("mip_filter")) {
        image.mipFilter = j.at("mip_filter").get<FilterMode>();
        if (image.mipFilter == FilterMode::Unknown) {
            throw std::runtime_error("Unknown mip_filter value");
        }
    }
    if (j.contains("border_address_mode")) {
        image.borderAddressMode = j.at("border_address_mode").get<AddressMode>();
        if (image.borderAddressMode == AddressMode::Unknown) {
            throw std::runtime_error("Unknown border_address_mode value");
        }
    }
    if (j.contains("border_color")) {
        image.borderColor = j.at("border_color").get<BorderColor>();
        if (image.borderColor == BorderColor::Unknown) {
            throw std::runtime_error("Unknown border_color value");
        }
    }
    if (j.contains("custom_border_color")) {
        if (!image.borderColor.has_value()) {
            throw std::runtime_error("custom_border_color requires border_color");
        }

        const json &customColorJson = j.at("custom_border_color");
        if (image.borderColor.value() == BorderColor::FloatCustomEXT) {
            image.customBorderColor = customColorJson.get<std::array<float, 4>>();
        } else {
            image.customBorderColor = customColorJson.get<std::array<int, 4>>();
        }
    }
    if (j.contains("tiling")) {
        image.tiling = j.at("tiling").get<scenariorunner::Tiling>();
        if (image.tiling == Tiling::Unknown) {
            throw std::runtime_error("Unknown tiling value");
        }
    }
    if (j.contains("memory_group")) {
        image.memoryGroup = j.at("memory_group").get<MemoryGroup>();
    }
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

    tensorBarrier.tensorResource = j.at("tensor_resource").get<std::string>();
}

/**
 * @brief De-serialize ImageBarrier from JSON.
 *
 * @param j
 * @param imageBarrier
 */
void from_json(const json &j, ImageBarrierDesc &imageBarrier) {
    parseBaseBarrierDesc(j, imageBarrier);
    imageBarrier.oldLayout = j.at("old_layout").get<ImageLayout>();
    if (imageBarrier.oldLayout == ImageLayout::Unknown) {
        throw std::runtime_error("Unknown old_layout value");
    }
    imageBarrier.newLayout = j.at("new_layout").get<ImageLayout>();
    if (imageBarrier.newLayout == ImageLayout::Unknown) {
        throw std::runtime_error("Unknown new_layout value");
    }
    imageBarrier.imageResource = j.at("image_resource").get<std::string>();
    if (j.contains("subresource_range")) {
        imageBarrier.imageRange = j.at("subresource_range").get<SubresourceRange>();
    }
}

/**
 * @brief De-serialize Buffer Barrier from JSON.
 *
 * @param j
 * @param bufferBarrier
 */
void from_json(const json &j, BufferBarrierDesc &bufferBarrier) {
    parseBaseBarrierDesc(j, bufferBarrier);
    bufferBarrier.size = j.at("size").get<uint64_t>();
    bufferBarrier.offset = j.at("offset").get<uint64_t>();
    bufferBarrier.bufferResource = j.at("buffer_resource").get<std::string>();
}

/**
 * @brief De-serialize SubresourceRange from JSON.
 *
 * @param j
 * @param subresourceRange
 */
void from_json(const json &j, SubresourceRange &subresourceRange) {
    subresourceRange.baseMipLevel = j.at("base_mip_level").get<uint32_t>();
    subresourceRange.levelCount = j.at("level_count").get<uint32_t>();
    subresourceRange.baseArrayLayer = j.at("base_array_layer").get<uint32_t>();
    subresourceRange.layerCount = j.at("layer_count").get<uint32_t>();
}

} // namespace mlsdk::scenariorunner
