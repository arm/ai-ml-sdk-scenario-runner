/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "json_reader.hpp"

#include "logging.hpp"
#include "scenario_desc.hpp"

#include <iostream>

namespace mlsdk::scenariorunner {
//==============
// Command details
// Function to de-serialize CommandDesc from JSON
void from_json(const json &j, CommandDesc &command);

// Function to de-serialize a PushConstantMap from JSON
void from_json(const json &j, PushConstantMap &pushConstantMap);

//==============
// Resource details
// Function to de-serialize a ResourceDesc from JSON
void from_json(const json &j, ResourceDesc &resource);

// Function to de-serialize a SpecializationConstant from JSON
void from_json(const json &j, SpecializationConstant &specializationConstant);

// Function to de-serialize a SpecializationConstantMap from JSON
void from_json(const json &j, SpecializationConstantMap &pushConstantMap);

// Function to de-serialize a ShaderSubstitution from JSON
void from_json(const json &j, ShaderSubstitution &shaderSubstitution);

// Function to de-serialize SubresourceRange from JSON
void from_json(const json &j, SubresourceRange &subresourceRange);

//==============
// Enums
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
        const json &item = resourceJson;
        ResourceDesc resource = item.get<ResourceDesc>();
        switch (resource.resourceType) {
        case (ResourceType::Shader): {
            ShaderDesc shader = item.at("shader").get<ShaderDesc>();
            scenarioSpec.addResource(std::make_unique<ShaderDesc>(shader));
        } break;
        case (ResourceType::Buffer): {
            BufferDesc buffer = item.at("buffer").get<BufferDesc>();
            scenarioSpec.addResource(std::make_unique<BufferDesc>(buffer));
        } break;
        case (ResourceType::RawData): {
            RawDataDesc raw_data = item.at("raw_data").get<RawDataDesc>();
            scenarioSpec.addResource(std::make_unique<RawDataDesc>(raw_data));
        } break;
        case (ResourceType::DataGraph): {
            DataGraphDesc dataGraph = item.at("graph").get<DataGraphDesc>();
            scenarioSpec.addResource(std::make_unique<DataGraphDesc>(dataGraph));
        } break;
        case (ResourceType::Tensor): {
            TensorDesc tensor = item.at("tensor").get<TensorDesc>();
            scenarioSpec.addResource(std::make_unique<TensorDesc>(tensor));
        } break;
        case (ResourceType::Image): {
            ImageDesc image = item.at("image").get<ImageDesc>();
            scenarioSpec.addResource(std::make_unique<ImageDesc>(image));
        } break;
        case (ResourceType::ImageBarrier): {
            ImageBarrierDesc imageBarrier = item.at("image_barrier").get<ImageBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<ImageBarrierDesc>(imageBarrier));
        } break;
        case (ResourceType::TensorBarrier): {
            TensorBarrierDesc tensorBarrier = item.at("tensor_barrier").get<TensorBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<TensorBarrierDesc>(tensorBarrier));
        } break;
        case (ResourceType::MemoryBarrier): {
            MemoryBarrierDesc memoryBarrier = item.at("memory_barrier").get<MemoryBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<MemoryBarrierDesc>(memoryBarrier));
        } break;
        case (ResourceType::BufferBarrier): {
            BufferBarrierDesc bufferBarrier = item.at("buffer_barrier").get<BufferBarrierDesc>();
            scenarioSpec.addResource(std::make_unique<BufferBarrierDesc>(bufferBarrier));
        } break;
        default:
            throw std::runtime_error("Unknown Resource type in resources");
        }
    }

    const json &commandsJson = j.at("commands");
    for (const auto &commandJson : commandsJson) {
        const json &item = commandJson;
        CommandDesc command = item.get<CommandDesc>();
        switch (command.commandType) {
        case (CommandType::DispatchCompute): {
            DispatchComputeDesc dispatchCompute = item.at("dispatch_compute").get<DispatchComputeDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchComputeDesc>(dispatchCompute));
        } break;
        case (CommandType::DispatchDataGraph): {
            DispatchDataGraphDesc dispatchDataGraph = item.at("dispatch_graph").get<DispatchDataGraphDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchDataGraphDesc>(dispatchDataGraph));
        } break;
        case (CommandType::DispatchBarrier): {
            DispatchBarrierDesc dispatchBarrier = item.at("dispatch_barrier").get<DispatchBarrierDesc>();
            scenarioSpec.addCommand(std::make_unique<DispatchBarrierDesc>(dispatchBarrier));
        } break;
        case (CommandType::MarkBoundary): {
            MarkBoundaryDesc markBoundary = item.at("mark_boundary").get<MarkBoundaryDesc>();
            scenarioSpec.addCommand(std::make_unique<MarkBoundaryDesc>(markBoundary));
        } break;
        default:
            throw std::runtime_error("Unknown Command type in commands");
        }
    }
}

/**
 * @brief De-serialize CommandDesc from JSON.
 *
 * @param j
 * @param command
 */
void from_json(const json &j, CommandDesc &command) {
    command.commandType = CommandType::Unknown;
    if (j.find("dispatch_compute") != j.end()) {
        command.commandType = CommandType::DispatchCompute;
    } else if (j.find("dispatch_graph") != j.end()) {
        command.commandType = CommandType::DispatchDataGraph;
    } else if (j.find("dispatch_barrier") != j.end()) {
        command.commandType = CommandType::DispatchBarrier;
    } else if (j.find("mark_boundary") != j.end()) {
        command.commandType = CommandType::MarkBoundary;
    } else {
        throw std::runtime_error("Unknown Command type");
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
 * @brief De-serialize ResourceDesc from JSON.
 *
 * @param j
 * @param resource
 */
void from_json(const json &j, ResourceDesc &resource) {
    resource.resourceType = ResourceType::Unknown;
    if (j.find("shader") != j.end()) {
        resource.resourceType = ResourceType::Shader;
    } else if (j.find("buffer") != j.end()) {
        resource.resourceType = ResourceType::Buffer;
    } else if (j.find("graph") != j.end()) {
        resource.resourceType = ResourceType::DataGraph;
    } else if (j.find("raw_data") != j.end()) {
        resource.resourceType = ResourceType::RawData;
    } else if (j.find("tensor") != j.end()) {
        resource.resourceType = ResourceType::Tensor;
    } else if (j.find("image") != j.end()) {
        resource.resourceType = ResourceType::Image;
    } else if (j.find("image_barrier") != j.end()) {
        resource.resourceType = ResourceType::ImageBarrier;
    } else if (j.find("memory_barrier") != j.end()) {
        resource.resourceType = ResourceType::MemoryBarrier;
    } else if (j.find("tensor_barrier") != j.end()) {
        resource.resourceType = ResourceType::TensorBarrier;
    } else if (j.find("buffer_barrier") != j.end()) {
        resource.resourceType = ResourceType::BufferBarrier;
    } else {
        throw std::runtime_error("Unknown Resource type");
    }
}

/**
 * @brief De-serialize BufferDesc from JSON.
 *
 * @param j
 * @param buffer
 */
void from_json(const json &j, BufferDesc &buffer) {
    buffer.guidStr = j.at("uid").get<std::string>();
    buffer.guid = buffer.guidStr;
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
    auto &val = j.at("value");
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
    dataGraph.guidStr = j.at("uid").get<std::string>();
    dataGraph.guid = dataGraph.guidStr;
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
    shader.guidStr = j.at("uid").get<std::string>();
    shader.guid = shader.guidStr;
    shader.src = j.at("src").get<std::string>();
    shader.shaderType = j.at("type").get<ShaderType>();
    if (shader.shaderType == ShaderType::Unknown) {
        throw std::runtime_error("Unknown shader type value");
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
 * @param raw_data
 */
void from_json(const json &j, RawDataDesc &raw_data) {
    raw_data.guidStr = j.at("uid").get<std::string>();
    raw_data.guid = raw_data.guidStr;
    raw_data.src = j.at("src").get<std::string>();
}

/**
 * @brief De-serialize TensorDesc from JSON.
 *
 * @param j
 * @param tensor
 */
void from_json(const json &j, TensorDesc &tensor) {
    tensor.guidStr = j.at("uid").get<std::string>();
    tensor.guid = tensor.guidStr;
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
    image.guidStr = j.at("uid").get<std::string>();
    image.guid = image.guidStr;
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
void from_json(const json &j, MemoryBarrierDesc &memoryBarrier) {
    memoryBarrier.guidStr = j.at("uid").get<std::string>();
    memoryBarrier.guid = memoryBarrier.guidStr;
    memoryBarrier.srcAccess = j.at("src_access").get<MemoryAccess>();
    if (memoryBarrier.srcAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown src_access value");
    }
    memoryBarrier.dstAccess = j.at("dst_access").get<MemoryAccess>();
    if (memoryBarrier.dstAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown dst_access value");
    }

    auto srcStagesIter = j.find("src_stage");
    if (srcStagesIter != j.end()) {
        const json &srcStagesJson = srcStagesIter.value();
        memoryBarrier.srcStages = srcStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : memoryBarrier.srcStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown src_stage value");
            }
        }
    }

    auto dstStagesIter = j.find("dst_stage");
    if (dstStagesIter != j.end()) {
        const json &dstStagesJson = dstStagesIter.value();
        memoryBarrier.dstStages = dstStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : memoryBarrier.dstStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown dst_stage value");
            }
        }
    }
}

/**
 * @brief De-serialize Tensor Barrier from JSON.
 *
 * @param j json object
 * @param tensorBarrier tensor barrier description struct
 */
void from_json(const json &j, TensorBarrierDesc &tensorBarrier) {
    tensorBarrier.guidStr = j.at("uid").get<std::string>();
    tensorBarrier.guid = tensorBarrier.guidStr;
    tensorBarrier.srcAccess = j.at("src_access").get<MemoryAccess>();
    if (tensorBarrier.srcAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown src_access value");
    }
    tensorBarrier.dstAccess = j.at("dst_access").get<MemoryAccess>();
    if (tensorBarrier.dstAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown dst_access value");
    }

    auto srcStagesIter = j.find("src_stage");
    if (srcStagesIter != j.end()) {
        const json &srcStagesJson = srcStagesIter.value();
        tensorBarrier.srcStages = srcStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : tensorBarrier.srcStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown src_stage value");
            }
        }
    }

    auto dstStagesIter = j.find("dst_stage");
    if (dstStagesIter != j.end()) {
        const json &dstStagesJson = dstStagesIter.value();
        tensorBarrier.dstStages = dstStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : tensorBarrier.dstStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown dst_stage value");
            }
        }
    }

    tensorBarrier.tensorResource = j.at("tensor_resource").get<std::string>();
}

/**
 * @brief De-serialize ImageBarrier from JSON.
 *
 * @param j
 * @param imageBarrier
 */
void from_json(const json &j, ImageBarrierDesc &imageBarrier) {
    imageBarrier.guidStr = j.at("uid").get<std::string>();
    imageBarrier.guid = imageBarrier.guidStr;
    imageBarrier.srcAccess = j.at("src_access").get<MemoryAccess>();
    if (imageBarrier.srcAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown src_access value");
    }
    imageBarrier.dstAccess = j.at("dst_access").get<MemoryAccess>();
    if (imageBarrier.dstAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown dst_access value");
    }
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

    auto srcStagesIter = j.find("src_stage");
    if (srcStagesIter != j.end()) {
        const json &srcStagesJson = srcStagesIter.value();
        imageBarrier.srcStages = srcStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : imageBarrier.srcStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown src_stage value");
            }
        }
    }

    auto dstStagesIter = j.find("dst_stage");
    if (dstStagesIter != j.end()) {
        const json &dstStagesJson = dstStagesIter.value();
        imageBarrier.dstStages = dstStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : imageBarrier.dstStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown dst_stage value");
            }
        }
    }
}

/**
 * @brief De-serialize Buffer Barrier from JSON.
 *
 * @param j
 * @param bufferBarrier
 */
void from_json(const json &j, BufferBarrierDesc &bufferBarrier) {
    bufferBarrier.guidStr = j.at("uid").get<std::string>();
    bufferBarrier.guid = bufferBarrier.guidStr;
    bufferBarrier.srcAccess = j.at("src_access").get<MemoryAccess>();
    if (bufferBarrier.srcAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown src_access value");
    }
    bufferBarrier.dstAccess = j.at("dst_access").get<MemoryAccess>();
    if (bufferBarrier.dstAccess == MemoryAccess::Unknown) {
        throw std::runtime_error("Unknown dst_access value");
    }
    bufferBarrier.size = j.at("size").get<uint64_t>();
    bufferBarrier.offset = j.at("offset").get<uint64_t>();

    auto srcStagesIter = j.find("src_stage");
    if (srcStagesIter != j.end()) {
        const json &srcStagesJson = srcStagesIter.value();
        bufferBarrier.srcStages = srcStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : bufferBarrier.srcStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown src_stage value");
            }
        }
    }

    auto dstStagesIter = j.find("dst_stage");
    if (dstStagesIter != j.end()) {
        const json &dstStagesJson = dstStagesIter.value();
        bufferBarrier.dstStages = dstStagesJson.get<std::vector<PipelineStage>>();
        for (PipelineStage it : bufferBarrier.dstStages) {
            if (it == PipelineStage::Unknown) {
                throw std::runtime_error("Unknown dst_stage value");
            }
        }
    }

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
