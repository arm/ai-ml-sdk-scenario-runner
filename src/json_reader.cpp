/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

// Function to de-serialize a ShaderSubstitutionDesc from JSON
void from_json(const json &j, ShaderSubstitutionDesc &shaderSubstitution);

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

    auto findResources = j.find("resources");
    if (findResources != j.end()) {
        const json &resourcesJson = findResources.value();
        for (auto resourceJson = resourcesJson.begin(); resourceJson != resourcesJson.end(); ++resourceJson) {
            ResourceDesc resource = resourceJson.value().get<ResourceDesc>();
            switch (resource.resourceType) {
            case (ResourceType::Shader): {
                ShaderDesc shader = resourceJson.value().find("shader").value().get<ShaderDesc>();
                scenarioSpec.addResource(std::make_unique<ShaderDesc>(shader));
            } break;
            case (ResourceType::Buffer): {
                BufferDesc buffer = resourceJson.value().find("buffer").value().get<BufferDesc>();
                scenarioSpec.addResource(std::make_unique<BufferDesc>(buffer));
            } break;
            case (ResourceType::RawData): {
                RawDataDesc raw_data = resourceJson.value().find("raw_data").value().get<RawDataDesc>();
                scenarioSpec.addResource(std::make_unique<RawDataDesc>(raw_data));
            } break;
            case (ResourceType::DataGraph): {
                DataGraphDesc dataGraph = resourceJson.value().find("graph").value().get<DataGraphDesc>();
                scenarioSpec.addResource(std::make_unique<DataGraphDesc>(dataGraph));
            } break;
            case (ResourceType::Tensor): {
                TensorDesc tensor = resourceJson.value().find("tensor").value().get<TensorDesc>();
                scenarioSpec.addResource(std::make_unique<TensorDesc>(tensor));
            } break;
            case (ResourceType::Image): {
                ImageDesc image = resourceJson.value().find("image").value().get<ImageDesc>();
                scenarioSpec.addResource(std::make_unique<ImageDesc>(image));
            } break;
            case (ResourceType::ImageBarrier): {
                ImageBarrierDesc imageBarrier =
                    resourceJson.value().find("image_barrier").value().get<ImageBarrierDesc>();
                scenarioSpec.addResource(std::make_unique<ImageBarrierDesc>(imageBarrier));
            } break;
            case (ResourceType::TensorBarrier): {
                TensorBarrierDesc tensorBarrier =
                    resourceJson.value().find("tensor_barrier").value().get<TensorBarrierDesc>();
                scenarioSpec.addResource(std::make_unique<TensorBarrierDesc>(tensorBarrier));
            } break;
            case (ResourceType::MemoryBarrier): {
                MemoryBarrierDesc memoryBarrier =
                    resourceJson.value().find("memory_barrier").value().get<MemoryBarrierDesc>();
                scenarioSpec.addResource(std::make_unique<MemoryBarrierDesc>(memoryBarrier));
            } break;
            case (ResourceType::BufferBarrier): {
                BufferBarrierDesc bufferBarrier =
                    resourceJson.value().find("buffer_barrier").value().get<BufferBarrierDesc>();
                scenarioSpec.addResource(std::make_unique<BufferBarrierDesc>(bufferBarrier));
            } break;
            default:
                throw std::runtime_error("Unknown Resource type in resources");
            }
        }
    }

    auto findCommands = j.find("commands");
    if (findCommands != j.end()) {
        const json &commandsJson = findCommands.value();
        for (auto commandJson = commandsJson.begin(); commandJson != commandsJson.end(); ++commandJson) {
            CommandDesc command = commandJson.value().get<CommandDesc>();
            switch (command.commandType) {
            case (CommandType::DispatchCompute): {
                DispatchComputeDesc dispatchCompute =
                    commandJson.value().find("dispatch_compute").value().get<DispatchComputeDesc>();
                scenarioSpec.addCommand(std::make_unique<DispatchComputeDesc>(dispatchCompute));
            } break;
            case (CommandType::DispatchDataGraph): {
                DispatchDataGraphDesc dispatchDataGraph =
                    commandJson.value().find("dispatch_graph").value().get<DispatchDataGraphDesc>();
                scenarioSpec.addCommand(std::make_unique<DispatchDataGraphDesc>(dispatchDataGraph));
            } break;
            case (CommandType::DispatchBarrier): {
                DispatchBarrierDesc dispatchBarrier =
                    commandJson.value().find("dispatch_barrier").value().get<DispatchBarrierDesc>();
                scenarioSpec.addCommand(std::make_unique<DispatchBarrierDesc>(dispatchBarrier));
            } break;
            case (CommandType::MarkBoundary): {
                MarkBoundaryDesc markBoundary =
                    commandJson.value().find("mark_boundary").value().get<MarkBoundaryDesc>();
                scenarioSpec.addCommand(std::make_unique<MarkBoundaryDesc>(markBoundary));
            } break;
            default:
                throw std::runtime_error("Unknown Command type in commands");
            }
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
    const json bindingsJson = j.find("bindings").value();
    for (auto binding = bindingsJson.begin(); binding != bindingsJson.end(); ++binding) {
        const json &newBinding = binding.value();
        dispatchCompute.bindings.push_back(newBinding.get<BindingDesc>());
    }
    const json rangeNDJson = j.find("rangeND").value();
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
    if (j.count("push_data_ref") != 0) {
        dispatchCompute.pushDataRef = j.at("push_data_ref").get<std::string>();
    }
    if (j.count("implicit_barrier") != 0) {
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
    const json bindingsJson = j.find("bindings").value();
    for (auto binding = bindingsJson.begin(); binding != bindingsJson.end(); ++binding) {
        const json &newBinding = binding.value();
        dispatchDataGraph.bindings.push_back(newBinding.get<BindingDesc>());
    }
    if (j.count("push_constants") != 0) {
        const json pushConstantsJson = j.find("push_constants").value();
        for (auto pushConstant = pushConstantsJson.begin(); pushConstant != pushConstantsJson.end(); ++pushConstant) {
            const json &newPushConstant = pushConstant.value();
            dispatchDataGraph.pushConstants.push_back(newPushConstant.get<PushConstantMap>());
        }
    }
    if (j.count("shader_substitutions") != 0) {
        const json shaderSubJson = j.find("shader_substitutions").value();
        for (auto shaderSub = shaderSubJson.begin(); shaderSub != shaderSubJson.end(); ++shaderSub) {
            const json &newShaderSub = shaderSub.value();
            dispatchDataGraph.shaderSubstitutions.push_back(newShaderSub.get<ShaderSubstitutionDesc>());
        }
    }
}

/**
 * @brief De-serialize DispatchBarrierDesc from JSON.
 *
 * @param j
 * @param dispatchBarrier
 */
void from_json(const json &j, DispatchBarrierDesc &dispatchBarrier) {
    const json imageBarriersJson = j.find("image_barrier_refs").value();
    for (auto imageBarrier = imageBarriersJson.begin(); imageBarrier != imageBarriersJson.end(); ++imageBarrier) {
        const json &newImageBarrier = imageBarrier.value();
        dispatchBarrier.imageBarriersRef.push_back(newImageBarrier.get<std::string>());
    }

    auto tensorBarriersIter = j.find("tensor_barrier_refs");
    if (tensorBarriersIter != j.end()) {
        const json &tensorBarriersJson = tensorBarriersIter.value();
        for (auto tensorBarrier = tensorBarriersJson.begin(); tensorBarrier != tensorBarriersJson.end();
             ++tensorBarrier) {
            const json &newTensorBarrier = tensorBarrier.value();
            dispatchBarrier.tensorBarriersRef.push_back(newTensorBarrier.get<std::string>());
        }
    }

    const json memoryBarriersJson = j.find("memory_barrier_refs").value();
    for (auto memoryBarrier = memoryBarriersJson.begin(); memoryBarrier != memoryBarriersJson.end(); ++memoryBarrier) {
        const json &newMemoryBarrier = memoryBarrier.value();
        dispatchBarrier.memoryBarriersRef.push_back(newMemoryBarrier.get<std::string>());
    }
    const json bufferBarriersJson = j.find("buffer_barrier_refs").value();
    for (auto bufferBarrier = bufferBarriersJson.begin(); bufferBarrier != bufferBarriersJson.end(); ++bufferBarrier) {
        const json &newBufferBarrier = bufferBarrier.value();
        dispatchBarrier.bufferBarriersRef.push_back(newBufferBarrier.get<std::string>());
    }
}

/**
 * @brief De-serialize MarkBoundary from JSON.
 *
 * @param j
 * @param markBoundaryDesc
 */
void from_json(const json &j, MarkBoundaryDesc &markBoundaryDesc) {
    try {
        markBoundaryDesc.frameId = j.at("frame_id").get<uint64_t>();
    } catch (...) {
        mlsdk::logging::warning("\"frame_id\" should be of type uint64");
        try {
            mlsdk::logging::warning("Attempting to parse \"frame_id\" as a string");
            uint64_t frameIdInt = std::stoull(j.at("frame_id").get<std::string>());
            mlsdk::logging::warning("String parsed successfully, \"frame_id\" set to: " + std::to_string(frameIdInt));
            markBoundaryDesc.frameId = frameIdInt;
        } catch (...) {
            throw std::runtime_error("Unable to parse \"frame_id\" as a string");
        }
    }

    const json resourcesJson = j.find("resources").value();
    for (auto resource = resourcesJson.begin(); resource != resourcesJson.end(); ++resource) {
        const json &newResource = resource.value();
        markBoundaryDesc.resources.push_back(newResource.get<std::string>());
    }
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
    if (j.count("lod") != 0) {
        binding.lod = j.at("lod").get<int>();
    }
    binding.resourceRef = j.at("resource_ref").get<std::string>();
    if (j.count("descriptor_type") != 0) {
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
    if (j.count("offset") != 0) {
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
    if (j.count("src") != 0) {
        buffer.src = j.at("src").get<std::string>();
    }
    if (j.count("dst") != 0) {
        buffer.dst = j.at("dst").get<std::string>();
    }
    if (j.count("memory_group") != 0) {
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
    const json constantsJson = j.find("specialization_constants").value();
    for (auto constant = constantsJson.begin(); constant != constantsJson.end(); ++constant) {
        const json &newConstant = constant.value();
        specializationConstantMap.specializationConstants.push_back(newConstant.get<SpecializationConstant>());
    }
}

/**
 * @brief De-serialize ShaderSubstitutionDesc from JSON.
 *
 * @param j
 * @param shaderSubstitution
 */
void from_json(const json &j, ShaderSubstitutionDesc &shaderSubstitution) {
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

    if (j.count("shader_substitutions") != 0) {
        const json shaderSubJson = j.find("shader_substitutions").value();
        for (auto shaderSub = shaderSubJson.begin(); shaderSub != shaderSubJson.end(); ++shaderSub) {
            const json &newShaderSub = shaderSub.value();
            dataGraph.shaderSubstitutions.push_back(newShaderSub.get<ShaderSubstitutionDesc>());
        }
    }
    if (j.count("specialization_constants") != 0) {
        const json conMapsJson = j.find("specialization_constants").value();
        for (auto conMap = conMapsJson.begin(); conMap != conMapsJson.end(); ++conMap) {
            const json &newConMap = conMap.value();
            dataGraph.specializationConstantMaps.push_back(newConMap.get<SpecializationConstantMap>());
        }
    }
    if (j.count("push_constants_size") != 0) {
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
    shader.entry = j.at("entry").get<std::string>();
    if (j.count("push_constants_size") != 0) {
        shader.pushConstantsSize = j.at("push_constants_size").get<uint32_t>();
    }
    if (j.count("specialization_constants") != 0) {
        const json specsJson = j.find("specialization_constants").value();
        for (auto spec = specsJson.begin(); spec != specsJson.end(); ++spec) {
            const json &newSpec = spec.value();
            shader.specializationConstants.push_back(newSpec.get<SpecializationConstant>());
        }
    }
    if (j.count("build_options") != 0) {
        shader.buildOpts = j.at("build_options").get<std::string>();
    }
    if (j.count("include_dirs") != 0) {
        const json includesJson = j.find("include_dirs").value();
        for (auto include = includesJson.begin(); include != includesJson.end(); ++include) {
            const json &newInclude = include.value();
            shader.includeDirs.push_back(newInclude.get<std::string>());
        }
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
    const json dimsJson = j.find("dims").value();
    for (auto dim = dimsJson.begin(); dim != dimsJson.end(); ++dim) {
        const json &newDim = dim.value();
        tensor.dims.push_back(newDim.get<int64_t>());
    }
    tensor.format = j.at("format").get<std::string>();
    tensor.shaderAccess = j.at("shader_access").get<ShaderAccessType>();
    if (tensor.shaderAccess == ShaderAccessType::Unknown) {
        throw std::runtime_error("Unknown shader_access type");
    }
    if (j.count("src") != 0) {
        tensor.src = j.at("src").get<std::string>();
    }
    if (j.count("dst") != 0) {
        tensor.dst = j.at("dst").get<std::string>();
    }
    if (j.count("alias_target") != 0) {
        if (j.count("memory_group") != 0) {
            throw std::runtime_error(
                "Unable to use both alias_target and memory_group types of aliasing simultaneously");
        }
        mlsdk::logging::warning("Use of \"alias_target\" in the scenario is deprecated. Use \"memory_group\" instead.");
        tensor.memoryGroup = MemoryGroup{Guid(j.at("alias_target").at("resource_ref").get<std::string>())};
    }
    if (j.count("memory_group") != 0) {
        tensor.memoryGroup = j.at("memory_group").get<MemoryGroup>();
    }
    if (j.count("tiling") != 0) {
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
    const json dimsJson = j.find("dims").value();
    for (auto dim = dimsJson.begin(); dim != dimsJson.end(); ++dim) {
        const json &newDim = dim.value();
        image.dims.push_back(newDim.get<uint32_t>());
    }
    // for compatibility with the old json configs that had this field set as "false"
    if (j.count("mips") == 0) {
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
    if (j.count("src") != 0) {
        image.src = j.at("src").get<std::string>();
    }
    if (j.count("dst") != 0) {
        image.dst = j.at("dst").get<std::string>();
    }
    if (j.count("min_filter") != 0) {
        image.minFilter = j.at("min_filter").get<FilterMode>();
        if (image.minFilter == FilterMode::Unknown) {
            throw std::runtime_error("Unknown min_filter value");
        }
    }
    if (j.count("mag_filter") != 0) {
        image.magFilter = j.at("mag_filter").get<FilterMode>();
        if (image.magFilter == FilterMode::Unknown) {
            throw std::runtime_error("Unknown mag_filter value");
        }
    }
    if (j.count("mip_filter") != 0) {
        image.mipFilter = j.at("mip_filter").get<FilterMode>();
        if (image.mipFilter == FilterMode::Unknown) {
            throw std::runtime_error("Unknown mip_filter value");
        }
    }
    if (j.count("border_address_mode") != 0) {
        image.borderAddressMode = j.at("border_address_mode").get<AddressMode>();
        if (image.borderAddressMode == AddressMode::Unknown) {
            throw std::runtime_error("Unknown border_address_mode value");
        }
    }
    if (j.count("border_color") != 0) {
        image.borderColor = j.at("border_color").get<BorderColor>();
        if (image.borderColor == BorderColor::Unknown) {
            throw std::runtime_error("Unknown border_color value");
        }
    }
    if (j.count("custom_border_color") != 0) {
        const json customColorJson = j.find("custom_border_color").value();
        if (image.borderColor.value() == BorderColor::FloatCustomEXT) {
            image.customBorderColor = customColorJson.get<std::array<float, 4>>();
        } else {
            image.customBorderColor = customColorJson.get<std::array<int, 4>>();
        }
    }
    if (j.count("tiling") != 0) {
        image.tiling = j.at("tiling").get<scenariorunner::Tiling>();
        if (image.tiling == Tiling::Unknown) {
            throw std::runtime_error("Unknown tiling value");
        }
    }
    if (j.count("memory_group") != 0) {
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
    if (j.count("subresource_range") != 0) {
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
