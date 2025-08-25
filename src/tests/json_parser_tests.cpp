/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "guid.hpp"
#include "json_reader.hpp"
#include "resource_desc.hpp"
#include "scenario_desc.hpp"

#include <fstream>
#include <regex>
#include <sstream>

/**
 * @brief Test the JSON parser.
 *
 */

using namespace mlsdk::scenariorunner;

namespace {
const std::string jsonData =
    R""(
{
    "commands": [
        {
            "dispatch_graph": {
                "bindings": [
                    {
                        "id": 0,
                        "resource_ref": "InBuffer1",
                        "set": 0
                    }
                ],
                "graph_ref": "graph1"
            }
        },
        {
            "dispatch_compute": {
                "bindings": [
                    {
                        "id": 0,
                        "resource_ref": "InBuffer1",
                        "set": 0
                    },
                    {
                        "id": 1,
                        "resource_ref": "InBuffer2",
                        "set": 0
                    },
                    {
                        "id": 2,
                        "resource_ref": "OutBuffer",
                        "set": 0
                    }
                ],
                "push_data_ref": "RawData",
                "rangeND": [10, 1, 1],
                "shader_ref": "add_shader"
            }
        }
    ],
    "resources": [
        {
            "shader": {
                "build_options": "-DQUANTIZE",
                "entry": "main",
                "specialization_constants": [
                    {
                        "id": 0,
                        "value": 8.0
                    },
                    {
                        "id": 1,
                        "value": 8.0
                    }
                ],
                "src": "./shaders/add_shader.spv",
                "type": "SPIR-V",
                "uid": "matmul_shader"
            }
        },
        {
            "buffer": {
                "shader_access": "readonly",
                "size": 0,
                "src": "./shader_data/inbuffer1.npy",
                "uid": "InBuffer1"
            }
        },
        {
            "buffer": {
                "shader_access": "readonly",
                "size": 0,
                "src": "./shader_data/inbuffer2.npy",
                "uid": "InBuffer2"
            }
        },
        {
            "buffer": {
                "dst": "./shader_data/outbuffer.npy",
                "shader_access": "readwrite",
                "size": 0,
                "uid": "OutBuffer"
            }
        },
        {
            "raw_data": {
                "src": "./graph_data/rawdata.npy",
                "uid": "RawData"
            }
        },
        {
            "graph": {
                "push_constants_size": 10,
                "shader_substitutions": [
                    {
                        "shader_ref": "prep_shader",
                        "target": "tfl_custom_pre_node"
                    },
                    {
                        "shader_ref": "post_shader",
                        "target": "tfl_custom_post_node"
                    }
                ],
                "specialization_constants": [
                    {
                        "specialization_constants": [
                            {
                                "id": 0,
                                "value": 8.0
                            },
                            {   "id": 1,
                                "value": 8.0
                            }
                        ],
                        "shader_target": "add_shader"
                    }
                ],
                "src": "./graphs/graph1.vgf",
                "uid": "graph1"
            }
        },
        {
            "tensor": {
                "src": "./graph_data/intensor1.npy",
                "dims": [1, 4, 8, 16],
                "format": "VK_FORMAT_R8_SINT",
                "shader_access": "readonly",
                "uid": "InTensor1"
            }
        },
        {
            "image": {
                "border_address_mode": "CLAMP_EDGE",
                "border_color": "INT_TRANSPARENT_BLACK",
                "dims": [
                    256,
                    256
                ],
                "dst": "",
                "format": "VK_FORMAT_R8G8B8A8_SRGB",
                "mag_filter": "LINEAR",
                "min_filter": "LINEAR",
                "mip_filter": "LINEAR",
                "mips": false,
                "shader_access": "readonly",
                "src": "./color.dds",
                "uid": "InputColorBuffer0"
            }
        },
        {
            "image": {
                "border_address_mode": "CLAMP_BORDER",
                "border_color": "FLOAT_CUSTOM_EXT",
                "custom_border_color": [
                    1,
                    2,
                    3,
                    4
                ],
                "dims": [
                    256,
                    256
                ],
                "dst": "",
                "format": "VK_FORMAT_R8G8B8A8_SRGB",
                "mag_filter": "LINEAR",
                "min_filter": "LINEAR",
                "mip_filter": "LINEAR",
                "mips": false,
                "shader_access": "readonly",
                "src": "./color.dds",
                "uid": "InputColorBuffer0"
            }
        },
        {
            "image": {
                "border_address_mode": "CLAMP_BORDER",
                "border_color": "INT_CUSTOM_EXT",
                "custom_border_color": [
                    5,
                    6,
                    7,
                    8
                ],
                "dims": [
                    256,
                    256
                ],
                "dst": "",
                "format": "VK_FORMAT_R8G8B8A8_SRGB",
                "mag_filter": "LINEAR",
                "min_filter": "LINEAR",
                "mip_filter": "LINEAR",
                "mips": false,
                "shader_access": "readonly",
                "src": "./color.dds",
                "uid": "InputColorBuffer0"
            }
        }
    ]
}
)"";

// Helper utility to create an object from a json string.
template <typename T> T MakeFromJSON(const std::string &s) {
    json j;
    std::istringstream iss(s);
    iss >> j;

    T obj;
    from_json(j, obj);
    return obj;
}

} // namespace

// Test the JSON parser:
// 1. Deserialize a JSON test case
// 2. Validate the number of commands and resources
TEST(JsonParser, DeSerialization) {
    std::istringstream iss(jsonData);
    ScenarioSpec spec{&iss, {}};

    ASSERT_TRUE(spec.commands.size() == 2);
    ASSERT_TRUE(spec.resources.size() == 10);
}

using namespace mlsdk::scenariorunner;

TEST(JsonParser, Empty) {
    std::istringstream empty1("{ \"resources\": [], \"commands\": [] }");
    ASSERT_NO_THROW(ScenarioSpec(&empty1, {}));

    std::istringstream empty2("");
    ASSERT_THROW(ScenarioSpec(&empty2, {}), nlohmann::json_abi_v3_11_3::detail::parse_error);
}

TEST(JsonParser, NoCommands) {
    const std::string jsonInput =
        R""(
    {
    "resources": [
        {
        "buffer": {
                "shader_access": "readonly",
                "size": 0,
                "src": "./shader_data/inbuffer1.npy",
                "uid": "InBuffer1"
            }
        }
    ]
    }
    )"";

    std::istringstream iss(jsonInput);
    ASSERT_NO_THROW(ScenarioSpec(&iss, {}));
}

TEST(JsonParser, NoResources) {
    const std::string jsonInput =
        R""(
    {
    "commands": [
        {
            "dispatch_graph": {
                "bindings": [
                    {
                        "id": 0,
                        "resource_ref": "InBuffer1",
                        "set": 0
                    }
                ],
                "graph_ref": "graph1"
            }
        }
    ]
    }
    )"";

    std::istringstream iss(jsonInput);
    ASSERT_NO_THROW(ScenarioSpec(&iss, {}));
}

TEST(JsonParser, UnknownResource) {
    const std::string jsonInput =
        R""(
    {
    "resources": [
            {
        "unknown resource": {
                "shader_access": "readonly",
                "size": 0,
                "src": "./shader_data/inbuffer1.npy",
                "uid": "InBuffer1"
            }
        }
    ]
    }
    )"";

    std::istringstream iss(jsonInput);
    ASSERT_THROW(ScenarioSpec(&iss, {}), std::runtime_error); // "Unknown Resource type"
}

TEST(JsonParser, UnknownCommand) {
    const std::string jsonInput =
        R""(
    {
    "commands": [
            {
        "unknown command": {
                "bindings": [
                    {
                        "id": 0,
                        "resource_ref": "InBuffer1",
                        "set": 0
                    }
                ],
                "graph_ref": "graph1"
            }
        }
    ]
    }
    )"";

    std::istringstream iss(jsonInput);
    ASSERT_THROW(ScenarioSpec(&iss, {}), std::runtime_error); // "Unknown Command type"
}

TEST(JsonParser, Resources) {
    const std::string jsonInputTemplate =
        R""(
    {
    "resources": [
        {RESOURCE}
    ]
    }
    )"";
    const std::regex resourceRegex("\\{RESOURCE\\}");

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
        "buffer": {
                "shader_access": "readonly",
                "size": 0,
                "src": "./shader_data/inbuffer1.npy",
                "uid": "InBuffer1"
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::Buffer);
        auto &buffer = reinterpret_cast<std::unique_ptr<BufferDesc> &>(resource);
        ASSERT_TRUE(buffer->size == 0);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
        "image": {
            "uid": "string",
            "dims": [1, 2, 3],
            "mips": 1,
            "format": "VkFormat enum",
            "shader_access": "readonly"

            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::Image);
        auto &resourcePtr = reinterpret_cast<std::unique_ptr<ImageDesc> &>(resource);
        ASSERT_TRUE(resourcePtr->dims.size() == 3);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "tensor": {
                "shader_access": "readonly",
                "dims": [1, 16, 16, 16],
                "format": "VK_FORMAT_R8_SINT",
                "uid": "input-0"
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::Tensor);
        auto &resourcePtr = reinterpret_cast<std::unique_ptr<TensorDesc> &>(resource);
        ASSERT_TRUE(resourcePtr->dims.size() == 4);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "raw_data": {
                "uid": "string",
                "src": "path"
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::RawData);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "shader": {
                "uid": "string",
                "src": "path",
                "type": "SPIR-V",
                "entry": "main"
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::Shader);
        auto &resourcePtr = reinterpret_cast<std::unique_ptr<ShaderDesc> &>(resource);
        ASSERT_TRUE(resourcePtr->shaderType == ShaderType::SPIR_V);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "graph": {
                "uid": "my_network",
                "src": "path"
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::DataGraph);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "memory_barrier": {
                "uid": "string",
                "src_access": "memory_read",
                "dst_access": "memory_write",
                "src_stage": ["graph"],
                "dst_stage": ["all"]
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::MemoryBarrier);
        auto &resourcePtr = reinterpret_cast<std::unique_ptr<MemoryBarrierDesc> &>(resource);
        ASSERT_TRUE(resourcePtr->srcAccess == MemoryAccess::MemoryRead);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "buffer_barrier": {
                "uid": "string",
                "src_access": "memory_read",
                "dst_access": "memory_write",
                "src_stage": ["graph"],
                "dst_stage": ["all"],
                "buffer_resource": "string",
                "offset": 1,
                "size": 1
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::BufferBarrier);
        auto &resourcePtr = reinterpret_cast<std::unique_ptr<BufferBarrierDesc> &>(resource);
        ASSERT_TRUE(resourcePtr->offset == 1);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "image_barrier": {
                "uid": "string",
                "src_access": "memory_read",
                "dst_access": "memory_write",
                "src_stage": ["graph"],
                "dst_stage": ["all"],
                "old_layout": "general",
                "new_layout": "undefined",
                "image_resource": "string",
                "image_range": {
                    "subresource_range": {
                        "base_mip_level": 0
                    }
                }
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::ImageBarrier);
        auto &resourcePtr = reinterpret_cast<std::unique_ptr<ImageBarrierDesc> &>(resource);
        ASSERT_TRUE(resourcePtr->oldLayout == ImageLayout::General);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, resourceRegex,
                                       R""(
        {
            "tensor_barrier": {
                "uid": "string",
                "src_access": "memory_read",
                "dst_access": "memory_write",
                "src_stage": ["graph"],
                "dst_stage": ["all"],
                "tensor_resource": "string"
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &resource = scenarioSpec.resources.at(0);
        ASSERT_TRUE(resource->resourceType == ResourceType::TensorBarrier);
        auto &resourcePtr = reinterpret_cast<std::unique_ptr<TensorBarrierDesc> &>(resource);
        ASSERT_TRUE(resourcePtr->tensorResource == "string");
    }
}

TEST(JsonParser, Commands) {
    const std::string jsonInputTemplate =
        R""(
    {
    "commands": [
        {COMMAND}
    ]
    }
    )"";
    const std::regex commandRegex("\\{COMMAND\\}");

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, commandRegex,
                                       R""(
        {
            "dispatch_graph": {
                "bindings": [
                    {
                        "id": 0,
                        "resource_ref": "InBuffer1",
                        "set": 0
                    }
                ],
                "graph_ref": "graph1",
                "push_constants": [
                    {
                        "push_data_ref": "RawData1",
                        "shader_target": "Shader1"
                    }
                ]
            }
        }

        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &command = scenarioSpec.commands.at(0);
        ASSERT_TRUE(command->commandType == CommandType::DispatchDataGraph);
        auto &commandPtr = reinterpret_cast<std::unique_ptr<DispatchDataGraphDesc> &>(command);

        auto &binding = commandPtr->bindings.at(0);
        ASSERT_TRUE(binding.id == 0);

        auto &pushConstant = commandPtr->pushConstants.at(0);
        ASSERT_TRUE(pushConstant.pushDataRef == Guid("RawData1"));
        ASSERT_TRUE(pushConstant.shaderTarget == Guid("Shader1"));
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, commandRegex,
                                       R""(
        {
            "dispatch_compute": {
                "shader_ref": "Shader",
                "push_data_ref": "RawData",
                   "bindings": [
                    {
                        "id": 0,
                        "resource_ref": "InBuffer1",
                        "set": 0
                    },
                    {
                        "id": 1,
                        "resource_ref": "InBuffer2",
                        "set": 0
                    },
                    {
                        "id": 2,
                        "resource_ref": "OutBuffer",
                        "set": 0
                    },
                    {
                        "id": 3,
                        "resource_ref": "InImage",
                        "set": 0,
                        "descriptor_type": "VK_DESCRIPTOR_TYPE_AUTO"
                    },
                    {
                        "id": 4,
                        "resource_ref": "OutImage",
                        "set": 0,
                        "descriptor_type": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
                    }
                ],
                "push_data_ref": "RawData",
                "rangeND": [10, 1, 1],
                "implicit_barrier": false

            }
        }

        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &command = scenarioSpec.commands.at(0);
        ASSERT_TRUE(command->commandType == CommandType::DispatchCompute);
        auto &commandPtr = reinterpret_cast<std::unique_ptr<DispatchComputeDesc> &>(command);

        auto &binding1 = commandPtr->bindings.at(1);
        ASSERT_TRUE(binding1.id == 1);
        ASSERT_TRUE(binding1.resourceRef == Guid("InBuffer2"));
        ASSERT_TRUE(binding1.descriptorType == DescriptorType::Auto);

        auto &binding2 = commandPtr->bindings.at(2);
        ASSERT_TRUE(binding2.id == 2);
        ASSERT_TRUE(binding2.resourceRef == Guid("OutBuffer"));
        ASSERT_TRUE(binding2.descriptorType == DescriptorType::Auto);

        auto &binding3 = commandPtr->bindings.at(3);
        ASSERT_TRUE(binding3.id == 3);
        ASSERT_TRUE(binding3.resourceRef == Guid("InImage"));
        ASSERT_TRUE(binding3.descriptorType == DescriptorType::Auto);

        auto &binding4 = commandPtr->bindings.at(4);
        ASSERT_TRUE(binding4.id == 4);
        ASSERT_TRUE(binding4.resourceRef == Guid("OutImage"));
        ASSERT_TRUE(binding4.descriptorType == DescriptorType::StorageImage);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, commandRegex,
                                       R""(
        {
            "dispatch_barrier": {
                "image_barrier_refs": ["string"],
                "memory_barrier_refs": ["string", "string"],
                "buffer_barrier_refs": ["string"],
                "tensor_barrier_refs": ["string"]
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &command = scenarioSpec.commands.at(0);
        ASSERT_TRUE(command->commandType == CommandType::DispatchBarrier);
        auto &commandPtr = reinterpret_cast<std::unique_ptr<DispatchBarrierDesc> &>(command);

        ASSERT_TRUE(commandPtr->imageBarriersRef.size() == 1);
        ASSERT_TRUE(commandPtr->memoryBarriersRef.size() == 2);
        ASSERT_TRUE(commandPtr->bufferBarriersRef.size() == 1);
        ASSERT_TRUE(commandPtr->tensorBarriersRef.size() == 1);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, commandRegex,
                                       R""(
        {
            "mark_boundary": {
                "resources": ["string"],
                "frame_id": 1
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &command = scenarioSpec.commands.at(0);
        ASSERT_TRUE(command->commandType == CommandType::MarkBoundary);
        auto &commandPtr = reinterpret_cast<std::unique_ptr<MarkBoundaryDesc> &>(command);

        ASSERT_TRUE(commandPtr->resources.size() == 1);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, commandRegex,
                                       R""(
        {
            "mark_boundary": {
                "resources": ["string"],
                "frame_id": "1"
            }
        }
        )"");

        std::istringstream iss(jsonInput);
        ScenarioSpec scenarioSpec{&iss, {}};
        auto &command = scenarioSpec.commands.at(0);
        ASSERT_TRUE(command->commandType == CommandType::MarkBoundary);
        auto &commandPtr = reinterpret_cast<std::unique_ptr<MarkBoundaryDesc> &>(command);

        ASSERT_TRUE(commandPtr->resources.size() == 1);
    }
}

TEST(JsonParser, DispatchDataGraph) {
    const std::string jsonInput =
        R""(
    {
        "bindings": [
            {
                "id": 0,
                "resource_ref": "InBuffer1",
                "set": 3
            }
        ],
        "graph_ref": "graph1"
    }
    )"";

    auto desc = MakeFromJSON<DispatchDataGraphDesc>(jsonInput);

    ASSERT_TRUE(desc.dataGraphRef.isValid());

    ASSERT_TRUE(desc.bindings.size() == 1);
    ASSERT_TRUE(desc.bindings[0].id == 0);
    ASSERT_TRUE(desc.bindings[0].set == 3);
    ASSERT_TRUE(desc.bindings[0].resourceRef.isValid());
}

TEST(JsonParser, DipatchCompute) {

    const std::string jsonInput =
        R""(
    {
        "bindings": [
            {
                "id": 0,
                "resource_ref": "InBuffer1",
                "set": 0
            },
            {
                "id": 1,
                "resource_ref": "InBuffer2",
                "set": 0
            },
            {
                "id": 2,
                "resource_ref": "OutBuffer",
                "set": 0
            }
        ],
        "push_data_ref": "RawData",
        "rangeND": [10, 1, 1],
        "shader_ref": "add_shader"
    }
    )"";

    auto desc = MakeFromJSON<DispatchComputeDesc>(jsonInput);

    ASSERT_TRUE(desc.shaderRef == Guid("add_shader"));

    ASSERT_TRUE(desc.bindings.size() == 3);
    ASSERT_TRUE(desc.bindings[2].id == 2);
    ASSERT_TRUE(desc.bindings[2].set == 0);
    ASSERT_TRUE(desc.bindings[2].resourceRef == Guid("OutBuffer"));

    ASSERT_TRUE(desc.rangeND.size() == 3);
    ASSERT_TRUE(desc.rangeND[0] == 10);
    ASSERT_TRUE(desc.rangeND[1] == 1);
    ASSERT_TRUE(desc.rangeND[2] == 1);

    ASSERT_TRUE(desc.pushDataRef.has_value());
    ASSERT_TRUE(desc.pushDataRef.value() == Guid("RawData"));
}

TEST(JsonParser, BufferResource) {
    {
        const std::string jsonInput =
            R""(
        {
            "shader_access": "readonly",
            "size": 48,
            "src": "./shader_data/inbuffer2.npy",
            "uid": "InBuffer2"
        }
        )"";

        auto desc = MakeFromJSON<BufferDesc>(jsonInput);

        ASSERT_TRUE(desc.guid == Guid("InBuffer2"));
        ASSERT_TRUE(desc.size == 48);
        ASSERT_TRUE(desc.shaderAccess == ShaderAccessType::ReadOnly);
        ASSERT_TRUE(desc.src.has_value() == true);
        ASSERT_TRUE(desc.src.value() == "./shader_data/inbuffer2.npy");
        ASSERT_TRUE(desc.dst.has_value() == false);
    }

    {
        const std::string jsonInput =
            R""(
        {
            "shader_access": "writeonly",
            "size": 52,
            "dst": "./shader_data/outbuffer.npy",
            "uid": "OutBuffer"
        }
        )"";

        auto desc = MakeFromJSON<BufferDesc>(jsonInput);

        ASSERT_TRUE(desc.guid == Guid("OutBuffer"));
        ASSERT_TRUE(desc.size == 52);
        ASSERT_TRUE(desc.shaderAccess == ShaderAccessType::WriteOnly);
        ASSERT_TRUE(desc.dst.has_value() == true);
        ASSERT_TRUE(desc.dst.value() == "./shader_data/outbuffer.npy");
        ASSERT_TRUE(desc.src.has_value() == false);
    }

    {
        const std::string jsonInput =
            R""(
        {
            "shader_access": "readwrite",
            "size": 16,
            "uid": "InOutBuffer"
        }
        )"";

        auto desc = MakeFromJSON<BufferDesc>(jsonInput);

        ASSERT_TRUE(desc.guid == Guid("InOutBuffer"));
        ASSERT_TRUE(desc.size == 16);
        ASSERT_TRUE(desc.shaderAccess == ShaderAccessType::ReadWrite);
        ASSERT_TRUE(desc.dst.has_value() == false);
        ASSERT_TRUE(desc.src.has_value() == false);
    }

    {
        const std::string jsonInput =
            R""(
        {
            "shader_access": "something not recognised",
            "size": 16,
            "uid": "InOutBuffer"
        }
        )"";

        ASSERT_THROW(MakeFromJSON<BufferDesc>(jsonInput), std::runtime_error); // "Unknown shader_access value"
    }
}

TEST(JsonParser, ShaderResource) {

    const std::string jsonInput =
        R""(
    {
        "build_options": "-DQUANTIZE",
        "entry": "main",
        "specialization_constants": [
            {
                "id": 0,
                "value": 8.0
            },
            {
                "id": 1,
                "value": 12.0
            }
        ],
        "src": "./shaders/add_shader.spv",
        "type": "SPIR-V",
        "uid": "matmul_shader"
    }
    )"";

    auto desc = MakeFromJSON<ShaderDesc>(jsonInput);

    ASSERT_TRUE(desc.guid == Guid("matmul_shader"));
    ASSERT_TRUE(desc.src == "./shaders/add_shader.spv");
    ASSERT_TRUE(desc.entry == "main");
    ASSERT_TRUE(desc.shaderType == ShaderType::SPIR_V);
    ASSERT_TRUE(desc.pushConstantsSize == 0);
    ASSERT_TRUE(desc.specializationConstants.size() == 2);
    ASSERT_TRUE(desc.specializationConstants[0].id == 0);
    ASSERT_TRUE(desc.specializationConstants[0].value.f == 8.0f);
    ASSERT_TRUE(desc.specializationConstants[1].id == 1);
    ASSERT_TRUE(desc.specializationConstants[1].value.f == 12.0f);
    ASSERT_TRUE(desc.buildOpts == "-DQUANTIZE");
}

TEST(JsonParser, RawDataResource) {

    const std::string jsonInput =
        R""(
    {
        "src": "./graph_data/rawdata.npy",
        "uid": "RawData"
    }
    )"";

    auto desc = MakeFromJSON<RawDataDesc>(jsonInput);

    ASSERT_TRUE(desc.guid == Guid("RawData"));
    ASSERT_TRUE(desc.src == "./graph_data/rawdata.npy");
}

TEST(JsonParser, TensorResource) {

    const std::string jsonInput =
        R""(
    {
        "src": "./graph_data/intensor1.npy",
        "dims": [1, 4, 8, 16],
        "format": "VK_FORMAT_R8_SINT",
        "shader_access": "readonly",
        "uid": "InTensor1",
        "tiling": "OPTIMAL"
    }
    )"";

    auto desc = MakeFromJSON<TensorDesc>(jsonInput);

    ASSERT_TRUE(desc.guid == Guid("InTensor1"));
    ASSERT_TRUE(desc.src.has_value());
    ASSERT_TRUE(desc.src.value() == "./graph_data/intensor1.npy");
    ASSERT_TRUE(desc.dims.size() == 4);
    ASSERT_TRUE(desc.dims[0] == 1);
    ASSERT_TRUE(desc.dims[1] == 4);
    ASSERT_TRUE(desc.dims[2] == 8);
    ASSERT_TRUE(desc.dims[3] == 16);
    ASSERT_TRUE(desc.format == "VK_FORMAT_R8_SINT");
    ASSERT_TRUE(desc.shaderAccess == ShaderAccessType::ReadOnly);
    ASSERT_TRUE(desc.tiling.has_value());
    ASSERT_TRUE(desc.tiling.value() == Tiling::Optimal);
}

TEST(JsonParser, ImageResource) {

    const std::string jsonInputTemplate =
        R""(
    {
        "border_address_mode": "{BORDER_ADDRESS_MODE}",
        "border_color": "{BORDER_COLOR}",
        "dims": [
            256,
            512
        ],
        "dst": "",
        "format": "VK_FORMAT_R8G8B8A8_SRGB",
        "mag_filter": "NEAREST",
        "min_filter": "LINEAR",
        "mip_filter": "LINEAR",
        "mips": false,
        "shader_access": "readwrite",
        "src": "./color.dds",
        "uid": "InputColorBuffer0",
        "tiling": "{TILING}"
    }
    )"";
    const std::regex border_color("\\{BORDER_COLOR\\}");
    const std::regex border_address_mode("\\{BORDER_ADDRESS_MODE\\}");
    const std::regex tiling("\\{TILING\\}");

    {

        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_color, "INT_TRANSPARENT_BLACK");
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "REPEAT");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.guid == Guid("InputColorBuffer0"));
        ASSERT_TRUE(desc.dims.size() == 2);
        ASSERT_TRUE(desc.dims[0] == 256);
        ASSERT_TRUE(desc.dims[1] == 512);
        ASSERT_TRUE(desc.format == "VK_FORMAT_R8G8B8A8_SRGB");
        ASSERT_TRUE(desc.shaderAccess == ShaderAccessType::ReadWrite);
        ASSERT_TRUE(desc.dst.has_value());
        ASSERT_TRUE(desc.dst.value() == "");
        ASSERT_TRUE(desc.src.has_value());
        ASSERT_TRUE(desc.src.value() == "./color.dds");

        ASSERT_TRUE(desc.mips == 1);
        ASSERT_TRUE(desc.minFilter.has_value());
        ASSERT_TRUE(desc.minFilter.value() == FilterMode::Linear);
        ASSERT_TRUE(desc.magFilter.has_value());
        ASSERT_TRUE(desc.magFilter.value() == FilterMode::Nearest);
        ASSERT_TRUE(desc.mipFilter.has_value());
        ASSERT_TRUE(desc.mipFilter.value() == FilterMode::Linear);

        ASSERT_TRUE(desc.borderAddressMode.has_value());
        ASSERT_TRUE(desc.borderAddressMode.value() == AddressMode::Repeat);
        ASSERT_TRUE(desc.borderColor.has_value());
        ASSERT_TRUE(desc.borderColor.value() == BorderColor::IntTransparentBlack);

        ASSERT_TRUE(desc.tiling.has_value());
        ASSERT_TRUE(desc.tiling.value() == Tiling::Linear);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "MIRRORED_REPEAT");
        jsonInput = std::regex_replace(jsonInput, border_color, "INT_TRANSPARENT_BLACK");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderAddressMode.has_value());
        ASSERT_TRUE(desc.borderAddressMode.value() == AddressMode::MirroredRepeat);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "CLAMP_EDGE");
        jsonInput = std::regex_replace(jsonInput, border_color, "INT_TRANSPARENT_BLACK");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderAddressMode.has_value());
        ASSERT_TRUE(desc.borderAddressMode.value() == AddressMode::ClampEdge);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "CLAMP_BORDER");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");

        jsonInput = std::regex_replace(jsonInput, border_color, "INT_TRANSPARENT_BLACK");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderColor.has_value());
        ASSERT_TRUE(desc.borderColor.value() == BorderColor::IntTransparentBlack);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "CLAMP_BORDER");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");
        jsonInput = std::regex_replace(jsonInput, border_color, "INT_OPAQUE_BLACK");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderColor.has_value());
        ASSERT_TRUE(desc.borderColor.value() == BorderColor::IntOpaqueBlack);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "CLAMP_BORDER");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");
        jsonInput = std::regex_replace(jsonInput, border_color, "INT_OPAQUE_WHITE");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderColor.has_value());
        ASSERT_TRUE(desc.borderColor.value() == BorderColor::IntOpaqueWhite);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "CLAMP_BORDER");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");
        jsonInput = std::regex_replace(jsonInput, border_color, "FLOAT_TRANSPARENT_BLACK");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderColor.has_value());
        ASSERT_TRUE(desc.borderColor.value() == BorderColor::FloatTransparentBlack);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "CLAMP_BORDER");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");
        jsonInput = std::regex_replace(jsonInput, border_color, "FLOAT_OPAQUE_BLACK");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderColor.has_value());
        ASSERT_TRUE(desc.borderColor.value() == BorderColor::FloatOpaqueBlack);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "CLAMP_BORDER");
        jsonInput = std::regex_replace(jsonInput, tiling, "LINEAR");
        jsonInput = std::regex_replace(jsonInput, border_color, "FLOAT_OPAQUE_WHITE");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.borderColor.has_value());
        ASSERT_TRUE(desc.borderColor.value() == BorderColor::FloatOpaqueWhite);
    }

    {
        std::string jsonInput = jsonInputTemplate;
        jsonInput = std::regex_replace(jsonInput, border_color, "INT_TRANSPARENT_BLACK");
        jsonInput = std::regex_replace(jsonInput, border_address_mode, "REPEAT");
        jsonInput = std::regex_replace(jsonInput, tiling, "OPTIMAL");

        auto desc = MakeFromJSON<ImageDesc>(jsonInput);
        ASSERT_TRUE(desc.tiling.has_value());
        ASSERT_TRUE(desc.tiling.value() == Tiling::Optimal);
    }
}

TEST(JsonParser, ImageBarrier) {
    const std::regex stagesRegex("\\{STAGES\\}");

    const std::string jsonBarrierInput = R""(
    {
        "uid": "uid",
        "src_access": "compute_shader_write",
        "dst_access": "compute_shader_read",
        {STAGES}
        "old_layout": "general",
        "new_layout": "general",
        "image_resource": "resource"
    }
    )"";

    {
        const std::string barrierStages = R""(
        "src_stage": ["compute"],
        "dst_stage": ["compute"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<ImageBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::Compute});
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::Compute});
        ASSERT_TRUE(desc.oldLayout == ImageLayout::General);
        ASSERT_TRUE(desc.newLayout == ImageLayout::General);
        ASSERT_TRUE(desc.imageResource == "resource");
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["all"],
        "dst_stage": ["all"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<ImageBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.oldLayout == ImageLayout::General);
        ASSERT_TRUE(desc.newLayout == ImageLayout::General);
        ASSERT_TRUE(desc.imageResource == "resource");
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["compute", "graph"],
        "dst_stage": ["compute", "graph"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<ImageBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>({PipelineStage::Compute, PipelineStage::Graph}));
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>({PipelineStage::Compute, PipelineStage::Graph}));
        ASSERT_TRUE(desc.oldLayout == ImageLayout::General);
        ASSERT_TRUE(desc.newLayout == ImageLayout::General);
        ASSERT_TRUE(desc.imageResource == "resource");
    }

    {

        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, "");
        auto desc = MakeFromJSON<ImageBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.oldLayout == ImageLayout::General);
        ASSERT_TRUE(desc.newLayout == ImageLayout::General);
        ASSERT_TRUE(desc.imageResource == "resource");
    }
}

TEST(JsonParser, TensorBarrier) {
    const std::regex stagesRegex("\\{STAGES\\}");

    const std::string jsonBarrierInput = R""(
    {
        "uid": "uid",
        "src_access": "graph_write",
        "dst_access": "compute_shader_read",
        {STAGES}
        "tensor_resource": "resource"
      }
    )"";

    {
        const std::string barrierStages = R""(
        "src_stage": ["graph"],
        "dst_stage": ["compute"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<TensorBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::GraphWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::Graph});
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::Compute});
        ASSERT_TRUE(desc.tensorResource == "resource");
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["all"],
        "dst_stage": ["all"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<TensorBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::GraphWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.tensorResource == "resource");
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["compute", "graph"],
        "dst_stage": ["graph", "compute"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<TensorBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::GraphWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>({PipelineStage::Compute, PipelineStage::Graph}));
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>({PipelineStage::Graph, PipelineStage::Compute}));
        ASSERT_TRUE(desc.tensorResource == "resource");
    }

    {

        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, "");
        auto desc = MakeFromJSON<TensorBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::GraphWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.tensorResource == "resource");
    }
}

TEST(JsonParser, BufferBarrier) {
    const std::regex stagesRegex("\\{STAGES\\}");

    const std::string jsonBarrierInput = R""(
    {
        "uid": "uid",
        "src_access": "compute_shader_write",
        "dst_access": "compute_shader_read",
        {STAGES}
        "buffer_resource": "buffer",
        "offset": 1024,
        "size": 2048
      }
    )"";

    {
        const std::string barrierStages = R""(
        "src_stage": ["compute"],
        "dst_stage": ["compute"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<BufferBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::Compute});
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::Compute});
        ASSERT_TRUE(desc.offset == 1024);
        ASSERT_TRUE(desc.size == 2048);
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["all"],
        "dst_stage": ["all"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<BufferBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.offset == 1024);
        ASSERT_TRUE(desc.size == 2048);
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["compute", "graph"],
        "dst_stage": ["graph", "compute"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<BufferBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>({PipelineStage::Compute, PipelineStage::Graph}));
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>({PipelineStage::Graph, PipelineStage::Compute}));
        ASSERT_TRUE(desc.offset == 1024);
        ASSERT_TRUE(desc.size == 2048);
    }

    {

        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, "");
        auto desc = MakeFromJSON<BufferBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.offset == 1024);
        ASSERT_TRUE(desc.size == 2048);
    }
}

TEST(JsonParser, GlobalMemBarrier) {
    const std::regex stagesRegex("\\{STAGES\\}");

    const std::string jsonBarrierInput = R""(
    {
        "uid": "uid",
        {STAGES}
        "src_access": "compute_shader_write",
        "dst_access": "compute_shader_read"
      }
    )"";

    {
        const std::string barrierStages = R""(
        "src_stage": ["graph"],
        "dst_stage": ["compute"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<MemoryBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::Graph});
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::Compute});
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["all"],
        "dst_stage": ["all"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<MemoryBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
    }

    {
        const std::string barrierStages = R""(
        "src_stage": ["compute", "graph"],
        "dst_stage": ["graph", "compute"],
        )"";
        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, barrierStages);
        auto desc = MakeFromJSON<MemoryBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>({PipelineStage::Compute, PipelineStage::Graph}));
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>({PipelineStage::Graph, PipelineStage::Compute}));
    }

    {

        auto json = std::regex_replace(jsonBarrierInput, stagesRegex, "");
        auto desc = MakeFromJSON<MemoryBarrierDesc>(json);

        ASSERT_TRUE(desc.guid == Guid("uid"));
        ASSERT_TRUE(desc.srcAccess == MemoryAccess::ComputeShaderWrite);
        ASSERT_TRUE(desc.dstAccess == MemoryAccess::ComputeShaderRead);
        ASSERT_TRUE(desc.dstStages == std::vector<PipelineStage>{PipelineStage::All});
        ASSERT_TRUE(desc.srcStages == std::vector<PipelineStage>{PipelineStage::All});
    }
}
