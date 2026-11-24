/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_options.hpp"

#include "compute.hpp"
#include "data_manager.hpp"
#include "device_selection.hpp"
#include "glsl_compiler.hpp"
#include "resource_manager.hpp"
#include "utils.hpp"
#include "workload_data_graph.hpp"
#include "workload_graph_diagnostics.hpp"

#include "vgf-utils/temp_folder.hpp"
#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;
namespace wl = mlsdk::workloadlib;

void replace(std::string &text, const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

ShaderInfo graphShader(const std::filesystem::path &asset) {
    std::ifstream file(std::filesystem::path(SCENARIO_RUNNER_TEST_SPVASM_DIR) / asset);
    if (!file) {
        throw std::runtime_error("Cannot open graph SPIR-V test asset: " + asset.string());
    }
    std::string assembly{std::istreambuf_iterator<char>(file), {}};
    replace(assembly, "INPUT_SET", "2");
    replace(assembly, "INPUT_BINDING", "3");
    replace(assembly, "OUTPUT_SET", "0");
    replace(assembly, "OUTPUT_BINDING", "7");
    replace(assembly, "\"main\"", "\"graph_entry\"");
    spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
    std::string diagnostics;
    tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
        diagnostics += std::string(message) + '\n';
    });
    auto code = std::make_shared<std::vector<uint32_t>>();
    if (!tools.Assemble(assembly, code.get()) || !tools.Validate(*code)) {
        throw std::runtime_error("Cannot assemble graph test asset: " + diagnostics);
    }
    ShaderInfo shader;
    shader.debugName = asset.string();
    shader.shaderType = ShaderType::SPIR_V;
    shader.entry = "graph_entry";
    shader.src = std::move(code);
    return shader;
}

template <typename T> std::vector<std::byte> bytes(const std::vector<T> &values) {
    std::vector<std::byte> result(values.size() * sizeof(T));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

detail::DeviceRequirements graphRequirements() {
    detail::DeviceRequirements requirements;
    requirements.queueFlags |= vk::QueueFlagBits::eDataGraphARM;
    requirements.capabilities = detail::DeviceCapability::DataGraphShaderModules | detail::DeviceCapability::Tensors |
                                detail::DeviceCapability::ShaderTensorAccess;
    return requirements;
}

class WorkloadDataGraph : public testing::Test {
  protected:
    Context context{ScenarioOptions{}, graphRequirements()};
    ResourceManager resources;
    DataManager data;
    std::shared_ptr<WorkloadRuntime> runtime = std::make_shared<WorkloadRuntime>(context);

    TensorId addTensor(const std::vector<int64_t> &shape, vk::Format format, Tiling tiling = Tiling::Linear,
                       std::shared_ptr<ResourceMemoryManager> memory = std::make_shared<ResourceMemoryManager>()) {
        TensorInfo info{};
        info.debugName = "tensor";
        info.shape = shape;
        info.format = format;
        info.tiling = tiling;
        const auto id = resources.addTensor(info);
        data.createTensor(id, info);
        auto &tensor = data.getTensorMut(id);
        tensor.setup(context, std::move(memory));
        tensor.allocateMemory(context);
        return id;
    }

    DispatchDataGraphData addGraph(const std::filesystem::path &asset) {
        return DispatchDataGraphData(resources.addShader(graphShader(asset)));
    }

    std::shared_ptr<WorkloadExecution> prepareVgf(TempFolder &folder, const std::vector<wl::ExecutableKind> &kinds) {
        const auto input = addTensor({1, 16, 16, 16}, vk::Format::eR8Sint);
        const auto output = addTensor({1, 8, 8, 16}, vk::Format::eR8Sint);
        const std::vector<int8_t> values(size_t{16} * 16 * 16, 0);
        data.getTensor(input).upload(context, {values.data(), values.size(), {1, 16, 16, 16}});
        namespace vgf = mlsdk::vgflib;
        auto encoder = vgf::CreateEncoder(VK_HEADER_VERSION);
        const auto code =
            GlslCompiler::get().compile("#version 450\nlayout(local_size_x=1) in; layout(set=0,binding=0) "
                                        "buffer Scratch { uint value; }; void main() { value=1; }",
                                        ShaderStage::Compute);
        if (code.second.empty()) {
            throw std::runtime_error(code.first);
        }
        const auto computeModule = encoder->AddModule(vgf::ModuleType::COMPUTE, "scratch", "main", code.second);
        const auto graph = encoder->AddModule(vgf::ModuleType::GRAPH, "pool", "graph_entry",
                                              *graphShader("test_vgf_graph/maxpool.spvasm").src);
        const auto descriptor = vgf::ToDescriptorType(VK_DESCRIPTOR_TYPE_TENSOR_ARM);
        const auto format = vgf::ToFormatType(VK_FORMAT_R8_SINT);
        const auto vgfInput = encoder->AddInputResource(descriptor, format, {1, 16, 16, 16}, {});
        const auto vgfOutput = encoder->AddOutputResource(descriptor, format, {1, 8, 8, 16}, {});
        const auto inputSlot = encoder->AddBindingSlot(3, vgfInput);
        const auto outputSlot = encoder->AddBindingSlot(7, vgfOutput);
        const auto inputSet = encoder->AddDescriptorSetInfo({inputSlot}, 2);
        const auto outputSet = encoder->AddDescriptorSetInfo({outputSlot}, 0);
        const auto scratch = encoder->AddIntermediateResource(vgf::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                                                              vgf::ToFormatType(VK_FORMAT_R32_UINT), {1}, {});
        const auto scratchSlot = encoder->AddBindingSlot(0, scratch);
        const auto scratchSet = encoder->AddDescriptorSetInfo({scratchSlot}, 0);
        encoder->AddModelSequenceInputsOutputs({inputSlot}, {"input"}, {outputSlot}, {"output"});
        const std::vector<vgf::GraphConstantBindingRef> constants;
        for (uint32_t index = 0; index < kinds.size(); ++index) {
            const auto name = "segment_" + std::to_string(index);
            if (kinds[index] == wl::ExecutableKind::Compute) {
                encoder->AddSegmentInfo(computeModule, name, {scratchSet}, {}, {scratchSlot}, constants, {1, 1, 1});
            } else {
                encoder->AddSegmentInfo(graph, name, {inputSet, outputSet}, {inputSlot}, {outputSlot}, constants);
            }
        }
        encoder->Finish();
        const auto path = folder.relative("mixed.vgf");
        {
            std::ofstream file(path, std::ios::binary);
            encoder->WriteTo(file);
        }
        wl::SessionOptions options;
        options.requireHostVisibleSessionMemory();
        const auto mixed = WorkloadExecution::fromVgf(runtime, loadVgfView(path.string()), options);
        mixed->session().configure();
        auto bindings = mixed->session().createBindingSet();
        bindings.bindTensor(mixed->workload().resource(0), {data.getTensor(input).tensor()});
        bindings.bindTensor(mixed->workload().resource(1), {data.getTensor(output).tensor()});
        mixed->prepare(bindings);
        return mixed;
    }

    GraphConstantResourceId addConstant(float value) {
        GraphConstantInfo constant{"weights", vk::Format::eR32Sfloat, {1, 3, 2, 1}};
        const std::vector<float> values(6, value);
        constant.data.resize(values.size() * sizeof(float));
        std::memcpy(constant.data.data(), values.data(), constant.data.size());
        return resources.addGraphConstant(std::move(constant));
    }
};

TEST_F(WorkloadDataGraph, RecordsMaxpoolWithReorderedSparseBindings) {
    const auto input = addTensor({1, 16, 16, 16}, vk::Format::eR8Sint);
    const auto output = addTensor({1, 8, 8, 16}, vk::Format::eR8Sint, Tiling::Optimal);
    auto dispatch = addGraph("test_vgf_graph/maxpool.spvasm");
    dispatch.bindings = {{0, 7, output, {}, vk::DescriptorType::eTensorARM},
                         {2, 3, input, {}, vk::DescriptorType::eTensorARM}};
    auto execution = prepareStandaloneDataGraph(runtime, dispatch, resources, data);
    ASSERT_TRUE(execution->isPrepared());
    const auto &workload = execution->workload();
    ASSERT_EQ(workload.resourceCount(), 2u);
    EXPECT_EQ(workload.executable(0).module().entryPoint(), "graph_entry");
    EXPECT_EQ(execution->session().tensorTiling(workload.resource(0)), vk::TensorTilingARM::eOptimal);

    std::vector<int8_t> values(size_t{16} * 16 * 16);
    for (size_t h = 0; h < 16; ++h) {
        for (size_t w = 0; w < 16; ++w) {
            for (size_t c = 0; c < 16; ++c) {
                values[(((h * 16) + w) * 16) + c] = static_cast<int8_t>(h + w + c);
            }
        }
    }
    data.getTensor(input).upload(context, {values.data(), values.size(), {1, 16, 16, 16}});
    Compute compute(context);
    compute.registerWorkloadExecution(execution, "maxpool", {"DataGraphDispatch", "maxpool"}, true);
    compute.submitAndWaitOnFence();
    std::vector<int8_t> expected(size_t{8} * 8 * 16);
    for (size_t h = 0; h < 8; ++h) {
        for (size_t w = 0; w < 8; ++w) {
            for (size_t c = 0; c < 16; ++c) {
                expected[(((h * 8) + w) * 16) + c] = static_cast<int8_t>((2 * h) + (2 * w) + c + 2);
            }
        }
    }
    EXPECT_EQ(data.getTensor(output).download(context).data, bytes(expected));
}

TEST_F(WorkloadDataGraph, ReportsSessionMemoryForStandaloneAndVgfDispatches) {
    const auto input = addTensor({1, 16, 16, 16}, vk::Format::eR8Sint);
    const auto output = addTensor({1, 8, 8, 16}, vk::Format::eR8Sint);
    auto dispatch = addGraph("test_vgf_graph/maxpool.spvasm");
    dispatch.bindings = {{0, 7, output, {}, vk::DescriptorType::eTensorARM},
                         {2, 3, input, {}, vk::DescriptorType::eTensorARM}};
    const auto execution = prepareStandaloneDataGraph(runtime, dispatch, resources, data);
    const auto &session = execution->session();
    uint64_t expectedBytes = 0;
    for (uint32_t i = 0; i < session.dataGraphSessionMemoryCount(0); ++i) {
        expectedBytes += session.dataGraphSessionMemory(0, i).size;
    }

    Compute compute(context);
    compute.registerWorkloadExecution(execution, "standalone", {"DataGraphDispatch", "standalone"}, true);
    uint32_t queries = 0;
    compute.registerVgfExecution(execution, "vgf", queries, true);

    const auto memory = compute.getMemoryProfilingData();
    ASSERT_EQ(memory.usages.size(), 2u);
    EXPECT_EQ(memory.usages[0].commandName, "standalone");
    EXPECT_EQ(memory.usages[0].sessionMemoryBytes, expectedBytes);
    EXPECT_EQ(memory.usages[1].commandName, "vgf/" + std::string(execution->workload().executable(0).name()));
    EXPECT_EQ(memory.usages[1].sessionMemoryBytes, expectedBytes);
}

TEST_F(WorkloadDataGraph, ProfilesVgfKindsWithStableQueryPairsAcrossRuns) {
    const auto input = addTensor({1, 16, 16, 16}, vk::Format::eR8Sint);
    const auto output = addTensor({1, 8, 8, 16}, vk::Format::eR8Sint);
    const std::vector<int8_t> values(size_t{16} * 16 * 16, 0);
    data.getTensor(input).upload(context, {values.data(), values.size(), {1, 16, 16, 16}});
    auto dispatch = addGraph("test_vgf_graph/maxpool.spvasm");
    dispatch.bindings = {{0, 7, output, {}, vk::DescriptorType::eTensorARM},
                         {2, 3, input, {}, vk::DescriptorType::eTensorARM}};
    const auto standalone = prepareStandaloneDataGraph(runtime, dispatch, resources, data);

    // Observe public recording calls and forward them to the driver.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto &dispatcher = const_cast<vk::raii::detail::DeviceDispatcher &>(*context.device().getDispatcher());
    static PFN_vkCmdWriteTimestamp2 timestamp;
    static PFN_vkCmdPipelineBarrier2 barrier;
    timestamp = dispatcher.vkCmdWriteTimestamp2;
    barrier = dispatcher.vkCmdPipelineBarrier2;
    static std::vector<std::pair<VkPipelineStageFlags2, uint32_t>> stamps;
    static uint32_t vgfQueryEnd;
    static bool inSegment;
    inSegment = false;
    const ScopeExit<void()> restore([&] {
        dispatcher.vkCmdWriteTimestamp2 = timestamp;
        dispatcher.vkCmdPipelineBarrier2 = barrier;
    });
    dispatcher.vkCmdWriteTimestamp2 =
        +[](VkCommandBuffer command, VkPipelineStageFlags2 stage, VkQueryPool pool, uint32_t query) {
            stamps.emplace_back(stage, query);
            if (query >= 2 && query < vgfQueryEnd) {
                inSegment = query % 2 == 0;
            }
            timestamp(command, stage, pool, query);
        };
    dispatcher.vkCmdPipelineBarrier2 = +[](VkCommandBuffer command, const VkDependencyInfo *info) {
        EXPECT_FALSE(inSegment) << "Synchronization must remain outside VGF segment query pairs";
        barrier(command, info);
    };
    const std::vector<std::vector<wl::ExecutableKind>> sequences{
        {wl::ExecutableKind::Compute},
        {wl::ExecutableKind::Graph},
        {wl::ExecutableKind::Compute, wl::ExecutableKind::Graph, wl::ExecutableKind::Compute,
         wl::ExecutableKind::Graph}};
    for (const auto &kinds : sequences) {
        TempFolder folder("vgf_profiling");
        const auto execution = prepareVgf(folder, kinds);
        for (const bool enabled : {false, true}) {
            SCOPED_TRACE(testing::Message() << "segments=" << kinds.size() << " profiling=" << enabled);
            Compute compute(context);
            uint32_t queries = 0;
            const auto registerStandalone = [&](const std::string &name) {
                compute.registerWriteTimestamp(queries++, vk::PipelineStageFlagBits2::eDataGraphARM);
                compute.registerWorkloadExecution(standalone, name, {"DataGraphDispatch", name}, true);
                compute.registerWriteTimestamp(queries++, vk::PipelineStageFlagBits2::eDataGraphARM);
            };
            registerStandalone("before");
            compute.registerVgfExecution(execution, "vgf", queries, true);
            vgfQueryEnd = queries;
            registerStandalone("after");
            ASSERT_EQ(queries, 2 * (kinds.size() + 2));
            if (enabled) {
                compute.setupQueryPool(queries);
            }
            const auto memory = compute.getMemoryProfilingData();
            size_t graph = 1;
            ASSERT_GE(memory.usages.size(), 2u);
            EXPECT_EQ(memory.usages.front().commandName, "before");
            for (uint32_t i = 0; i < kinds.size(); ++i) {
                if (kinds[i] == wl::ExecutableKind::Graph) {
                    ASSERT_LT(graph, memory.usages.size());
                    EXPECT_EQ(memory.usages[graph++].commandName, "vgf/segment_" + std::to_string(i));
                }
            }
            ASSERT_EQ(memory.usages.size(), graph + 1);
            EXPECT_EQ(memory.usages.back().commandName, "after");
            for (int iteration = 0; iteration < 2; ++iteration) {
                stamps.clear();
                compute.submitAndWaitOnFence();
                EXPECT_FALSE(inSegment);
                if (enabled) {
                    const auto profile = compute.getRuntimeProfilingData();
                    ASSERT_EQ(profile.commands.size(), kinds.size() + 2);
                    ASSERT_EQ(profile.timestamps.size(), queries);
                    ASSERT_EQ(stamps.size(), queries);
                    EXPECT_EQ(profile.commands.front().name, "before");
                    EXPECT_EQ(profile.commands.back().name, "after");
                    for (uint32_t i = 0; i < queries; ++i) {
                        EXPECT_EQ(stamps[i].second, i);
                        const auto kind = i >= 2 && i < vgfQueryEnd ? kinds[(i - 2) / 2] : wl::ExecutableKind::Graph;
                        const auto stage = kind == wl::ExecutableKind::Graph ? VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM
                                                                             : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        EXPECT_EQ(stamps[i].first, stage);
                    }
                    for (uint32_t i = 0; i < kinds.size(); ++i) {
                        EXPECT_EQ(profile.commands[i + 1].name, "vgf/segment_" + std::to_string(i));
                        EXPECT_EQ(profile.commands[i + 1].type,
                                  kinds[i] == wl::ExecutableKind::Graph ? "DataGraphDispatch" : "ComputeDispatch");
                    }
                } else {
                    EXPECT_TRUE(stamps.empty());
                }
                const auto repeatedMemory = compute.getMemoryProfilingData();
                ASSERT_EQ(repeatedMemory.usages.size(), memory.usages.size());
                for (size_t i = 0; i < memory.usages.size(); ++i) {
                    EXPECT_EQ(repeatedMemory.usages[i].commandName, memory.usages[i].commandName);
                    EXPECT_EQ(repeatedMemory.usages[i].sessionMemoryBytes, memory.usages[i].sessionMemoryBytes);
                }
            }
        }
    }
}

TEST_F(WorkloadDataGraph, DumpsVgfGraphExecutablesInCommandOrder) {
    TempFolder folder("vgf_graph_dumps");
    const auto input = addTensor({1, 16, 16, 16}, vk::Format::eR8Sint);
    const auto output = addTensor({1, 8, 8, 16}, vk::Format::eR8Sint);
    auto dispatch = addGraph("test_vgf_graph/maxpool.spvasm");
    dispatch.bindings = {{0, 7, output, {}, vk::DescriptorType::eTensorARM},
                         {2, 3, input, {}, vk::DescriptorType::eTensorARM}};
    mlsdk::workloadlib::SessionOptions options;
    options.requireHostVisibleSessionMemory();
    const auto standalone = prepareStandaloneDataGraph(runtime, dispatch, resources, data, options);

    const auto mixed = prepareVgf(folder, {wl::ExecutableKind::Compute, wl::ExecutableKind::Graph,
                                           wl::ExecutableKind::Compute, wl::ExecutableKind::Graph});

    // Mock property bytes so routing checks do not depend on hardware diagnostics
    // or nonzero graph-session RAM allocations (the emulator may allocate none).
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto &dispatcher = const_cast<vk::raii::detail::DeviceDispatcher &>(*context.device().getDispatcher());
    const auto originalAvailable = dispatcher.vkGetDataGraphPipelineAvailablePropertiesARM;
    const auto originalProperty = dispatcher.vkGetDataGraphPipelinePropertiesARM;
    const ScopeExit<void()> restore([&] {
        dispatcher.vkGetDataGraphPipelineAvailablePropertiesARM = originalAvailable;
        dispatcher.vkGetDataGraphPipelinePropertiesARM = originalProperty;
    });
    static std::map<VkPipeline, std::string> payloads;
    static std::string payload;
    payloads.clear();
    dispatcher.vkGetDataGraphPipelineAvailablePropertiesARM =
        +[](VkDevice, const VkDataGraphPipelineInfoARM *info, uint32_t *count,
            VkDataGraphPipelinePropertyARM *properties) {
            if (!payload.empty()) {
                payloads[info->dataGraphPipeline] = payload;
            }
            *count = 1;
            if (properties) {
                properties[0] = static_cast<VkDataGraphPipelinePropertyARM>(0x7ffffffe);
            }
            return VK_SUCCESS;
        };
    dispatcher.vkGetDataGraphPipelinePropertiesARM =
        +[](VkDevice, const VkDataGraphPipelineInfoARM *info, uint32_t count,
            VkDataGraphPipelinePropertyQueryResultARM *properties) {
            const auto &value = payloads.at(info->dataGraphPipeline);
            for (uint32_t i = 0; i < count; ++i) {
                if (properties[i].pData) {
                    std::memcpy(properties[i].pData, value.data(), value.size());
                }
                properties[i].isText = VK_TRUE;
                properties[i].dataSize = value.size();
            }
            return VK_SUCCESS;
        };
    payload = "standalone";
    standalone->session().dataGraphPipelineProperties(0);
    payload = "first graph";
    mixed->session().dataGraphPipelineProperties(1);
    payload = "second graph";
    mixed->session().dataGraphPipelineProperties(3);
    payload.clear();
    Compute compute(context);
    compute.registerWorkloadExecution(standalone, "before", {"DataGraphDispatch", "before"}, true);
    uint32_t queries = 0;
    compute.registerVgfExecution(mixed, "mixed", queries, true);
    compute.registerWorkloadExecution(standalone, "after", {"DataGraphDispatch", "after"}, true);
    const auto directory = folder.relative("profiling");
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    compute.dumpGraphProfilingData(directory);
    uint32_t fileIndex = 0;
    for (const std::string expected : {"standalone", "first graph", "second graph", "standalone"}) {
        const auto name = "EmulationLayerPipeline_" + std::to_string(fileIndex++) + ".json";
        std::ifstream file(directory / name, std::ios::binary);
        ASSERT_TRUE(file) << name;
        const std::string actual{std::istreambuf_iterator<char>(file), {}};
        EXPECT_EQ(actual, expected);
    }
    EXPECT_EQ(static_cast<uint32_t>(
                  std::distance(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator{})),
              fileIndex);

    // Property helpers must also select a graph after the initial compute segment.
    EXPECT_NO_THROW(dumpWorkloadGraphDebugDatabase(mixed->session(), 3, folder.relative("debug.bin")));
    EXPECT_NO_THROW(
        dumpWorkloadGraphStatistics(mixed->session(), 1, context.device(), folder.relative("statistics.bin")));
    const auto ramDirectory = folder.relative("ram");
    ASSERT_TRUE(std::filesystem::create_directory(ramDirectory));
    uint32_t ramIndex = 7;
    dumpWorkloadGraphSessionRAMs(mixed->session(), 3, context.device(), ramDirectory, ramIndex);
    EXPECT_EQ(ramIndex, 7 + mixed->session().dataGraphSessionMemoryCount(3));
}

TEST_F(WorkloadDataGraph, BorrowsConstantsAndRecordsRepeatedlyAfterSetupReturns) {
    const auto input = addTensor({1, 3, 2, 1}, vk::Format::eR32Sfloat);
    const auto output = addTensor({1, 3, 2, 1}, vk::Format::eR32Sfloat);
    const auto constant = addConstant(0.5F);
    std::shared_ptr<WorkloadExecution> execution;
    {
        auto dispatch = addGraph("test_data_graph/add_f32_constant.spvasm");
        dispatch.bindings = {{0, 7, output, {}, vk::DescriptorType::eTensorARM},
                             {2, 3, input, {}, vk::DescriptorType::eTensorARM}};
        dispatch.graphConstants = {constant};
        execution = prepareStandaloneDataGraph(runtime, dispatch, resources, data);
    }
    ASSERT_TRUE(execution->isPrepared());
    Compute compute(context);
    compute.registerWorkloadExecution(execution, "add", {"DataGraphDispatch", "add"}, true);
    // Compute owns the prepared execution after the adapter and dispatch have gone away.
    execution.reset();
    for (int iteration = 0; iteration < 2; ++iteration) {
        const std::vector<float> values{static_cast<float>(iteration), 1, 2, 3, 4, 5};
        auto expected = values;
        std::transform(expected.begin(), expected.end(), expected.begin(), [](float value) { return value + 0.5F; });
        data.getTensor(input).upload(context, {values.data(), values.size() * sizeof(float), {1, 3, 2, 1}});
        compute.submitAndWaitOnFence();
        EXPECT_EQ(data.getTensor(output).download(context).data, bytes(expected));
        compute.reset();
    }
}

TEST_F(WorkloadDataGraph, DescribesConstantsInDispatchOrderWithoutCopyingPayloads) {
    const auto first = addConstant(1.0F);
    const auto second = addConstant(2.0F);
    auto dispatch = addGraph("test_data_graph/add_f32_constant.spvasm");
    dispatch.graphConstants = {second, first};
    dispatch.implicitBarrier = false;
    const auto description = describeStandaloneDataGraph(dispatch, resources, data);
    EXPECT_FALSE(description.implicitBarrier);
    ASSERT_EQ(description.constants.size(), 2u);
    EXPECT_EQ(description.constants[0].data, resources.get(second).data.data());
    EXPECT_EQ(description.constants[1].data, resources.get(first).data.data());
    EXPECT_EQ(description.constants[0].size, 6u * sizeof(float));
}

TEST_F(WorkloadDataGraph, DescribesResolvedScalarShapeAndImageAliasStrides) {
    const auto scalar = addTensor({}, vk::Format::eR32Sfloat);
    auto memory = std::make_shared<ResourceMemoryManager>();
    memory->updateFormat(vk::Format::eR32Sfloat);
    memory->updateImageType(vk::ImageType::e2D);
    memory->updateSubResourceRowPitch(32);
    memory->updateMemSize(128);
    const auto alias = addTensor({1, 2, 3, 1}, vk::Format::eR32Sfloat, Tiling::Linear, memory);
    auto dispatch = addGraph("test_data_graph/add_f32_constant.spvasm");
    dispatch.bindings = {{2, 3, scalar, {}, vk::DescriptorType::eTensorARM},
                         {0, 7, alias, {}, vk::DescriptorType::eTensorARM}};
    const auto description = describeStandaloneDataGraph(dispatch, resources, data);
    EXPECT_EQ(description.resources[0].resource.tensor.shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(description.resources[1].resource.tensor.stride, (std::vector<int64_t>{128, 32, 4, 4}));
}

TEST(WorkloadDataGraphDescription, RejectsNonSpirvAndMissingCode) {
    ResourceManager resources;
    DataManager data;
    ShaderInfo shader{};
    shader.shaderType = ShaderType::GLSL;
    DispatchDataGraphData dispatch(resources.addShader(shader));
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
    shader.shaderType = ShaderType::SPIR_V;
    dispatch.graphShader = resources.addShader(shader);
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
    shader.src = std::make_shared<const std::vector<uint32_t>>();
    dispatch.graphShader = resources.addShader(shader);
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
}

TEST(WorkloadDataGraphDescription, RejectsNonTensorBindingsAndMismatchedDescriptors) {
    ResourceManager resources;
    DataManager data;
    // Description translation checks representation and presence; compilation validates SPIR-V.
    ShaderInfo shader{};
    shader.shaderType = ShaderType::SPIR_V;
    shader.src = std::make_shared<const std::vector<uint32_t>>(1, 0);
    DispatchDataGraphData dispatch(resources.addShader(shader));
    dispatch.bindings = {{0, 0, BufferId{0}, {}, vk::DescriptorType::eStorageBuffer}};
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
    dispatch.bindings = {{0, 0, TensorId{0}, {}, vk::DescriptorType::eStorageImage}};
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
    dispatch.bindings = {{0, 0, ImageId{0}, {}, vk::DescriptorType::eStorageImage}};
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
    dispatch.bindings = {{0, 0, ImageId{0}, {}, vk::DescriptorType::eCombinedImageSampler}};
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
    dispatch.bindings = {{0, 0, ImageId{0}, {}, vk::DescriptorType::eTensorARM}};
    EXPECT_THROW(describeStandaloneDataGraph(dispatch, resources, data), std::runtime_error);
}

} // namespace
