/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_options.hpp"

#include "compute.hpp"
#include "data_manager.hpp"
#include "device_selection.hpp"
#include "resource_manager.hpp"
#include "workload_data_graph.hpp"

#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;

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
