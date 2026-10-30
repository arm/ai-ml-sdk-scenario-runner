/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_options.hpp"

#include "compute.hpp"
#include "device_selection.hpp"
#include "glsl_compiler.hpp"
#include "vgf_view.hpp"
#include "workload_bindings.hpp"
#include "workload_execution.hpp"

#include "vgf-utils/temp_folder.hpp"
#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;
namespace vgf = mlsdk::vgflib;

// Use ordinary 32-bit storage buffers so the test needs only compute support.
std::shared_ptr<VgfView> makeTwoSegmentVgf(TempFolder &folder, bool reuseModule = false) {
    const std::string shader = R"(
#version 450
layout(local_size_x = 1) in;
layout(binding = 0) readonly buffer Input { int inputValue; };
layout(binding = 1) writeonly buffer Output { int outputValue; };
void main() { outputValue = inputValue OPERATION; }
)";
    const auto compile = [&](const std::string &operation) {
        auto source = shader;
        const std::string_view operationView = "OPERATION";
        source.replace(source.find(operationView), operationView.size(), operation);
        auto result = GlslCompiler::get().compile(source, ShaderStage::Compute);
        if (result.second.empty()) {
            throw std::runtime_error(result.first);
        }
        return result.second;
    };
    auto encoder = vgf::CreateEncoder(VK_HEADER_VERSION);
    const auto add = encoder->AddModule(vgf::ModuleType::COMPUTE, "add", "main", compile("+ 1"));
    const auto second =
        reuseModule ? add : encoder->AddModule(vgf::ModuleType::COMPUTE, "multiply", "main", compile("* 2"));
    const auto descriptor = vgf::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    const auto format = vgf::ToFormatType(VK_FORMAT_R32_SINT);
    const auto input = encoder->AddInputResource(descriptor, format, {1}, {});
    const auto intermediate = encoder->AddIntermediateResource(descriptor, format, {1}, {});
    const auto output = encoder->AddOutputResource(descriptor, format, {1}, {});
    const auto inputBinding = encoder->AddBindingSlot(0, input);
    const auto intermediateOutput = encoder->AddBindingSlot(1, intermediate);
    const auto intermediateInput = encoder->AddBindingSlot(0, intermediate);
    const auto outputBinding = encoder->AddBindingSlot(1, output);
    const auto firstSet = encoder->AddDescriptorSetInfo({inputBinding, intermediateOutput});
    const auto secondSet = encoder->AddDescriptorSetInfo({intermediateInput, outputBinding});
    encoder->AddModelSequenceInputsOutputs({inputBinding}, {"input"}, {outputBinding}, {"output"});
    const std::vector<vgf::GraphConstantBindingRef> constants;
    encoder->AddSegmentInfo(add, "first", {firstSet}, {inputBinding}, {intermediateOutput}, constants, {1, 1, 1});
    encoder->AddSegmentInfo(second, "second", {secondSet}, {intermediateInput}, {outputBinding}, constants, {1, 1, 1});
    encoder->Finish();
    const auto path = folder.relative("two_segments.vgf").string();
    {
        std::ofstream stream(path, std::ios::binary);
        encoder->WriteTo(stream);
    }
    return std::make_shared<VgfView>(VgfView::createVgfView(path));
}

TEST(VgfWorkload, ResolvesRepeatedModuleTargetsAndRetainsSource) {
    TempFolder folder("vgf_workload_targets");
    auto source = makeTwoSegmentVgf(folder, true);
    std::weak_ptr<const VgfView> retained = source;
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    auto execution = WorkloadExecution::fromVgf(std::make_shared<WorkloadRuntime>(context), source);
    source.reset();
    EXPECT_FALSE(retained.expired());

    const auto targets = execution->executableTargets("add");
    ASSERT_EQ(targets.size(), 2u);
    EXPECT_EQ(targets[0].index(), 0u);
    EXPECT_EQ(targets[0].name(), "first");
    EXPECT_EQ(targets[1].index(), 1u);
    EXPECT_EQ(targets[1].name(), "second");
    EXPECT_THROW(execution->executableTargets("missing"), std::runtime_error);
    execution.reset();
    EXPECT_TRUE(retained.expired());
}

TEST(VgfWorkload, RecordsSegmentsInOrderWithProfiling) {
    TempFolder folder("vgf_workload_recording");
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    auto execution = WorkloadExecution::fromVgf(std::make_shared<WorkloadRuntime>(context), makeTwoSegmentVgf(folder));
    execution->session().configure();

    DataManager resources;
    BufferInfo info;
    info.size = sizeof(int32_t);
    const BufferId input{0};
    const BufferId output{1};
    for (const auto id : {input, output}) {
        resources.createBuffer(id, info);
        auto &buffer = resources.getBufferMut(id);
        buffer.setup(context);
        buffer.allocateMemory(context);
        const int32_t value = id == input ? 3 : 0;
        buffer.fill(context, &value, sizeof(value));
    }

    auto bindings = execution->session().createBindingSet();
    const WorkloadBindings adapter(resources);
    // Workload exposes the external resources; it owns the intermediate buffer.
    ASSERT_EQ(execution->workload().resourceCount(), 2u);
    adapter.bind(bindings, execution->workload().resource(0),
                 {0, 0, input, std::nullopt, vk::DescriptorType::eStorageBuffer});
    adapter.bind(bindings, execution->workload().resource(1),
                 {0, 1, output, std::nullopt, vk::DescriptorType::eStorageBuffer});
    execution->prepare(bindings);

    Compute compute(context);
    uint32_t queries = 0;
    compute.registerVgfExecution(execution, "test_vgf", queries, true);
    ASSERT_EQ(queries, 4u);
    compute.setupQueryPool(queries);
    compute.submitAndWaitOnFence();

    auto &buffer = resources.getBufferMut(output);
    buffer.memoryManager()->downloadData(context, 0, sizeof(int32_t));
    int32_t result = 0;
    std::memcpy(&result, buffer.memoryManager()->mapStagingBufferMemory(0, sizeof(result)), sizeof(result));
    buffer.memoryManager()->unmapStagingBufferMemory();
    EXPECT_EQ(result, 8); // (3 + 1) * 2, with the intermediate supplied by workload-lib.

    const auto profile = compute.getRuntimeProfilingData();
    ASSERT_EQ(profile.commands.size(), 2u);
    EXPECT_EQ(profile.commands[0].name, "test_vgf/first");
    EXPECT_EQ(profile.commands[1].name, "test_vgf/second");
    EXPECT_EQ(profile.commands[0].type, "ComputeDispatch");
    EXPECT_EQ(profile.commands[1].type, "ComputeDispatch");
    EXPECT_EQ(profile.timestamps.size(), 4u);
}
} // namespace
