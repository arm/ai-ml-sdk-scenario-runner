/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_options.hpp"

#include "compute.hpp"
#include "device_selection.hpp"
#include "glsl_compiler.hpp"
#include "vgf_bindings.hpp"
#include "vgf_view.hpp"
#include "workload_execution.hpp"

#include "vgf-utils/temp_folder.hpp"
#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;
namespace vgf = mlsdk::vgflib;

std::shared_ptr<VgfView> writeVgf(TempFolder &folder, vgf::Encoder &encoder) {
    encoder.Finish();
    const auto path = folder.relative("workload.vgf").string();
    {
        std::ofstream stream(path, std::ios::binary);
        encoder.WriteTo(stream);
    }
    return std::make_shared<VgfView>(VgfView::createVgfView(path));
}

// Use ordinary 32-bit storage buffers so the test needs only compute support.
std::shared_ptr<VgfView> makeTwoSegmentVgf(TempFolder &folder, bool reuseModule = false,
                                           std::optional<vgf::AliasGroupId> aliasGroup = std::nullopt,
                                           bool unusedInput = false) {
    const std::string shader = R"(
#version 450
layout(local_size_x = 1) in;
layout(set = 1, binding = 4) readonly buffer Input { int inputValue; };
layout(set = 2, binding = 9) writeonly buffer Output { int outputValue; };
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
    // Deliberately put an internal resource before the public resources.
    const auto intermediate = encoder->AddIntermediateResource(descriptor, format, {1}, {}, aliasGroup);
    const auto input = encoder->AddInputResource(descriptor, format, {1}, {}, aliasGroup);
    const auto output = encoder->AddOutputResource(descriptor, format, {1}, {});
    if (aliasGroup) {
        // A second, larger peer ensures allocation planning visits every peer.
        encoder->AddIntermediateResource(descriptor, format, {16}, {}, aliasGroup);
    }
    const auto inputBinding = encoder->AddBindingSlot(4, input);
    const auto intermediateOutput = encoder->AddBindingSlot(9, intermediate);
    const auto intermediateInput = encoder->AddBindingSlot(4, intermediate);
    const auto outputBinding = encoder->AddBindingSlot(9, output);
    const auto reusedInput = encoder->AddBindingSlot(15, input);
    const auto firstInputSet = encoder->AddDescriptorSetInfo({inputBinding}, 1);
    const auto firstOutputSet = encoder->AddDescriptorSetInfo({intermediateOutput}, 2);
    const auto secondInputSet = encoder->AddDescriptorSetInfo({intermediateInput, reusedInput}, 1);
    const auto secondOutputSet = encoder->AddDescriptorSetInfo({outputBinding}, 2);
    // Interface IDs and names do not identify segment descriptors or public indices.
    std::vector<vgf::BindingSlotRef> interfaceInputs{encoder->AddBindingSlot(17, input)};
    std::vector<std::string> inputNames{"same_name"};
    if (unusedInput) {
        const auto unused = encoder->AddInputResource(descriptor, format, {1}, {});
        interfaceInputs.push_back(encoder->AddBindingSlot(99, unused));
        inputNames.emplace_back("unused");
        // This public resource is unused and absent from the model interface.
        encoder->AddOutputResource(descriptor, format, {1}, {});
    }
    std::vector<vgf::BindingSlotRef> interfaceOutputs{encoder->AddBindingSlot(29, output)};
    std::vector<std::string> outputNames{"same_name"};
    encoder->AddModelSequenceInputsOutputs(interfaceInputs, inputNames, interfaceOutputs, outputNames);
    const std::vector<vgf::GraphConstantBindingRef> constants;
    encoder->AddSegmentInfo(add, "first", {firstInputSet, firstOutputSet}, {inputBinding}, {intermediateOutput},
                            constants, {1, 1, 1});
    encoder->AddSegmentInfo(second, "second", {secondInputSet, secondOutputSet}, {intermediateInput, reusedInput},
                            {outputBinding}, constants, {1, 1, 1});
    return writeVgf(folder, *encoder);
}

TEST(VgfWorkload, ResolvesRepeatedModuleTargetsAndRetainsSource) {
    TempFolder folder("vgf_workload_targets");
    auto source = makeTwoSegmentVgf(folder, true);
    std::weak_ptr<const VgfView> retained = source;
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    auto execution = WorkloadExecution::fromVgf(std::make_shared<WorkloadRuntime>(context), source);
    source.reset();
    EXPECT_FALSE(retained.expired());
    ASSERT_EQ(retained.lock()->getModelInterface().inputs.size(), 1u);
    EXPECT_EQ(retained.lock()->getModelInterface().inputs[0].bindingSlot.mBinding, 17u);

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
    const auto source = makeTwoSegmentVgf(folder);
    auto execution = WorkloadExecution::fromVgf(std::make_shared<WorkloadRuntime>(context), source);

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

    const VgfBindings adapter(*source, execution->workload(), resources,
                              {{0, 29, output, std::nullopt, vk::DescriptorType::eStorageBuffer},
                               {0, 17, input, std::nullopt, vk::DescriptorType::eStorageBuffer}});
    ASSERT_EQ(adapter.resources().size(), 2u);
    EXPECT_EQ(adapter.resources()[0].binding.resource, MemoryResourceId(input));
    EXPECT_EQ(adapter.resources()[1].binding.resource, MemoryResourceId(output));
    adapter.setResourceTilings(execution->session());
    execution->session().configure();
    EXPECT_TRUE(adapter.aliasMemoryRequirements(execution->session()).empty());
    auto bindings = execution->session().createBindingSet();
    adapter.bind(bindings);
    execution->prepare(bindings);

    Compute compute(context);
    uint32_t queries = 0;
    compute.registerVgfExecution(execution, "test_vgf", queries, true);
    ASSERT_EQ(queries, 4u);
    compute.setupQueryPool(queries);
    compute.submitAndWaitOnFence();

    const auto &buffer = resources.getBuffer(output);
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

TEST(VgfWorkload, RetainsContextualExternalResourceValidation) {
    TempFolder folder("vgf_bindings_validation");
    const auto source = makeTwoSegmentVgf(folder);
    const auto workload = source->createWorkload();
    DataManager resources;
    resources.createBuffer(BufferId{0}, BufferInfo{"input", 4});
    resources.createBuffer(BufferId{1}, BufferInfo{"output", 4});
    resources.createBuffer(BufferId{2}, BufferInfo{"wrong_size", 8});
    const TypedBinding input{0, 17, BufferId{0}, std::nullopt, vk::DescriptorType::eStorageBuffer};
    const TypedBinding output{0, 29, BufferId{1}, std::nullopt, vk::DescriptorType::eStorageBuffer};
    auto wrongSize = input;
    wrongSize.resource = BufferId{2};
    try {
        const VgfBindings bindings(*source, workload, resources, {wrongSize, output});
        FAIL() << "Expected VGF input size validation to fail";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("input idx 0 expects buffer size 4"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("wrong_size"), std::string::npos);
    }
}

TEST(VgfWorkload, BindsUnusedInterfaceResourcesAndSkipsUndeclaredResources) {
    for (const bool bindUnused : {false, true}) {
        SCOPED_TRACE(bindUnused);
        TempFolder folder("vgf_unused_interface");
        Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
        const auto source = makeTwoSegmentVgf(folder, false, std::nullopt, true);
        auto execution = WorkloadExecution::fromVgf(std::make_shared<WorkloadRuntime>(context), source);
        ASSERT_EQ(execution->workload().resourceCount(), 4u);
        DataManager resources;
        const BufferId input{0};
        const BufferId output{1};
        const BufferId unused{2};
        for (const auto id : {input, output, unused}) {
            resources.createBuffer(id, BufferInfo{"buffer", sizeof(int32_t)});
            auto &buffer = resources.getBufferMut(id);
            buffer.setup(context);
            buffer.allocateMemory(context);
            const int32_t value = id == input ? 3 : 101;
            buffer.fill(context, &value, sizeof(value));
        }
        const auto descriptor = vk::DescriptorType::eStorageBuffer;
        const TypedBinding inputBinding{0, 17, input, std::nullopt, descriptor};
        const TypedBinding outputBinding{0, 29, output, std::nullopt, descriptor};
        std::vector<TypedBinding> supplied{outputBinding, inputBinding};
        if (bindUnused) {
            supplied.push_back({0, 99, unused, std::nullopt, descriptor});
        }
        const VgfBindings adapter(*source, execution->workload(), resources, supplied);
        ASSERT_EQ(adapter.resources().size(), bindUnused ? 3u : 2u);
        ASSERT_EQ(source->getModelInterface().inputs.size(), 2u);
        EXPECT_EQ(source->getModelInterface().inputs[1].bindingSlot.mBinding, 99u);
        EXPECT_EQ(source->getModelInterface().inputs[1].inputOutputIndex, 2u);
        if (bindUnused) {
            EXPECT_EQ(adapter.resources()[2].resource.index(), 2u);
            EXPECT_EQ(adapter.resources()[2].binding.resource, MemoryResourceId(unused));
        }
        adapter.setResourceTilings(execution->session());
        execution->session().configure();
        const VgfBindings missingInput(*source, execution->workload(), resources, {outputBinding});
        auto incompleteBindings = execution->session().createBindingSet();
        missingInput.bind(incompleteBindings);
        EXPECT_THROW(execution->prepare(incompleteBindings), std::runtime_error);
        auto bindings = execution->session().createBindingSet();
        adapter.bind(bindings);
        execution->prepare(bindings);
        Compute compute(context);
        uint32_t queries = 0;
        compute.registerVgfExecution(execution, "test_vgf", queries, true);
        compute.submitAndWaitOnFence();
        for (const auto id : {output, unused}) {
            const auto result = resources.getBuffer(id).download(context);
            ASSERT_EQ(result.data.size(), sizeof(int32_t));
            int32_t value = 0;
            std::memcpy(&value, result.data.data(), sizeof(value));
            EXPECT_EQ(value, id == output ? 8 : 101);
        }
    }
}

TEST(VgfWorkload, QueriesEveryAliasPeerBeforeAllocatingAndBindsBackingMemory) {
    TempFolder folder("vgf_alias_requirements");
    const auto source = makeTwoSegmentVgf(folder, false, 17);
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    auto execution = WorkloadExecution::fromVgf(std::make_shared<WorkloadRuntime>(context), source);
    DataManager resources;
    const BufferId input{0};
    const BufferId output{1};
    resources.createBuffer(input, BufferInfo{"input", 4});
    resources.createBuffer(output, BufferInfo{"output", 4});
    auto backing = std::make_shared<ResourceMemoryManager>();
    backing->markShared();
    resources.getBufferMut(input).setup(context, backing);
    resources.getBufferMut(output).setup(context);
    const auto descriptor = vk::DescriptorType::eStorageBuffer;
    const VgfBindings adapter(*source, execution->workload(), resources,
                              {{0, 17, input, std::nullopt, descriptor}, {0, 29, output, std::nullopt, descriptor}});
    ASSERT_EQ(adapter.resources().size(), 2u);
    EXPECT_EQ(adapter.resources()[0].aliasGroup, 17u);
    EXPECT_FALSE(adapter.resources()[1].aliasGroup.has_value());
    adapter.setResourceTilings(execution->session());
    execution->session().configure();
    const auto requirements = adapter.aliasMemoryRequirements(execution->session());
    ASSERT_EQ(requirements.size(), 2u);
    EXPECT_FALSE(backing->isInitalized());
    for (const auto &entry : requirements) {
        EXPECT_EQ(entry.resource, MemoryResourceId(input));
        ASSERT_FALSE(entry.requirements.requiresDedicatedAllocation);
        EXPECT_GT(entry.requirements.memory.alignment, 0u);
        EXPECT_NE(entry.requirements.memory.memoryTypeBits, 0u);
        backing->updateMemSize(entry.requirements.memory.size);
        backing->updateMemType(entry.requirements.memory.memoryTypeBits);
    }
    EXPECT_GE(backing->getMemSize(), 16 * sizeof(int32_t));
    resources.getBufferMut(input).allocateMemory(context);
    resources.getBufferMut(output).allocateMemory(context);
    auto bindings = execution->session().createBindingSet();
    EXPECT_THROW(adapter.bind(bindings), std::invalid_argument);
    adapter.bind(bindings, [&](MemoryResourceId resource) {
        EXPECT_EQ(resource, MemoryResourceId(input));
        return mlsdk::workloadlib::BoundMemoryInfo{*backing->getDeviceMemory(), 0, backing->getMemSize()};
    });
    EXPECT_NO_THROW(execution->prepare(bindings));
}

TEST(VgfWorkload, AppliesExternalTensorAndImageTilingsBeforeConfiguration) {
    TempFolder folder("vgf_resource_tilings");
    auto encoder = vgf::CreateEncoder(VK_HEADER_VERSION);
    const auto module = encoder->AddModule(vgf::ModuleType::COMPUTE, "placeholder", "main");
    const auto tensor = encoder->AddInputResource(vgf::ToDescriptorType(VK_DESCRIPTOR_TYPE_TENSOR_ARM),
                                                  vgf::ToFormatType(VK_FORMAT_R32_SINT), {1}, {});
    const auto image = encoder->AddOutputResource(vgf::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
                                                  vgf::ToFormatType(VK_FORMAT_R8G8B8A8_UNORM), {2, 2, 4}, {});
    const auto input = encoder->AddBindingSlot(4, tensor);
    const auto output = encoder->AddBindingSlot(9, image);
    const auto set = encoder->AddDescriptorSetInfo({input, output});
    encoder->AddModelSequenceInputsOutputs({input}, {"tensor"}, {output}, {"image"});
    encoder->AddSegmentInfo(module, "segment", {set}, {input}, {output}, std::vector<vgf::GraphConstantBindingRef>{},
                            {1, 1, 1});
    const auto source = writeVgf(folder, *encoder);
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    const auto runtime = std::make_shared<WorkloadRuntime>(context);
    auto execution = WorkloadExecution::fromVgf(runtime, source);
    DataManager resources;
    TensorInfo tensorInfo{"tensor", {1}, vk::Format::eR32Sint};
    tensorInfo.tiling = Tiling::Optimal;
    resources.createTensor(TensorId{0}, tensorInfo);
    ImageInfo imageInfo{};
    imageInfo.shape = {1, 2, 2, 1};
    imageInfo.format = imageInfo.targetFormat = vk::Format::eR8G8B8A8Unorm;
    imageInfo.isStorage = true;
    imageInfo.mips = 1;
    imageInfo.tiling = Tiling::Linear;
    resources.createImage(ImageId{0}, imageInfo);
    resources.getImageMut(ImageId{0}).setup(context);
    const VgfBindings adapter(*source, execution->workload(), resources,
                              {{0, 4, TensorId{0}, std::nullopt, vk::DescriptorType::eTensorARM},
                               {0, 9, ImageId{0}, std::nullopt, vk::DescriptorType::eStorageImage}});
    // No tensor object or pipeline is needed to select session tilings.
    adapter.setResourceTilings(execution->session());
    EXPECT_EQ(execution->session().tensorTiling(execution->workload().resource(0)), vk::TensorTilingARM::eOptimal);
    EXPECT_EQ(execution->session().imageTiling(execution->workload().resource(1)), vk::ImageTiling::eLinear);
    auto foreign = WorkloadExecution::fromVgf(runtime, source);
    EXPECT_THROW(adapter.setResourceTilings(foreign->session()), std::invalid_argument);
}
} // namespace
