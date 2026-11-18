/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_options.hpp"

#include "compute.hpp"
#include "device_selection.hpp"
#include "glsl_compiler.hpp"
#include "resource_manager.hpp"
#include "vgf_bindings.hpp"
#include "vgf_configuration.hpp"
#include "workload_execution.hpp"

#include "vgf-utils/numpy.hpp"
#include "vgf-utils/temp_folder.hpp"
#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include "spirv-tools/libspirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;
namespace vgf = mlsdk::vgflib;

const std::string shaderSource = R"(
#version 450
layout(local_size_x = 1) in;
layout(constant_id = 5) const int factor = 2;
layout(push_constant) uniform Push { int bias; } push;
layout(set = 0, binding = 0) readonly buffer Input { int inputValue; };
layout(set = 0, binding = 1) writeonly buffer Output { int outputValue; };
void main() { outputValue = inputValue * factor + push.bias; }
)";

std::vector<uint32_t> compileShader(const std::string &source) {
    auto compiled = GlslCompiler::get().compile(source, ShaderStage::Compute);
    if (compiled.second.empty()) {
        throw std::runtime_error(compiled.first);
    }
    return std::move(compiled.second);
}

ShaderInfo replacementShader() {
    ShaderInfo shader;
    shader.debugName = "replacement";
    shader.entry = "replacement_entry";
    shader.pushConstantsSize = 4;
    shader.specializationConstants = {{5, {4}}};
    const spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    std::string assembly;
    if (!tools.Disassemble(compileShader(shaderSource), &assembly)) {
        throw std::runtime_error("Could not disassemble replacement shader");
    }
    assembly.replace(assembly.find("\"main\""), std::string("\"main\"").size(), "\"replacement_entry\"");
    std::vector<uint32_t> code;
    if (!tools.Assemble(assembly, &code)) {
        throw std::runtime_error("Could not assemble replacement shader");
    }
    shader.src = std::make_shared<const std::vector<uint32_t>>(std::move(code));
    return shader;
}

std::shared_ptr<VgfView> makeVgf(TempFolder &folder) {
    auto encoder = vgf::CreateEncoder(VK_HEADER_VERSION);
    const auto first =
        encoder->AddModule(vgf::ModuleType::COMPUTE, "first", "main", vgf::ShaderType::GLSL, shaderSource);
    auto replacedSource = shaderSource;
    replacedSource.replace(replacedSource.find("push.bias;"), std::string("push.bias;").size(), "push.bias + 1000;");
    const auto second = encoder->AddModule(vgf::ModuleType::COMPUTE, "second", "main", compileShader(replacedSource));
    const auto descriptor = vgf::ToDescriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    const auto format = vgf::ToFormatType(VK_FORMAT_R32_SINT);
    const auto input = encoder->AddInputResource(descriptor, format, {1}, {});
    const auto intermediate = encoder->AddIntermediateResource(descriptor, format, {1}, {});
    const auto output = encoder->AddOutputResource(descriptor, format, {1}, {});
    const auto inputBinding = encoder->AddBindingSlot(0, input);
    const auto intermediateOutput = encoder->AddBindingSlot(1, intermediate);
    const auto intermediateInput = encoder->AddBindingSlot(0, intermediate);
    const auto outputBinding = encoder->AddBindingSlot(1, output);
    encoder->AddModelSequenceInputsOutputs({encoder->AddBindingSlot(17, input)}, {"input"},
                                           {encoder->AddBindingSlot(29, output)}, {"output"});
    const std::vector<vgf::GraphConstantBindingRef> constants;
    encoder->AddSegmentInfo(first, "first", {encoder->AddDescriptorSetInfo({inputBinding, intermediateOutput})},
                            {inputBinding}, {intermediateOutput}, constants, {1, 1, 1});
    encoder->AddSegmentInfo(second, "second", {encoder->AddDescriptorSetInfo({intermediateInput, outputBinding})},
                            {intermediateInput}, {outputBinding}, constants, {1, 1, 1});
    encoder->Finish();
    const auto path = folder.relative("configured.vgf").string();
    {
        std::ofstream stream(path, std::ios::binary);
        encoder->WriteTo(stream);
    }
    return std::make_shared<VgfView>(VgfView::createVgfView(path));
}

void addPushData(DataManager &resources, TempFolder &folder, RawDataId id, int32_t value) {
    const auto path = folder.relative(std::to_string(value) + ".npy").string();
    mlsdk::vgfutils::numpy::write(path, reinterpret_cast<const char *>(&value), {1}, 'i', sizeof(value));
    resources.createRawData(id, RawDataInfo{"push", path});
}

TEST(VgfConfigurationIntegration, RecordsSourceAndSubstitutionWithPerTargetData) {
    TempFolder folder("vgf_configuration");
    const auto source = makeVgf(folder);
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    auto execution = WorkloadExecution::fromVgf(std::make_shared<WorkloadRuntime>(context), source);
    ResourceManager resources;
    DispatchVgfData dispatch{VgfId{0}};
    dispatch.shaderSubstitutions = {{resources.addShader(replacementShader()), "second"}};
    VgfInfo info;
    info.src = source;
    // VGF settings override replacement metadata; the first matching target wins.
    info.pushConstantsSize = 8;
    info.specializationConstantMaps = {{{{5, {3}}}, "first"}, {{{5, {5}}}, "second"}, {{{5, {99}}}, "second"}};
    configureVgfExecutables(*execution, info, dispatch, resources);
    for (const auto executable : execution->workload().executables()) {
        EXPECT_EQ(execution->session().pushConstantSize(executable), 8u);
    }

    DataManager data;
    const BufferId input{0};
    const BufferId output{1};
    for (const auto id : {input, output}) {
        data.createBuffer(id, BufferInfo{"buffer", sizeof(int32_t)});
        auto &buffer = data.getBufferMut(id);
        buffer.setup(context);
        buffer.allocateMemory(context);
        const int32_t value = id == input ? 3 : 0;
        buffer.fill(context, &value, sizeof(value));
    }
    const VgfBindings adapter(*source, execution->workload(), data,
                              {{0, 17, input, std::nullopt, vk::DescriptorType::eStorageBuffer},
                               {0, 29, output, std::nullopt, vk::DescriptorType::eStorageBuffer}});
    adapter.setResourceTilings(execution->session());
    execution->session().configure();
    auto bindings = execution->session().createBindingSet();
    adapter.bind(bindings);
    addPushData(data, folder, RawDataId{0}, 1);
    addPushData(data, folder, RawDataId{1}, 2);
    // Four-byte prefix updates are valid for the eight-byte effective layout.
    dispatch.pushConstants = {{RawDataId{0}, "first"}, {RawDataId{1}, "second"}, {RawDataId{0}, "second"}};
    bindVgfPushConstants(*execution, bindings, dispatch, data);
    execution->prepare(bindings);
    Compute compute(context);
    uint32_t queries = 0;
    compute.registerVgfExecution(execution, "configured", queries, true);
    compute.submitAndWaitOnFence();
    const auto downloaded = data.getBuffer(output).download(context);
    ASSERT_EQ(downloaded.data.size(), sizeof(int32_t));
    int32_t value = 0;
    std::memcpy(&value, downloaded.data.data(), sizeof(value));
    EXPECT_EQ(value, 52); // (3 * 3 + 1) * 5 + 2, with WL owning the intermediate.
}

} // namespace
