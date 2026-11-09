/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_options.hpp"

#include "compute.hpp"
#include "device_selection.hpp"
#include "glsl_compiler.hpp"
#include "workload_compute.hpp"

#include "vgf-utils/numpy.hpp"
#include "vgf-utils/temp_folder.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;

TEST(ComputeWorkload, RecordsExistingBuffersWithSpecializationAndDispatchShape) {
    const std::string source = R"(
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 7) const uint factor = 1;
layout(set = 1, binding = 3) readonly buffer Input { uint inputValues[]; };
layout(set = 1, binding = 9) writeonly buffer Output { uint outputValues[]; };
void main() {
    uint index = gl_GlobalInvocationID.x + 2 * gl_GlobalInvocationID.y;
    outputValues[index] = inputValues[index] * factor;
}
)";
    auto spirv = GlslCompiler::get().compile(source, ShaderStage::Compute);
    ASSERT_TRUE(spirv.first.empty()) << spirv.first;

    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    DataManager resources;
    const BufferId input{0};
    const BufferId output{1};
    const std::array<uint32_t, 4> values{1, 2, 3, 4};
    for (const auto id : {input, output}) {
        resources.createBuffer(id, BufferInfo{"buffer", sizeof(values), 0});
        auto &buffer = resources.getBufferMut(id);
        buffer.setup(context);
        buffer.allocateMemory(context);
        buffer.fill(context, values.data(), sizeof(values));
    }

    ShaderInfo shader;
    shader.entry = "main";
    shader.stage = ShaderStage::Compute;
    shader.src = std::make_shared<const std::vector<uint32_t>>(std::move(spirv.second));
    SpecializationConstant specialization{};
    specialization.id = 7;
    specialization.value.ui = 3;
    shader.specializationConstants.push_back(specialization);

    DispatchComputeData dispatch{ShaderId{0}};
    dispatch.debugName = "multiply";
    dispatch.computeDispatch = {2, 2, 1, "multiply"};
    // Descriptor locations and SR resource IDs are independent of public indices.
    dispatch.bindings = {{1, 9, output, std::nullopt, vk::DescriptorType::eStorageBuffer},
                         {1, 3, input, std::nullopt, vk::DescriptorType::eStorageBuffer}};
    auto execution = prepareComputeExecution(std::make_shared<WorkloadRuntime>(context), resources, shader, dispatch);
    ASSERT_TRUE(execution->isPrepared());

    Compute compute(context);
    compute.setupQueryPool(2);
    compute.registerWriteTimestamp(0, vk::PipelineStageFlagBits2::eComputeShader);
    compute.registerWorkloadExecution(execution, dispatch.debugName, {"ComputeDispatch", "multiply"}, true);
    compute.registerWriteTimestamp(1, vk::PipelineStageFlagBits2::eComputeShader);
    compute.submitAndWaitOnFence();

    const auto result = resources.getBuffer(output).download(context);
    const std::array<uint32_t, 4> expected{3, 6, 9, 12};
    ASSERT_EQ(result.data.size(), sizeof(expected));
    EXPECT_EQ(std::memcmp(result.data.data(), expected.data(), sizeof(expected)), 0);

    const auto profile = compute.getRuntimeProfilingData();
    ASSERT_EQ(profile.commands.size(), 1u);
    EXPECT_EQ(profile.commands.front().type, "ComputeDispatch");
    EXPECT_EQ(profile.commands.front().name, "multiply");
    EXPECT_EQ(profile.timestamps.size(), 2u);
}

TEST(ComputeWorkload, SamplesRepeatedImageInTrackedLayouts) {
    const std::string source = R"(
#version 450
layout(local_size_x = 1) in;
layout(binding = 0) uniform sampler2D inputImage;
layout(set = 1, binding = 3) uniform sampler2D repeatedImage;
layout(binding = 1) writeonly buffer Output { vec4 result; };
void main() {
    result = texelFetch(inputImage, ivec2(0), 0) + 2 * texelFetch(repeatedImage, ivec2(0), 0);
}
)";
    auto spirv = GlslCompiler::get().compile(source, ShaderStage::Compute);
    ASSERT_TRUE(spirv.first.empty()) << spirv.first;
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    DataManager resources;
    const ImageId input{0};
    ImageInfo info{};
    info.debugName = "sampled_input";
    info.shape = {1, 1, 1, 1};
    info.format = vk::Format::eR8G8B8A8Unorm;
    info.targetFormat = info.format;
    info.isInput = true;
    info.isSampled = true;
    info.mips = 1;
    resources.createImage(input, info);
    auto &image = resources.getImageMut(input);
    image.setup(context);
    image.allocateMemory(context);
    const std::array<uint8_t, 4> pixel{255, 0, 255, 255};
    image.upload(context, {pixel.data(), sizeof(pixel), info.shape, info.format});

    const std::array<float, 4> expected{3, 0, 3, 3};
    const BufferId output{0};
    resources.createBuffer(output, BufferInfo{"output", sizeof(expected), 0});
    auto &buffer = resources.getBufferMut(output);
    buffer.setup(context);
    buffer.allocateMemory(context);
    ShaderInfo shader;
    shader.entry = "main";
    shader.stage = ShaderStage::Compute;
    shader.src = std::make_shared<const std::vector<uint32_t>>(std::move(spirv.second));
    DispatchComputeData dispatch{ShaderId{0}};
    dispatch.bindings = {{0, 0, input, std::nullopt, vk::DescriptorType::eCombinedImageSampler},
                         {0, 1, output, std::nullopt, vk::DescriptorType::eStorageBuffer},
                         {1, 3, input, std::nullopt, vk::DescriptorType::eCombinedImageSampler}};

    for (const auto layout : {vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral}) {
        SCOPED_TRACE(vk::to_string(layout));
        image.transitionLayout(context, layout);
        auto execution =
            prepareComputeExecution(std::make_shared<WorkloadRuntime>(context), resources, shader, dispatch);
        ASSERT_EQ(execution->workload().resourceCount(), 3u);
        ASSERT_EQ(execution->workload().resource(0).requirements().asImage().requiredLayout(), layout);
        EXPECT_EQ(execution->workload().resource(2).requirements().asImage().requiredLayout(), layout);
        Compute compute(context);
        compute.registerWorkloadExecution(execution, "sample", {"ComputeDispatch", "sample"}, true);
        for (int repeat = 0; repeat < 2; ++repeat) {
            compute.reset();
            buffer.fillZero(context);
            compute.submitAndWaitOnFence();
            const auto result = buffer.download(context);
            ASSERT_EQ(result.data.size(), sizeof(expected));
            std::array<float, 4> actual{};
            std::memcpy(actual.data(), result.data.data(), sizeof(actual));
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_FLOAT_EQ(actual[i], expected[i]);
            }
            EXPECT_EQ(image.getImageLayout(), layout);
        }
    }
}

TEST(ComputeWorkload, ValidatesSuppliedPushDataWithoutPushRange) {
    TempFolder folder("compute_zero_push");
    const auto invalidPath = folder.relative("invalid.npy").string();
    std::ofstream(invalidPath, std::ios::binary) << "not a NumPy file";
    const auto validPath = folder.relative("valid.npy").string();
    const uint32_t value = 7;
    mlsdk::vgfutils::numpy::write(validPath, reinterpret_cast<const char *>(&value), {1}, 'u', sizeof(value));

    auto spirv = GlslCompiler::get().compile("#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}\n",
                                             ShaderStage::Compute);
    ASSERT_TRUE(spirv.first.empty()) << spirv.first;
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    const auto runtime = std::make_shared<WorkloadRuntime>(context);
    DataManager resources;
    resources.createRawData(RawDataId{0}, {"invalid_push", invalidPath});
    resources.createRawData(RawDataId{1}, {"valid_push", validPath});
    ShaderInfo shader;
    shader.entry = "main";
    shader.stage = ShaderStage::Compute;
    shader.src = std::make_shared<const std::vector<uint32_t>>(std::move(spirv.second));
    DispatchComputeData dispatch{ShaderId{0}};
    dispatch.pushData = RawDataId{0};
    EXPECT_THROW(prepareComputeExecution(runtime, resources, shader, dispatch), std::runtime_error);
    // A valid payload is parsed but not uploaded when the shader has no range.
    dispatch.pushData = RawDataId{1};
    EXPECT_NO_THROW(prepareComputeExecution(runtime, resources, shader, dispatch));
}

TEST(ComputeWorkload, RecordsShortPushPayloadWithinDeclaredRange) {
    TempFolder folder("compute_partial_push");
    const auto path = folder.relative("push.npy").string();
    const std::array<uint32_t, 7> payload{1, 2, 3, 4, 5, 6, 7};
    mlsdk::vgfutils::numpy::write(path, reinterpret_cast<const char *>(payload.data()), {7}, 'u', sizeof(uint32_t));

    const std::string source = R"(
#version 450
layout(local_size_x = 1) in;
layout(push_constant) uniform Push { uint values[7]; } pushData;
layout(binding = 0) writeonly buffer Output { uint result; };
void main() {
    result = 0;
    for (uint i = 0; i < 7; ++i) result += pushData.values[i];
}
)";
    auto spirv = GlslCompiler::get().compile(source, ShaderStage::Compute);
    ASSERT_TRUE(spirv.first.empty()) << spirv.first;
    Context context{ScenarioOptions{}, detail::DeviceRequirements{}};
    DataManager resources;
    const BufferId output{0};
    resources.createBuffer(output, BufferInfo{"output", sizeof(uint32_t), 0});
    auto &buffer = resources.getBufferMut(output);
    buffer.setup(context);
    buffer.allocateMemory(context);
    buffer.fillZero(context);

    const RawDataId push{0};
    resources.createRawData(push, {"push", path});
    ShaderInfo shader;
    shader.entry = "main";
    shader.stage = ShaderStage::Compute;
    shader.src = std::make_shared<const std::vector<uint32_t>>(std::move(spirv.second));
    shader.pushConstantsSize = 36;
    DispatchComputeData dispatch{ShaderId{0}};
    dispatch.pushData = push;
    dispatch.bindings = {{0, 0, output, std::nullopt, vk::DescriptorType::eStorageBuffer}};
    auto execution = prepareComputeExecution(std::make_shared<WorkloadRuntime>(context), resources, shader, dispatch);
    Compute compute(context);
    compute.registerWorkloadExecution(execution, "partial_push", {"ComputeDispatch", "partial_push"}, true);
    compute.submitAndWaitOnFence();
    const auto result = buffer.download(context);
    const uint32_t expected = 28;
    ASSERT_EQ(result.data.size(), sizeof(expected));
    EXPECT_EQ(std::memcmp(result.data.data(), &expected, sizeof(expected)), 0);
}

} // namespace
