/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include "commands.hpp"
#include "compute.hpp"
#include "glsl_compiler.hpp"
#include "pipeline.hpp"
#include "scenario.hpp"

#include <vector>

#include "temp_folder.hpp"

namespace mlsdk::scenariorunner {

constexpr float epsilon = 0.0001f;

const std::string add_shader =
    R""(
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer in1 { float In1Buffer[10]; };
layout(binding = 1) buffer in2 { float In2Buffer[10]; };
layout(binding = 2) buffer out1 { float OutBuffer[10]; };

void main()
{
    OutBuffer[gl_GlobalInvocationID.x] = In1Buffer[gl_GlobalInvocationID.x] + In2Buffer[gl_GlobalInvocationID.x];
}
)"";

// Test the initialization of a Vulkan® Compute pipeline by running a simple
// shader. Here we do the following:
//      1. Instantiate a shader runner that creates a Vulkan® Compute pipeline.
//      2. Create and initialize input buffers.
//      3. Run the shader.
//      4. Check that the output buffer matches the expected values.
TEST(VulkanStartUp, RunShader) { // cppcheck-suppress syntaxError
    TempFolder tempFolder("scenario_runner_start_up_tests");

    constexpr uint32_t numElements = 10;

    // Compile compute shader to SPIR-V

    std::string addShaderSPIRV = tempFolder.relative("add_shader.spv").string();
    auto spirv = GlslCompiler::get().compile(add_shader);
    EXPECT_TRUE(spirv.first.empty());
    GlslCompiler::get().save(spirv.second, addShaderSPIRV);

    Context ctx{{}};

    DataManager dataManager;
    std::vector<float> inDataA(numElements);
    std::vector<float> inDataB(numElements);
    std::vector<float> outDataAdd(numElements, 0.f);
    std::vector<float> expectedOutput(numElements);
    for (uint32_t i = 0; i < numElements; ++i) {
        inDataA[i] = static_cast<float>(i);
        inDataB[i] = static_cast<float>(i + 1);
        expectedOutput[i] = inDataA[i] + inDataB[i];
    }

    BufferInfo info;
    std::vector<char> data;
    auto guidA = Guid("inBufferA");
    auto guidB = Guid("inBufferB");
    auto guidOut = Guid("outBufferAdd");

    const auto prepareBuffer = [&ctx, &dataManager](Guid guid, const std::vector<char> &values) {
        auto &buffer = dataManager.getBufferMut(guid);
        buffer.setup(ctx);
        buffer.allocateMemory(ctx);
        buffer.fill(values.data(), values.size());
    };
    info.size = numElements * sizeof(float);
    data.resize(info.size);
    std::memcpy(data.data(), inDataA.data(), info.size);
    dataManager.createBuffer(guidA, info);
    prepareBuffer(guidA, data);
    std::memcpy(data.data(), inDataB.data(), info.size);
    dataManager.createBuffer(guidB, info);
    prepareBuffer(guidB, data);
    std::memset(data.data(), 0, info.size);
    dataManager.createBuffer(guidOut, info);
    prepareBuffer(guidOut, data);

    std::vector<TypedBinding> bindings;
    TypedBinding binding;
    binding.set = 0;
    binding.id = 0;
    binding.resourceRef = guidA;
    binding.vkDescriptorType = vk::DescriptorType::eStorageBuffer;
    bindings.push_back(binding);
    binding.id = 1;
    binding.resourceRef = guidB;
    bindings.push_back(binding);
    binding.id = 2;
    binding.resourceRef = guidOut;
    bindings.push_back(binding);

    // Create compute pipeline
    std::optional<PipelineCache> pipelineCache{};
    const Pipeline::CommonArguments args{ctx, "test_pipeline", bindings, pipelineCache};
    ShaderDesc shaderDesc(Guid("add_shader"), "add_shader", addShaderSPIRV, "main", ShaderType::SPIR_V);
    Pipeline pipe(args, shaderDesc);

    // Create compute orchestrator to run commands
    Compute compute(ctx);
    bool implicitBarriers = true;
    compute.registerPipelineFenced(pipe, dataManager, bindings, nullptr, 0, implicitBarriers, {numElements, 1, 1});

    // Run and wait on fence
    compute.submitAndWaitOnFence();

    // Retrieve results
    auto &outputBuf = dataManager.getBufferMut(guidOut);
    float *outputPtr = static_cast<float *>(outputBuf.map());

    const std::vector<float> output(outputPtr, outputPtr + numElements);
    for (uint32_t i = 0; i < numElements; ++i) {
        EXPECT_NEAR(expectedOutput[i], output[i], epsilon);
    }
    outputBuf.unmap();
}
} // namespace mlsdk::scenariorunner
