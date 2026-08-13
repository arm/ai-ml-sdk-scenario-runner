/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include "commands.hpp"
#include "compute.hpp"
#include "glsl_compiler.hpp"
#include "scenario.hpp"

#include <vector>

#include "vgf-utils/temp_folder.hpp"

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
void runShader(const ScenarioOptions &scenarioOptions) {
    TempFolder tempFolder("scenario_runner_start_up_tests");

    constexpr uint32_t numElements = 10;

    // Compile compute shader to SPIR-V

    std::string addShaderSPIRV = tempFolder.relative("add_shader.spv").string();
    auto spirv = GlslCompiler::get().compile(add_shader, ShaderStage::Compute);
    EXPECT_TRUE(spirv.first.empty());
    GlslCompiler::get().save(spirv.second, addShaderSPIRV);

    Context ctx{scenarioOptions};

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
    const BufferId bufferA{0};
    const BufferId bufferB{1};
    const BufferId bufferOut{2};

    const auto prepareBuffer = [&ctx, &dataManager](BufferId id, const std::vector<char> &values) {
        auto &buffer = dataManager.getBufferMut(id);
        buffer.setup(ctx);
        buffer.allocateMemory(ctx);
        buffer.fill(ctx, values.data(), values.size());
    };
    info.size = numElements * sizeof(float);
    data.resize(info.size);
    std::memcpy(data.data(), inDataA.data(), info.size);
    dataManager.createBuffer(bufferA, info);
    prepareBuffer(bufferA, data);
    std::memcpy(data.data(), inDataB.data(), info.size);
    dataManager.createBuffer(bufferB, info);
    prepareBuffer(bufferB, data);
    std::memset(data.data(), 0, info.size);
    dataManager.createBuffer(bufferOut, info);
    prepareBuffer(bufferOut, data);

    const std::vector<TypedBinding> bindings{
        {0, 0, bufferA, std::nullopt, vk::DescriptorType::eStorageBuffer},
        {0, 1, bufferB, std::nullopt, vk::DescriptorType::eStorageBuffer},
        {0, 2, bufferOut, std::nullopt, vk::DescriptorType::eStorageBuffer},
    };

    // Create compute orchestrator to run commands
    Compute compute(ctx);

    // Create compute pipeline
    const Compute::PipelineCreateArguments args{"test_pipeline", bindings, nullptr};
    ShaderInfo shaderInfo;
    shaderInfo.debugName = "add_shader";
    shaderInfo.src = addShaderSPIRV;
    shaderInfo.entry = "main";
    shaderInfo.shaderType = ShaderType::SPIR_V;
    compute.createPipeline(args, shaderInfo);
    bool implicitBarriers = true;
    compute.registerPipelineFenced(dataManager, bindings, nullptr, 0, implicitBarriers,
                                   {numElements, 1, 1, "test_pipeline"});

    // Run and wait on fence
    compute.submitAndWaitOnFence();

    // Retrieve results
    auto &outputBuf = dataManager.getBufferMut(bufferOut);
    outputBuf.memoryManager()->downloadData(ctx, 0, info.size);
    float *outputPtr = static_cast<float *>(outputBuf.memoryManager()->mapStagingBufferMemory(0, info.size));

    const std::vector<float> output(outputPtr, outputPtr + numElements);
    for (uint32_t i = 0; i < numElements; ++i) {
        EXPECT_NEAR(expectedOutput[i], output[i], epsilon);
    }
    outputBuf.memoryManager()->unmapStagingBufferMemory();
}

TEST(VulkanStartUp, RunShader) { // cppcheck-suppress syntaxError
    runShader({});
}

TEST(VulkanStartUp, RunShaderWithRobustnessFeatures) { // cppcheck-suppress syntaxError
    ScenarioOptions scenarioOptions;
    scenarioOptions.enableRobustnessFeatures = true;
    runShader(scenarioOptions);
}
} // namespace mlsdk::scenariorunner
