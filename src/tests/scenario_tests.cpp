/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "glsl_compiler.hpp"
#include "resource_data.hpp"
#include "scenario.hpp"
#include "scenario_desc.hpp"
#include "scenario_options.hpp"
#include "shader_stage.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "vgf-utils/temp_folder.hpp"

using namespace mlsdk::scenariorunner;

namespace {
const std::string scenarioJson = R"(
        {
            "commands": [],
            "resources": [
                {
                    "buffer": {
                        "uid": "inBuffer",
                        "size": 4,
                        "shader_access": "readwrite"
                    }
                },
                {
                    "tensor": {
                        "uid": "inTensor",
                        "dims": [1, 2, 2, 1],
                        "format": "VK_FORMAT_R8_SINT",
                        "shader_access": "readwrite"
                    }
                }
            ]
        }
    )";

template <typename T> std::vector<char> asBytes(const std::vector<T> &values) {
    std::vector<char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}
} // namespace

TEST(IScenario, ScenarioSupportsVirtualDispatch) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};
    IScenario &api = scenario;

    const auto bufferId = api.getBufferId("inBuffer");
    const std::vector<char> payload{1, 2, 3, 4};
    api.upload(bufferId, {payload.data(), payload.size()});
    api.run();

    EXPECT_EQ(api.download(bufferId).data, payload);
    EXPECT_EQ(api.getTensorId("inTensor"), scenario.getTensorId("inTensor"));
}

TEST(ScenarioInMemoryTransfer, UploadsAndDownloadsByTypedIdAcrossRuns) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};

    const auto bufferId = scenario.getBufferId("inBuffer");
    const auto tensorId = scenario.getTensorId("inTensor");

    std::vector<char> firstBuffer{1, 2, 3, 4};
    std::vector<char> firstTensor{5, 6, 7, 8};
    scenario.upload(bufferId, {firstBuffer.data(), firstBuffer.size()});
    scenario.upload(tensorId, {firstTensor.data(), firstTensor.size(), {1, 2, 2, 1}, vk::Format::eR8Sint});
    scenario.run();

    EXPECT_EQ(scenario.download(bufferId).data, firstBuffer);
    EXPECT_EQ(scenario.download(tensorId).data, firstTensor);

    std::vector<char> secondBuffer{9, 10, 11, 12};
    std::vector<char> secondTensor{13, 14, 15, 16};
    scenario.upload(bufferId, {secondBuffer.data(), secondBuffer.size()});
    scenario.upload(tensorId, {secondTensor.data(), secondTensor.size(), {1, 2, 2, 1}, vk::Format::eR8Sint});
    scenario.run();

    EXPECT_EQ(scenario.download(bufferId).data, secondBuffer);
    EXPECT_EQ(scenario.download(tensorId).data, secondTensor);
}

TEST(ScenarioInMemoryTransfer, InitializesResourcesWithoutSourcesThroughTypedUploadPath) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};

    const auto buffer = scenario.download(scenario.getBufferId("inBuffer"));
    const auto tensor = scenario.download(scenario.getTensorId("inTensor"));

    EXPECT_EQ(buffer.data, std::vector<char>(4, 0));
    EXPECT_EQ(tensor.data, std::vector<char>(4, 0));
    EXPECT_EQ(tensor.shape, (std::vector<int64_t>{1, 2, 2, 1}));
    ASSERT_TRUE(tensor.format.has_value());
    EXPECT_EQ(tensor.format.value(), vk::Format::eR8Sint);
}

TEST(ScenarioInMemoryTransfer, RejectsNonPositiveRepeatCount) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};

    try {
        scenario.run(0);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument &error) {
        EXPECT_STREQ(error.what(), "Scenario repeat count must be greater than zero; received 0.");
    }
}

TEST(ScenarioInMemoryTransfer, RejectsUnknownTypedIds) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};

    const std::vector<char> data(4);
    const auto expectError = [](const auto &operation, const char *expectedMessage) {
        try {
            operation();
            FAIL() << "Expected std::runtime_error";
        } catch (const std::runtime_error &error) {
            EXPECT_STREQ(error.what(), expectedMessage);
        }
    };

    expectError([&] { scenario.upload(BufferId{1}, {data.data(), data.size()}); },
                "Scenario::upload: Buffer resource not found.");
    expectError([&] { scenario.upload(TensorId{1}, {data.data(), data.size(), {1, 2, 2, 1}, vk::Format::eR8Sint}); },
                "Scenario::upload: Tensor resource not found.");
    expectError([&] { static_cast<void>(scenario.download(BufferId{1})); },
                "Scenario::download: Buffer resource not found.");
    expectError([&] { static_cast<void>(scenario.download(TensorId{1})); },
                "Scenario::download: Tensor resource not found.");

    expectError([&] { static_cast<void>(scenario.getBufferId("missing")); },
                "Scenario::getBufferId: resource UID 'missing' not found.");
    expectError([&] { static_cast<void>(scenario.getTensorId("missing")); },
                "Scenario::getTensorId: resource UID 'missing' not found.");
    expectError([&] { static_cast<void>(scenario.getBufferId("inTensor")); },
                "Scenario::getBufferId: resource UID 'inTensor' does not identify a Buffer resource.");
    expectError([&] { static_cast<void>(scenario.getTensorId("inBuffer")); },
                "Scenario::getTensorId: resource UID 'inBuffer' does not identify a Tensor resource.");
}

TEST(IScenario, SupportsTensorUploadAndDownload) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};
    IScenario &api = scenario;

    const auto tensorId = api.getTensorId("inTensor");
    const std::vector<char> payload{1, 2, 3, 4};
    api.upload(tensorId, {payload.data(), payload.size(), {1, 2, 2, 1}, vk::Format::eR8Sint});

    const auto result = api.download(tensorId);
    EXPECT_EQ(result.data, payload);
    EXPECT_EQ(result.shape, (std::vector<int64_t>{1, 2, 2, 1}));
    ASSERT_TRUE(result.format.has_value());
    EXPECT_EQ(result.format.value(), vk::Format::eR8Sint);
}

TEST(IScenario, SupportsRepeatedUploadRunDownload) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};
    IScenario &api = scenario;
    const auto bufferId = api.getBufferId("inBuffer");

    const std::vector<char> firstPayload{1, 2, 3, 4};
    api.upload(bufferId, {firstPayload.data(), firstPayload.size()});
    api.run();
    EXPECT_EQ(api.download(bufferId).data, firstPayload);

    const std::vector<char> secondPayload{5, 6, 7, 8};
    api.upload(bufferId, {secondPayload.data(), secondPayload.size()});
    api.run();
    EXPECT_EQ(api.download(bufferId).data, secondPayload);
}

TEST(IScenario, ExecutesCommandWithDifferentInputsAcrossRuns) {
    constexpr std::string_view shaderSource = R"(
        #version 450
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(set = 0, binding = 0) readonly buffer Input { uint values[]; } inputBuffer;
        layout(set = 0, binding = 1) writeonly buffer Output { uint values[]; } outputBuffer;
        void main() {
            outputBuffer.values[gl_GlobalInvocationID.x] = inputBuffer.values[gl_GlobalInvocationID.x] + 1;
        }
    )";
    constexpr std::string_view computeScenarioJson = R"(
        {
            "commands": [
                {
                    "dispatch_compute": {
                        "bindings": [
                            {"id": 0, "set": 0, "resource_ref": "input"},
                            {"id": 1, "set": 0, "resource_ref": "output"}
                        ],
                        "rangeND": [4],
                        "shader_ref": "increment"
                    }
                }
            ],
            "resources": [
                {"shader": {"src": "increment.spv", "type": "SPIR-V", "uid": "increment"}},
                {"buffer": {"uid": "input", "size": 16, "shader_access": "readonly"}},
                {"buffer": {"uid": "output", "size": 16, "shader_access": "readwrite"}}
            ]
        }
    )";

    TempFolder tempFolder("iscenario_compute_test");
    const auto shaderPath = tempFolder.relative("increment.spv");
    const auto spirv = GlslCompiler::get().compile(std::string{shaderSource}, ShaderStage::Compute);
    ASSERT_TRUE(spirv.first.empty()) << spirv.first;
    ASSERT_TRUE(GlslCompiler::get().save(spirv.second, shaderPath.string()));

    ScenarioSpec spec{std::string{computeScenarioJson}, shaderPath.parent_path()};
    Scenario scenario{ScenarioOptions{}, spec};
    IScenario &api = scenario;
    const auto inputId = api.getBufferId("input");
    const auto outputId = api.getBufferId("output");

    const auto firstInput = asBytes(std::vector<uint32_t>{1, 2, 3, 4});
    api.upload(inputId, {firstInput.data(), firstInput.size()});
    api.run();
    EXPECT_EQ(api.download(outputId).data, asBytes(std::vector<uint32_t>{2, 3, 4, 5}));

    const auto secondInput = asBytes(std::vector<uint32_t>{10, 20, 30, 40});
    api.upload(inputId, {secondInput.data(), secondInput.size()});
    api.run();
    EXPECT_EQ(api.download(outputId).data, asBytes(std::vector<uint32_t>{11, 21, 31, 41}));
}

TEST(IScenario, RejectsUnknownTypedIds) {
    ScenarioSpec spec{scenarioJson};
    spec.useComputeFamilyQueue = true;
    Scenario scenario{ScenarioOptions{}, spec};
    IScenario &api = scenario;
    const std::vector<char> payload(4);

    EXPECT_THROW(api.upload(BufferId{1}, {payload.data(), payload.size()}), std::runtime_error);
    EXPECT_THROW(api.upload(TensorId{1}, {payload.data(), payload.size(), {1, 2, 2, 1}, vk::Format::eR8Sint}),
                 std::runtime_error);
    EXPECT_THROW(api.upload(ImageId{0}, {}), std::runtime_error);
    EXPECT_THROW(static_cast<void>(api.download(BufferId{1})), std::runtime_error);
    EXPECT_THROW(static_cast<void>(api.download(TensorId{1})), std::runtime_error);
    EXPECT_THROW(static_cast<void>(api.download(ImageId{0})), std::runtime_error);
}
