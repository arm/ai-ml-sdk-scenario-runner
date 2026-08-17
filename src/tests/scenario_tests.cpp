/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resource_data.hpp"
#include "scenario.hpp"
#include "scenario_desc.hpp"

#include <gtest/gtest.h>

#include <vector>

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
} // namespace

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
