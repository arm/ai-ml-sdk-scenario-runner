/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"
#include "samples.hpp"

#include "scenario_runner/scenario_builder.hpp"

#include <exception>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

int mlsdk::scenariorunner::samples::runComputeSample(std::string_view executable) {
    using namespace mlsdk::scenariorunner;
    using namespace mlsdk::scenariorunner::samples;

    try {
        // API documentation: programmatic scenario example begins.
        // Register the shader and all resources before defining the dispatch.
        auto builder = createScenarioBuilder();

        ShaderInfo shaderInfo{};
        shaderInfo.debugName = "increment";
        shaderInfo.entry = "main";
        shaderInfo.shaderType = ShaderType::GLSL;
        shaderInfo.stage = ShaderStage::Compute;
        shaderInfo.src = readShaderCode(assetPath(executable, "increment.comp").string(), shaderInfo);
        const auto shaderId = builder->addShader(shaderInfo);

        const auto inputId = builder->addBuffer(BufferInfo{"input", 16});
        const auto outputId = builder->addBuffer(BufferInfo{"output", 16});

        DispatchComputeData dispatch{shaderId};
        dispatch.debugName = "increment";
        dispatch.bindings = {{0, 0, inputId, std::nullopt, vk::DescriptorType::eStorageBuffer},
                             {0, 1, outputId, std::nullopt, vk::DescriptorType::eStorageBuffer}};
        dispatch.computeDispatch.gwcx = 4;
        dispatch.computeDispatch.profileName = dispatch.debugName;
        builder->addDispatchCompute(std::move(dispatch));

        // The returned IDs remain valid for uploading inputs and downloading outputs.
        auto scenario = builder->build(ScenarioOptions{});
        const std::vector<uint32_t> input{1, 2, 3, 4};
        const std::vector<uint32_t> expected{2, 3, 4, 5};
        scenario->upload(inputId, view(input));
        scenario->run();
        const auto output = scenario->download(outputId);
        // API documentation: programmatic scenario example ends.

        if (!matches(output, expected)) {
            std::cerr << "Unexpected compute output\n";
            return 1;
        }
        std::cout << "Compute output matched the expected values\n";
    } catch (const std::exception &error) {
        std::cerr << "Failed to run compute sample: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
