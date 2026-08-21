/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"
#include "samples.hpp"

#include "scenario_runner/scenario_builder.hpp"

#include <vgf/encoder.hpp>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace mlsdk;
using namespace mlsdk::scenariorunner;

constexpr uint16_t VULKAN_HEADER_VERSION = 123;
constexpr vgflib::DescriptorType STORAGE_BUFFER_DESCRIPTOR_TYPE = 6;
constexpr vgflib::FormatType R32_UINT_FORMAT = 98;

void writeIncrementVgf(const std::filesystem::path &path) {
    auto encoder = vgflib::CreateEncoder(VULKAN_HEADER_VERSION);
    const auto module = encoder->AddModule(vgflib::ModuleType::COMPUTE, "increment", "main");
    const auto input = encoder->AddInputResource(STORAGE_BUFFER_DESCRIPTOR_TYPE, R32_UINT_FORMAT, {4}, {});
    const auto output = encoder->AddOutputResource(STORAGE_BUFFER_DESCRIPTOR_TYPE, R32_UINT_FORMAT, {4}, {});
    const auto inputBinding = encoder->AddBindingSlot(0, input);
    const auto outputBinding = encoder->AddBindingSlot(1, output);
    const auto descriptorSet = encoder->AddDescriptorSetInfo({inputBinding, outputBinding});

    encoder->AddModelSequenceInputsOutputs({inputBinding}, {"input"}, {outputBinding}, {"output"});
    encoder->AddSegmentInfo(module, "increment_segment", {descriptorSet}, {inputBinding}, {outputBinding},
                            std::vector<vgflib::GraphConstantBindingRef>{}, {4, 1, 1});
    encoder->Finish();

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !encoder->WriteTo(stream)) {
        throw std::runtime_error("Failed to write sample VGF file");
    }
}

} // namespace

int mlsdk::scenariorunner::samples::runVgfInferenceSample(std::string_view executable) {
    using namespace mlsdk::scenariorunner::samples;

    try {
        const auto outputDir = std::filesystem::temp_directory_path();
        const auto vgfPath = outputDir / "scenario_runner_increment.vgf";
        const auto shaderPath = assetPath(executable, "increment.comp");
        const auto profilingPath = outputDir / "scenario_runner_vgf_inference_profiling.json";
        writeIncrementVgf(vgfPath);
        std::filesystem::remove(profilingPath);

        // Production VGF files normally come from ML SDK Model Converter. This
        // sample generates a minimal graph so it can be built and run independently.
        auto builder = createScenarioBuilder();
        VgfInfo vgfInfo{};
        vgfInfo.debugName = "increment graph";
        vgfInfo.src = loadVgfView(vgfPath.string());
        const auto vgf = builder->addVgf(vgfInfo);

        ShaderInfo shaderInfo{};
        shaderInfo.debugName = "increment";
        shaderInfo.entry = "main";
        shaderInfo.shaderType = ShaderType::GLSL;
        shaderInfo.stage = ShaderStage::Compute;
        shaderInfo.src = readShaderCode(shaderPath.string(), shaderInfo);
        const auto shader = builder->addShader(shaderInfo);

        const auto input = builder->addBuffer(BufferInfo{"input", 16});
        const auto output = builder->addBuffer(BufferInfo{"output", 16});

        DispatchVgfData dispatch{vgf};
        dispatch.debugName = "increment inference";
        dispatch.bindings = {{0, 0, input, std::nullopt, vk::DescriptorType::eStorageBuffer},
                             {0, 1, output, std::nullopt, vk::DescriptorType::eStorageBuffer}};
        dispatch.shaderSubstitutions = {{shader, "increment"}};
        builder->addDispatchVgf(std::move(dispatch));

        ScenarioOptions options{};
        options.profilingPath = profilingPath;
        auto scenario = builder->build(options);

        const std::vector<uint32_t> inputValues{1, 2, 3, 4};
        const std::vector<uint32_t> expectedValues{2, 3, 4, 5};
        scenario->upload(input, view(inputValues));
        scenario->run();

        if (!matches(scenario->download(output), expectedValues)) {
            std::cerr << "Unexpected VGF inference output\n";
            return 1;
        }
        if (!std::filesystem::is_regular_file(profilingPath) || std::filesystem::file_size(profilingPath) == 0) {
            std::cerr << "Profiling output was not created\n";
            return 1;
        }

        std::cout << "VGF inference output matched the expected values\n";
        std::cout << "Profiling data written to " << profilingPath << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Failed to run VGF inference sample: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
