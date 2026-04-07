/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <argparse/argparse.hpp>

#include "glsl_compiler.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>

using namespace mlsdk::scenariorunner;

int main(int argc, const char **argv) {

    try {
        argparse::ArgumentParser parser(argv[0]);

        parser.add_argument("--input").help("the GLSL file to be compiled to SPIR-V").required();
        parser.add_argument("--output").help("the SPIR-V output file").required();
        parser.add_argument("--build-opts").help("list of preprocessor defines to be used for compilation");
        parser.add_argument("--include").help("list of shader include directories");
        parser.add_argument("--stage")
            .help("shader stage to compile (compute, vertex, fragment)")
            .default_value(std::string("compute"));

        parser.parse_args(argc, argv);
        std::string input = parser.get("--input");
        std::string output = parser.get("--output");
        std::string opts = parser.present("--build-opts").value_or("");
        std::vector<std::string> includes;
        if (auto include = parser.present("--include")) {
            includes.push_back(*include);
        }

        std::string glsl;
        if (!GlslCompiler::get().load(input, glsl)) {
            std::cerr << "Failed to load input file" << std::endl;
            return -1;
        }

        auto stageStr = parser.get<std::string>("--stage");
        std::transform(stageStr.begin(), stageStr.end(), stageStr.begin(), ::tolower);
        auto toStage = [](std::string_view stage) -> ShaderStage {
            if (stage == "compute") {
                return ShaderStage::Compute;
            }
            if (stage == "vertex") {
                return ShaderStage::Vertex;
            }
            if (stage == "fragment") {
                return ShaderStage::Fragment;
            }
            throw std::runtime_error("Unsupported shader stage: " + std::string(stage));
        };

        auto spirv = GlslCompiler::get().compile(glsl, toStage(stageStr), opts, includes);
        if (!spirv.first.empty()) {
            std::cerr << "Failed to compiled GLSL input to SPIR-V:\n" << spirv.first << std::endl;
            return -1;
        }

        if (!GlslCompiler::get().save(spirv.second, output)) {
            std::cerr << "Failed to save compiled output" << std::endl;
            return -1;
        }
    } catch (const std::exception &error) {
        std::cerr << "[ERROR]:" << error.what() << "\n";
        return -1;
    }
    return 0;
}
