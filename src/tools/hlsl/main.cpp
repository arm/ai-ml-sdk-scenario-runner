/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <argparse/argparse.hpp>

#include "hlsl_compiler.hpp"

#include <iostream>

using namespace mlsdk::scenariorunner;

int main(int argc, const char **argv) {

    try {
        argparse::ArgumentParser parser(argv[0]);

        parser.add_argument("--input").help("the HLSL file to be compiled to SPIR-V").required();
        parser.add_argument("--output").help("the SPIR-V output file").required();
        parser.add_argument("--entry").help("the entry point of the HLSL shader");
        parser.add_argument("--debug-name").help("the name of the shader");
        parser.add_argument("--build-opts").help("list of preprocessor defines to be used for compilation");
        parser.add_argument("--include").help("list of shader include directories");

        parser.parse_args(argc, argv);
        std::string input = parser.get("--input");
        std::string output = parser.get("--output");
        std::string entry = parser.present("--entry").value_or("main");
        std::string name = parser.present("--debug-name").value_or("");
        std::string opts = parser.present("--build-opts").value_or("");
        std::vector<std::string> includes;
        if (auto include = parser.present("--include")) {
            includes.push_back(*include);
        }

        std::string hlsl;
        if (!HlslCompiler::get().load(input, hlsl)) {
            std::cerr << "Failed to load input file" << std::endl;
            return -1;
        }

        auto spirv = HlslCompiler::get().compile(hlsl, entry, name, opts, includes);
        if (!spirv.first.empty()) {
            std::cerr << "Failed to compiled HLSL input to SPIR-V:\n" << spirv.first << std::endl;
            return -1;
        }

        if (!HlslCompiler::get().save(spirv.second, output)) {
            std::cerr << "Failed to save compiled output" << std::endl;
            return -1;
        }
    } catch (const std::exception &error) {
        std::cerr << "[ERROR]:" << error.what() << "\n";
        return -1;
    }
    return 0;
}
