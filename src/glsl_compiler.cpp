/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "glsl_compiler.hpp"

#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/DirStackFileIncluder.h>
#include <glslang/Public/ShaderLang.h>

#include <cstring>
#include <fstream>

extern const TBuiltInResource *GetDefaultResources();

namespace mlsdk::scenariorunner {
namespace {
std::string parsePreprocessorOptions(const std::string &options) {
    std::string preamble = "";
    const std::string delimiter = " ";
    size_t start = 0;
    while (start < options.size()) {
        preamble.append("#define ");
        size_t end = options.find(delimiter, start);
        if (end == std::string::npos) {
            end = options.size();
        }
        std::string op = options.substr(start + 2, end - (start + 2)); // +2 skips the -D
        // Turn the first "=" into a space
        const size_t equal = op.find_first_of("=");
        if (equal != std::string::npos) {
            op[equal] = ' ';
        }
        preamble.append(op);
        start = end + 1;
        preamble.append("\n");
    }
    return preamble;
}
} // namespace

GlslCompiler::GlslCompiler() { glslang::InitializeProcess(); }

GlslCompiler::~GlslCompiler() { glslang::FinalizeProcess(); }

GlslCompiler &GlslCompiler::get() {
    static GlslCompiler glslCompiler;
    return glslCompiler;
}

std::pair<std::string, std::vector<uint32_t>> GlslCompiler::compile(const std::string &source,
                                                                    const std::string &preprocessorOptions,
                                                                    const std::vector<std::string> &shaderDirs) {
    std::string log;
    const EShLanguage language = EShLanguage::EShLangCompute;
    const EShMessages messages = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

    glslang::TShader shader(language);
    shader.setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClientVulkan, 110);
    shader.setEnvClient(glslang::EShClient::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
    const char *cSource = source.c_str();
    shader.setStrings(&cSource, 1);

    std::string msg;

    DirStackFileIncluder includer;
    for (const auto &dir : shaderDirs) {
        includer.pushExternalLocalDirectory(dir);
    };

    const std::string userPreamble = parsePreprocessorOptions(preprocessorOptions);
    shader.setPreamble(userPreamble.c_str());

    if (!shader.preprocess(GetDefaultResources(), 460, ENoProfile, false, false, messages, &msg, includer)) {
        log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
        return {log, {}};
    }

    if (!shader.parse(GetDefaultResources(), 460, false, messages, includer)) {
        log += std::string(shader.getInfoLog()) + "\n";
        log += std::string(shader.getInfoDebugLog()) + "\n";
        return {log, {}};
    }

    // Attach shader to a program
    glslang::TProgram program;
    program.addShader(&shader);

    // Link program
    if (!program.link(messages)) {
        log += std::string(program.getInfoLog()) + "\n";
        log += std::string(program.getInfoDebugLog()) + "\n";
        return {log, {}};
    }

    // Translate to SPIR-V
    glslang::TIntermediate *intermediate = program.getIntermediate(language);
    if (!intermediate) {
        log += std::string("[ERROR] Could not extract intermediate code\n");
        return {log, {}};
    }
    glslang::SpvOptions spvOpts;
    spvOpts.generateDebugInfo = true;

    spv::SpvBuildLogger spvLogger;
    std::vector<std::uint32_t> spv;
    glslang::GlslangToSpv(*intermediate, spv, &spvLogger, &spvOpts);

    const std::string spvLoggerMessages = spvLogger.getAllMessages();
    if (!spvLoggerMessages.empty()) {
        log += spvLoggerMessages;
    }

    return {log, spv};
}

bool GlslCompiler::load(const std::string &fname, std::string &glsl) {
    std::ifstream ifile{fname, std::ios::in | std::ios::binary};
    if (!ifile.is_open()) {
        return false;
    }

    glsl = std::string((std::istreambuf_iterator<char>(ifile)), (std::istreambuf_iterator<char>()));
    if (ifile.bad()) {
        return false;
    }

    return true;
}

bool GlslCompiler::save(const std::vector<uint32_t> &mod, const std::string &fname) {
    std::ofstream ofile;

    ofile.open(fname, std::ios::out | std::ios::binary);

    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(reinterpret_cast<const char *>(mod.data()), std::streamsize(mod.size() * sizeof(uint32_t)));
    return ofile.good();
}
} // namespace mlsdk::scenariorunner
