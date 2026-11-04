/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf_configuration.hpp"

#include "data_manager.hpp"
#include "glsl_compiler.hpp"
#ifdef SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT
#    include "hlsl_compiler.hpp"
#endif
#include "logging.hpp"
#include "resource_manager.hpp"
#include "utils.hpp"
#include "workload_execution.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlsdk::scenariorunner {
namespace {

workloadlib::SpecializationInfo specializationInfo(const std::vector<SpecializationConstant> &constants) {
    workloadlib::SpecializationInfo info;
    for (const auto &constant : constants) {
        const auto offset = static_cast<uint32_t>(info.data.size());
        info.mapEntries.emplace_back(static_cast<uint32_t>(constant.id), offset, sizeof(constant.value));
        const auto *bytes = reinterpret_cast<const std::byte *>(&constant.value);
        info.data.insert(info.data.end(), bytes, bytes + sizeof(constant.value));
    }
    return info;
}

const ShaderInfo *substitutionShader(const DispatchVgfData &dispatch, const ResourceManager &resources,
                                     std::string_view moduleName) {
    const auto substitution = std::find_if(dispatch.shaderSubstitutions.begin(), dispatch.shaderSubstitutions.end(),
                                           [&](const auto &candidate) { return candidate.target == moduleName; });
    return substitution == dispatch.shaderSubstitutions.end() ? nullptr : &resources.get(substitution->shader);
}

void bindEmbeddedSource(workloadlib::Session &session, workloadlib::ExecutableView executable, const VgfView &source) {
    const auto module = executable.module();
    std::pair<std::string, std::vector<uint32_t>> compiled;
    if (module.codeKind() == workloadlib::ModuleCodeKind::Glsl) {
        compiled = GlslCompiler::get().compile(source.getGLSLModuleCode(executable.index()), ShaderStage::Compute);
    } else {
#ifdef SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT
        compiled = HlslCompiler::get().compile(source.getHLSLModuleCode(executable.index()),
                                               std::string(module.entryPoint()), std::string(module.name()));
#else
        throw std::runtime_error("HLSL shaders are not supported on this platform.");
#endif
    }
    if (!compiled.first.empty()) {
        throwShaderCompilationError(std::string(module.name()), compiled.first);
    }
    workloadlib::ModuleImplementation implementation;
    implementation.codeKind = workloadlib::ModuleCodeKind::Spirv;
    implementation.spirv = std::move(compiled.second);
    session.bindModule(executable, std::move(implementation));
}

} // namespace

void applyGraphResourceShaderMetadata(ShaderInfo &shader, const VgfInfo &vgf, std::string_view moduleName) {
    if (vgf.pushConstantsSize > 0) {
        shader.pushConstantsSize = vgf.pushConstantsSize;
    }
    const auto map = std::find_if(vgf.specializationConstantMaps.begin(), vgf.specializationConstantMaps.end(),
                                  [&](const auto &candidate) { return candidate.shaderTarget == moduleName; });
    if (map != vgf.specializationConstantMaps.end()) {
        shader.specializationConstants = map->specializationConstants;
    }
}

std::optional<RawDataId> getGraphPushData(const std::vector<ResolvedPushConstantMap> &pushConstants,
                                          std::string_view moduleName) {
    const auto data = std::find_if(pushConstants.begin(), pushConstants.end(),
                                   [&](const auto &candidate) { return candidate.shaderTarget == moduleName; });
    if (data != pushConstants.end()) {
        return data->pushData;
    }
    return std::nullopt;
}

ShaderInfo resolveVgfShaderConfiguration(const VgfInfo &vgf, const DispatchVgfData &dispatch,
                                         const ResourceManager &resources, std::string_view moduleName,
                                         std::string_view entryPoint, bool hasEmbeddedCode) {
    ShaderInfo shader;
    shader.debugName = moduleName;
    shader.entry = entryPoint;
    shader.shaderType = ShaderType::SPIR_V;
    shader.stage = ShaderStage::Compute;
    if (const auto *substitution = substitutionShader(dispatch, resources, moduleName)) {
        shader = *substitution;
        if (!shader.src || shader.src->empty()) {
            throw std::runtime_error("VGF shader substitution has no SPIR-V code: " + shader.debugName);
        }
    } else if (!hasEmbeddedCode) {
        throw std::runtime_error("No shader code present and no shader substitution defined for module '" +
                                 std::string(moduleName) + "'.");
    }
    applyGraphResourceShaderMetadata(shader, vgf, moduleName);
    return shader;
}

void configureVgfExecutables(WorkloadExecution &execution, const VgfInfo &vgf, const DispatchVgfData &dispatch,
                             const ResourceManager &resources) {
    auto &session = execution.session();
    for (const auto executable : execution.workload().executables()) {
        if (executable.type() != workloadlib::ExecutableKind::Compute) {
            continue;
        }
        const auto module = executable.module();
        const auto shader = resolveVgfShaderConfiguration(vgf, dispatch, resources, module.name(), module.entryPoint(),
                                                          !module.requiresImplementation());
        if (shader.src) {
            workloadlib::ModuleImplementation implementation;
            implementation.codeKind = workloadlib::ModuleCodeKind::Spirv;
            implementation.spirv = *shader.src;
            implementation.entryPoint = shader.entry;
            session.bindModule(executable, std::move(implementation));
            if (!module.requiresImplementation()) {
                mlsdk::logging::warning("Performing shader substitution despite shader module containing code");
            }
        } else if (module.codeKind() == workloadlib::ModuleCodeKind::Glsl ||
                   module.codeKind() == workloadlib::ModuleCodeKind::Hlsl) {
            if (!vgf.src) {
                throw std::invalid_argument("Embedded VGF source compilation requires the source view");
            }
            bindEmbeddedSource(session, executable, *vgf.src);
        }
        // The direct compute-segment path uses SR settings, including shader
        // defaults and a zero push range, rather than the imported VGF values.
        session.setSpecializationInfo(executable, specializationInfo(shader.specializationConstants));
        session.setPushConstantSize(executable, shader.pushConstantsSize);
    }
}

void bindVgfPushConstants(WorkloadExecution &execution, workloadlib::BindingSet &bindings,
                          const DispatchVgfData &dispatch, const DataManager &resources) {
    if (&bindings.workload() != &execution.workload()) {
        throw std::invalid_argument("VGF push bindings belong to a different workload");
    }
    for (const auto executable : execution.workload().executables()) {
        if (executable.type() != workloadlib::ExecutableKind::Compute) {
            continue;
        }
        workloadlib::ArrayView<const std::byte> bytes;
        if (const auto data = getGraphPushData(dispatch.pushConstants, executable.module().name())) {
            const auto payload = resources.getRawData(*data).numpyData();
            if (execution.session().pushConstantSize(executable) != 0) {
                bytes = {reinterpret_cast<const std::byte *>(payload.ptr), payload.size()};
            }
        }
        bindings.bindPushConstants(executable, bytes);
    }
}

} // namespace mlsdk::scenariorunner
