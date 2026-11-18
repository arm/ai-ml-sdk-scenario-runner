/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "scenario_runner/command_types.hpp"

#include <optional>
#include <string_view>

namespace mlsdk::workloadlib {
class BindingSet;
}

namespace mlsdk::scenariorunner {

class DataManager;
class ResourceManager;
class WorkloadExecution;

// Shared with the direct path until VGF dispatch is migrated.
void applyGraphResourceShaderMetadata(ShaderInfo &shader, const VgfInfo &vgf, std::string_view moduleName);

std::optional<RawDataId> getGraphPushData(const std::vector<ResolvedPushConstantMap> &pushConstants,
                                          std::string_view moduleName);

/// Resolve SR's compute settings from module metadata, without importing a
/// workload. The returned source is set only when a substitution was selected.
ShaderInfo resolveVgfShaderConfiguration(const VgfInfo &vgf, const DispatchVgfData &dispatch,
                                         const ResourceManager &resources, std::string_view moduleName,
                                         std::string_view entryPoint, bool hasEmbeddedCode);

/// Apply SR's compute-segment settings before configuring the session.
/// Graph executables retain their imported configuration. Replacements currently
/// require the imported entry point; all intermediate resources remain WL-owned.
void configureVgfExecutables(WorkloadExecution &execution, const VgfInfo &vgf, const DispatchVgfData &dispatch,
                             const ResourceManager &resources);

/// Bind per-compute-executable updates after configure(), including explicit
/// empty updates. A zero-sized effective layout ignores push data as in SR.
void bindVgfPushConstants(WorkloadExecution &execution, workloadlib::BindingSet &bindings,
                          const DispatchVgfData &dispatch, const DataManager &resources);

} // namespace mlsdk::scenariorunner
