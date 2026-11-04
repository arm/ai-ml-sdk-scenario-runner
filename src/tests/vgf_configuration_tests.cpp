/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resource_manager.hpp"
#include "vgf_configuration.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;

ShaderInfo replacementShader() {
    ShaderInfo shader;
    shader.debugName = "replacement";
    shader.entry = "main";
    shader.pushConstantsSize = 4;
    shader.specializationConstants = {{5, {4}}};
    // Opaque bytes: selecting SR settings does not interpret shader code.
    shader.src = std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{1, 2, 3, 4});
    return shader;
}

TEST(VgfConfiguration, UsesShaderDefaultsWithoutMatchingSettings) {
    ResourceManager resources;
    DispatchVgfData dispatch{VgfId{0}};
    dispatch.shaderSubstitutions = {{ShaderId{99}, "other"}};
    const auto shader = resolveVgfShaderConfiguration(VgfInfo{}, dispatch, resources, "module", "main", true);
    EXPECT_EQ(shader.debugName, "module");
    EXPECT_EQ(shader.entry, "main");
    EXPECT_EQ(shader.pushConstantsSize, 0u);
    EXPECT_TRUE(shader.specializationConstants.empty());
    EXPECT_FALSE(shader.src);
}

TEST(VgfConfiguration, SelectsFirstSubstitutionAndRetainsItsSettings) {
    ResourceManager resources;
    auto replacement = replacementShader();
    replacement.entry = "different_entry";
    const auto first = resources.addShader(replacement);
    const auto second = resources.addShader(ShaderInfo{});
    DispatchVgfData dispatch{VgfId{0}};
    dispatch.shaderSubstitutions = {{first, "module"}, {second, "module"}};
    const auto shader = resolveVgfShaderConfiguration(VgfInfo{}, dispatch, resources, "module", "main", false);
    EXPECT_EQ(shader.src, resources.get(first).src);
    EXPECT_EQ(shader.entry, "different_entry");
    EXPECT_EQ(shader.debugName, "replacement");
    EXPECT_EQ(shader.pushConstantsSize, 4u);
    ASSERT_EQ(shader.specializationConstants.size(), 1u);
    EXPECT_EQ(shader.specializationConstants[0].id, 5);
    EXPECT_EQ(shader.specializationConstants[0].value.i, 4);
}

TEST(VgfConfiguration, AppliesVgfOverridesUsingFirstMatchingTarget) {
    ResourceManager resources;
    DispatchVgfData dispatch{VgfId{0}};
    dispatch.shaderSubstitutions = {{resources.addShader(replacementShader()), "module"}};
    VgfInfo vgf;
    vgf.pushConstantsSize = 8;
    vgf.specializationConstantMaps = {{{{5, {42}}}, "other"}, {{{5, {5}}}, "module"}, {{{5, {99}}}, "module"}};
    const auto shader = resolveVgfShaderConfiguration(vgf, dispatch, resources, "module", "main", true);
    EXPECT_EQ(shader.pushConstantsSize, 8u);
    ASSERT_EQ(shader.specializationConstants.size(), 1u);
    EXPECT_EQ(shader.specializationConstants[0].value.i, 5);
}

TEST(VgfConfiguration, EmptySpecializationOverrideClearsSubstitutionSettings) {
    ResourceManager resources;
    DispatchVgfData dispatch{VgfId{0}};
    dispatch.shaderSubstitutions = {{resources.addShader(replacementShader()), "module"}};
    VgfInfo vgf;
    vgf.specializationConstantMaps = {{{}, "module"}};
    const auto shader = resolveVgfShaderConfiguration(vgf, dispatch, resources, "module", "main", true);
    EXPECT_TRUE(shader.specializationConstants.empty());
    // Zero in VgfInfo means no size override; retain the replacement's range.
    EXPECT_EQ(shader.pushConstantsSize, 4u);
}

TEST(VgfConfiguration, RejectsMissingModuleAndEmptySubstitutionCode) {
    ResourceManager resources;
    DispatchVgfData dispatch{VgfId{0}};
    EXPECT_THROW(resolveVgfShaderConfiguration(VgfInfo{}, dispatch, resources, "module", "main", false),
                 std::runtime_error);
    auto shader = replacementShader();
    shader.src.reset();
    dispatch.shaderSubstitutions = {{resources.addShader(shader), "module"}};
    EXPECT_THROW(resolveVgfShaderConfiguration(VgfInfo{}, dispatch, resources, "module", "main", true),
                 std::runtime_error);
    shader.src = std::make_shared<const std::vector<uint32_t>>();
    dispatch.shaderSubstitutions = {{resources.addShader(shader), "module"}};
    EXPECT_THROW(resolveVgfShaderConfiguration(VgfInfo{}, dispatch, resources, "module", "main", true),
                 std::runtime_error);
}

TEST(VgfConfiguration, SelectsFirstPushPayloadForTargetOrNoUpdate) {
    const std::vector<ResolvedPushConstantMap> updates{
        {RawDataId{0}, "first"}, {RawDataId{1}, "second"}, {RawDataId{2}, "first"}};
    EXPECT_EQ(getGraphPushData(updates, "first"), RawDataId{0});
    EXPECT_EQ(getGraphPushData(updates, "second"), RawDataId{1});
    EXPECT_FALSE(getGraphPushData(updates, "absent"));
    EXPECT_FALSE(getGraphPushData({}, "first"));
}
} // namespace
