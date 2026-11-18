/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vgf_binding_map.hpp"

#include "vgf-utils/model_interface.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>

namespace {
using namespace mlsdk::scenariorunner;

const TypedBinding input{0, 17, BufferId{0}, std::nullopt, vk::DescriptorType::eStorageBuffer};
const TypedBinding output{0, 29, BufferId{1}, std::nullopt, vk::DescriptorType::eStorageBuffer};

TEST(VgfBindingMap, UsesFirstInterfaceMatchWithInputsBeforeOutputs) {
    VgfBindingMap bindings({{{{0, 17, 8}, 1}, {{1, 17, 9}, 2}}, {{{0, 17, 10}, 3}, {{1, 29, 7}, 0}}});
    bindings.add(output);
    bindings.add(input);
    EXPECT_EQ(bindings.at(0).resource, output.resource);
    EXPECT_EQ(bindings.at(1).resource, input.resource);
    EXPECT_TRUE(bindings.hasResourceBinding(0));
    EXPECT_TRUE(bindings.hasResourceBinding(1));
    EXPECT_FALSE(bindings.hasResourceBinding(2));
    EXPECT_FALSE(bindings.hasResourceBinding(3));
    EXPECT_FALSE(bindings.hasResourceBinding(4));
    EXPECT_THROW(bindings.at(2), std::runtime_error);
    EXPECT_THROW(bindings.at(3), std::runtime_error);
}

TEST(VgfBindingMap, AcceptsRepeatedAndSharedTargetsKeepingFirstBinding) {
    VgfBindingMap bindings({{{{0, 17, 7}, 0}, {{1, 18, 7}, 0}}, {{{0, 29, 8}, 1}}});
    bindings.add(input);
    auto repeated = input;
    repeated.set = 7; // Set numbers do not identify VGF model-interface entries.
    bindings.add(repeated);
    auto shared = repeated;
    shared.id = 18;
    bindings.add(shared);
    EXPECT_EQ(bindings.at(0).id, 17u);
    EXPECT_EQ(bindings.at(0).set, 0u);
    EXPECT_EQ(bindings.at(0).resource, input.resource);
}

TEST(VgfBindingMap, RejectsDifferentResourcesForOnePublicTarget) {
    VgfBindingMap bindings({{{{0, 17, 7}, 0}, {{1, 18, 7}, 0}}, {}});
    bindings.add(input);
    auto conflicting = input;
    conflicting.resource = BufferId{1};
    EXPECT_THROW(bindings.add(conflicting), std::runtime_error);
    conflicting.id = 18;
    EXPECT_THROW(bindings.add(conflicting), std::runtime_error);
    conflicting.resource = ImageId{0}; // Equal numeric IDs of different kinds also conflict.
    EXPECT_THROW(bindings.add(conflicting), std::runtime_error);
    EXPECT_EQ(bindings.at(0).resource, input.resource);
}

TEST(VgfBindingMap, IgnoresUnknownInterfaceIdsAndRequiresMissingBindings) {
    VgfBindingMap bindings({{{{0, 17, 7}, 0}}, {{{0, 29, 8}, 1}}});
    EXPECT_TRUE(bindings.hasInterfaceBinding(17));
    EXPECT_FALSE(bindings.hasInterfaceBinding(100));
    auto unknown = input;
    unknown.id = 100;
    bindings.add(unknown);
    EXPECT_THROW(bindings.at(0), std::runtime_error);
    bindings.add(input);
    EXPECT_THROW(bindings.at(1), std::runtime_error);
}
} // namespace
