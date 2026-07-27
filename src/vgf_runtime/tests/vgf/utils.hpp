/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../utils.hpp"

#include "vgf/encoder.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace vgf_runtime::test {

template <typename Populate> std::string writeVgf(Populate populate) {
    auto encoder = mlsdk::vgflib::CreateEncoder(VK_HEADER_VERSION);
    populate(*encoder);
    encoder->Finish();

    std::stringstream stream;
    EXPECT_TRUE(encoder->WriteTo(stream));
    return stream.str();
}

inline std::string makeMaxpoolVgf() {
    const auto &code = assembleMaxpool16x16To8x8Spirv("maxpool_set0", {0, 0, 1, 1});
    return writeVgf([&](mlsdk::vgflib::Encoder &encoder) {
        const auto module = encoder.AddModule(mlsdk::vgflib::ModuleType::GRAPH, "maxpool", "main", code);
        const auto input =
            encoder.AddInputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 16, 16, 16}, {});
        const auto output =
            encoder.AddOutputResource(VK_DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, {1, 8, 8, 16}, {});
        const auto inputBinding = encoder.AddBindingSlot(0, input);
        const auto outputBinding = encoder.AddBindingSlot(1, output);
        const auto inputSet = encoder.AddDescriptorSetInfo({inputBinding}, 0);
        const auto outputSet = encoder.AddDescriptorSetInfo({outputBinding}, 1);
        encoder.AddSegmentInfo(module, "maxpool_graph_segment", {inputSet, outputSet}, {inputBinding}, {outputBinding},
                               {});
    });
}

} // namespace vgf_runtime::test
