/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hlsl_compiler.hpp"

#include <gtest/gtest.h>

namespace mlsdk::scenariorunner {

TEST(HlslCompiler, CompilesRepeatedlyAfterErrors) {
    const std::string source = R"(
        RWStructuredBuffer<uint> output : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { output[id.x] = id.x + 1; }
    )";
    auto &compiler = HlslCompiler::get();
    for (int iteration = 0; iteration < 3; ++iteration) {
        const auto [errorLog, invalidSpirv] = compiler.compile("invalid HLSL", "main", "invalid.hlsl");
        EXPECT_FALSE(errorLog.empty());
        EXPECT_TRUE(invalidSpirv.empty());

        const auto [log, spirv] = compiler.compile(source, "main", "valid.hlsl");
        ASSERT_FALSE(spirv.empty()) << log;
        EXPECT_EQ(spirv.front(), 0x07230203u);
    }
}

} // namespace mlsdk::scenariorunner
