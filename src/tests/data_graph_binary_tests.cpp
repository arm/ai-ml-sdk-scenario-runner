/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace {

using mlsdk::scenariorunner::validateDataGraphBinaryHeader;

constexpr std::array<uint8_t, 17> validBinary = {0xff, 0xdd, 0x22, 0x11, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0xaa};

TEST(DataGraphBinaryHeader, AcceptsMatchingPublicHeader) {
    EXPECT_NO_THROW(validateDataGraphBinaryHeader(validBinary.data(), validBinary.size()));

    auto binary = validBinary;
    binary[4] = 1;
    EXPECT_NO_THROW(validateDataGraphBinaryHeader(binary.data(), binary.size()));
}

TEST(DataGraphBinaryHeader, RejectsMalformedPublicHeader) {
    EXPECT_THROW(validateDataGraphBinaryHeader(validBinary.data(), 15), std::runtime_error);
    EXPECT_THROW(validateDataGraphBinaryHeader(nullptr, validBinary.size()), std::runtime_error);

    auto binary = validBinary;
    binary[0] = 0;
    EXPECT_THROW(validateDataGraphBinaryHeader(binary.data(), binary.size()), std::runtime_error);

    binary = validBinary;
    binary[8] = 2;
    EXPECT_THROW(validateDataGraphBinaryHeader(binary.data(), binary.size()), std::runtime_error);

    binary = validBinary;
    binary[12] = 1;
    EXPECT_THROW(validateDataGraphBinaryHeader(binary.data(), binary.size()), std::runtime_error);
}

} // namespace
