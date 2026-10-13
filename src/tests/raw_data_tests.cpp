/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "raw_data.hpp"

#include "vgf-utils/temp_folder.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace mlsdk::scenariorunner;

TEST(RawData, MapsOpaqueFileWithoutNumPyParsing) {
    TempFolder tempFolder("raw_data_binary_test");
    const auto path = tempFolder.relative("model.dgb");
    const std::string bytes = "not a NumPy file";
    std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    const RawData data("binary", path.string());
    const auto &mapping = data.mapping();
    EXPECT_EQ(mapping.size(), bytes.size());
    EXPECT_EQ(std::memcmp(mapping.ptr(), bytes.data(), bytes.size()), 0);
    EXPECT_THROW(data.numpyData(), std::runtime_error);
}
