/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "json_writer.hpp"

#include "nlohmann/json.hpp"
#include "vgf-utils/temp_folder.hpp"

#include <fstream>
#include <vector>

namespace mlsdk::scenariorunner {

TEST(JsonWriter, WritesMemoryUsageWithoutTimestamps) {
    TempFolder tempFolder("scenario_runner_json_writer_tests");
    const auto profilingPath = tempFolder.relative("dry_run_profiling.json");
    const MemoryProfilingData memoryProfilingData{{{"graph_ref/conv2d_graph_segment", 4096}}};

    writeProfilingData(std::nullopt, memoryProfilingData, profilingPath, 0, 1);

    std::ifstream dumpFile(profilingPath);
    const auto profilingData = nlohmann::json::parse(dumpFile);

    ASSERT_FALSE(profilingData.contains("Timestamps"));
    ASSERT_TRUE(profilingData.contains("Memory Usage"));
    ASSERT_EQ(profilingData["Memory Usage"].size(), 1);
    EXPECT_EQ(profilingData["Memory Usage"][0]["Command type"], "DataGraphDispatch");
    EXPECT_EQ(profilingData["Memory Usage"][0]["Command name"], "graph_ref/conv2d_graph_segment");
    EXPECT_EQ(profilingData["Memory Usage"][0]["Session memory [bytes]"], 4096);
}

TEST(JsonWriter, WritesEmptyObjectWithoutProfilingData) { // cppcheck-suppress syntaxError
    TempFolder tempFolder("scenario_runner_json_writer_tests");
    const auto profilingPath = tempFolder.relative("empty_dry_run_profiling.json");

    writeProfilingData(std::nullopt, {}, profilingPath, 0, 1);

    std::ifstream dumpFile(profilingPath);
    const auto profilingData = nlohmann::json::parse(dumpFile);

    EXPECT_TRUE(profilingData.is_object());
    EXPECT_TRUE(profilingData.empty());
}

} // namespace mlsdk::scenariorunner
