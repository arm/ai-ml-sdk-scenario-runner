/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph_dump.hpp"
#include "logging.hpp"

#include "vgf-utils/temp_folder.hpp"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace mlsdk::scenariorunner;
namespace logging = mlsdk::logging;

class GraphDump : public testing::TestWithParam<bool> {
  protected:
    TempFolder folder{"scenario_graph_dump_tests"};
    std::vector<std::pair<logging::LogLevel, std::string>> messages;
    const std::array<unsigned char, 4> data{0x01, 0x7f, 0xff, 0x00};

    void SetUp() override {
        logging::setDefaultLogLevel(logging::LogLevel::Info);
        logging::setDefaultHandler([this](const std::string &, logging::LogLevel level, const std::string &message) {
            messages.emplace_back(level, message);
        });
    }

    void TearDown() override {
        logging::setDefaultHandler([](const std::string &, logging::LogLevel, const std::string &) {});
    }

    bool dump(const std::filesystem::path &path) {
        return GetParam() ? writeGraphSessionRAMDump(path, data.data(), data.size())
                          : writeGraphBinaryDump(path, data.data(), data.size());
    }

    void expectWarning(const std::filesystem::path &path) {
        ASSERT_EQ(messages.size(), 1u);
        EXPECT_EQ(messages.front().first, logging::LogLevel::Warning);
        EXPECT_NE(messages.front().second.find(path.string()), std::string::npos);
    }
};

TEST_P(GraphDump, PreservesDumpContents) {
    const auto path = folder.relative("dump");
    ASSERT_TRUE(dump(path));
    // Match the writer mode so RAM dumps use native text newline translation.
    std::ifstream file(path, GetParam() ? std::ios::in : std::ios::binary);
    ASSERT_TRUE(file);
    const std::string contents{std::istreambuf_iterator<char>(file), {}};
    EXPECT_EQ(contents, GetParam() ? std::string("\n00000000:   01 7F FF 00 ")
                                   : std::string(reinterpret_cast<const char *>(data.data()), data.size()));
    EXPECT_TRUE(messages.empty());
}

TEST_P(GraphDump, OpenFailureWarnsAndAllowsLaterDumps) {
    const auto blocked = folder.relative("blocked");
    ASSERT_TRUE(std::filesystem::create_directory(blocked));
    bool stored = true;
    EXPECT_NO_THROW(stored = dump(blocked));
    EXPECT_FALSE(stored);
    expectWarning(blocked);

    const auto next = folder.relative("next");
    EXPECT_TRUE(dump(next));
    EXPECT_GT(std::filesystem::file_size(next), 0u);
    EXPECT_EQ(messages.size(), 1u);
}

TEST_P(GraphDump, WriteFailureWarnsWithoutThrowing) {
    const std::filesystem::path fullDevice{"/dev/full"};
    if (!std::filesystem::exists(fullDevice)) {
        GTEST_SKIP() << "Requires a device that rejects writes";
    }
    bool stored = true;
    EXPECT_NO_THROW(stored = dump(fullDevice));
    EXPECT_FALSE(stored);
    expectWarning(fullDevice);
}

INSTANTIATE_TEST_SUITE_P(Formats, GraphDump, testing::Bool(),
                         [](const testing::TestParamInfo<bool> &info) { return info.param ? "SessionRAM" : "Binary"; });
} // namespace
