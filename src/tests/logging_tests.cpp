/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "logging.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using namespace mlsdk::logging;

TEST(Logging, CreateMessages) {
    std::vector<std::string> logMessages;
    auto logHandler = [&logMessages](const std::string &logger, LogLevel logLevel, const std::string &message) {
        std::stringstream stream;
        stream << logger << " " << logLevel << " " << message;
        logMessages.emplace_back(stream.str());
    };

    setDefaultHandler(logHandler);
    setDefaultLoggerName("logger");
    setDefaultLogLevel(LogLevel::Debug);

    debug("debug message");
    info("info message");
    warning("warning message");
    error("error message");

    log(LogLevel::Debug, "another debug message");
    log("another logger", LogLevel::Warning, "another warning message");

    std::vector<std::string> expected{
        "logger DEBUG debug message",         "logger INFO info message",
        "logger WARNING warning message",     "logger ERROR error message",
        "logger DEBUG another debug message", "another logger WARNING another warning message"};

    ASSERT_TRUE(std::equal(logMessages.begin(), logMessages.end(), expected.begin(), expected.end()));
}

TEST(Logging, MessageFiltering) {
    std::vector<std::string> logMessages;
    auto logHandler = [&logMessages](const std::string &logger, LogLevel logLevel, const std::string &message) {
        std::stringstream stream;
        stream << logger << " " << logLevel << " " << message;
        logMessages.emplace_back(stream.str());
    };

    setDefaultHandler(logHandler);
    setDefaultLoggerName("logger");
    setDefaultLogLevel(LogLevel::Warning);

    debug("debug message");
    info("info message");
    warning("warning message");
    error("error message");

    std::vector<std::string> expected{
        "logger WARNING warning message",
        "logger ERROR error message",
    };

    ASSERT_TRUE(std::equal(logMessages.begin(), logMessages.end(), expected.begin(), expected.end()));
}
