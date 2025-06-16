/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "logging.hpp"

namespace mlsdk::logging {

std::ostream &operator<<(std::ostream &os, const LogLevel &logLevel) {
    switch (logLevel) {
    case LogLevel::Info:
        os << "INFO";
        break;
    case LogLevel::Warning:
        os << "WARNING";
        break;
    case LogLevel::Debug:
        os << "DEBUG";
        break;
    case LogLevel::Error:
        os << "ERROR";
        break;
    }

    return os;
}

namespace {

void noLogging([[maybe_unused]] const std::string &logger, [[maybe_unused]] LogLevel logLevel,
               [[maybe_unused]] const std::string &message) {}

struct LoggingConfig {
    std::string loggerName;
    LogHandler handler{noLogging};
    LogLevel logLevel{LogLevel::Info};
};

LoggingConfig defaultConfig;

} // namespace

void setDefaultHandler(const LogHandler &logHandler) { defaultConfig.handler = logHandler; }
void setDefaultLogLevel(LogLevel logLevel) { defaultConfig.logLevel = logLevel; }
void setDefaultLoggerName(const std::string &name) { defaultConfig.loggerName = name; }

void log(const std::string &logger, LogLevel logLevel, const std::string &message) {
    if (logLevel < defaultConfig.logLevel) {
        return;
    }

    defaultConfig.handler(logger, logLevel, message);
}

void log(LogLevel logLevel, const std::string &message) { log(defaultConfig.loggerName, logLevel, message); }

void debug(const std::string &message) { log(LogLevel::Debug, message); }
void info(const std::string &message) { log(LogLevel::Info, message); }
void warning(const std::string &message) { log(LogLevel::Warning, message); }
void error(const std::string &message) { log(LogLevel::Error, message); }

} // namespace mlsdk::logging
