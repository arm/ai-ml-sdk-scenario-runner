/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <ostream>
#include <string>

namespace mlsdk::logging {

/// \brief Logging levels
enum class LogLevel { Debug, Info, Warning, Error };

/// \brief Utility function to support printing logging level in the output stream
std::ostream &operator<<(std::ostream &os, const LogLevel &logLevel);

/// \brief Callable to handle the log messages
using LogHandler = std::function<void(const std::string &logger, LogLevel logLevel, const std::string &message)>;

/// \brief Log the message with the provided logging level using default logger
///
/// \param logLevel Logging level
/// \param message Logging message
void log(LogLevel logLevel, const std::string &message);

/// \brief Log the message with the provided logging level using logger
///
/// \param logger Logger's name
/// \param logLevel Logging level
/// \param message Logging message
void log(const std::string &logger, LogLevel logLevel, const std::string &message);

/// \brief Log the message with logging level DEBUG using default logger
///
/// \param message Logging message
void debug(const std::string &message);

/// \brief Log the message with logging level INFO using default logger
///
/// \param message Logging message
void info(const std::string &message);

/// \brief Log the message with logging level WARNING using default logger
///
/// \param message Logging message
void warning(const std::string &message);

/// \brief Log the message with logging level ERROR using default logger
///
/// \param message Logging message
void error(const std::string &message);

/// \brief Set the logging handler for the default logger
///
/// \param logHandler Logging handler
void setDefaultHandler(const LogHandler &logHandler);

/// \brief Set the logging level for the default logger
///
/// \param logLevel Logging level
void setDefaultLogLevel(LogLevel logLevel);

/// \brief Set the name for the default logger
///
/// \param name Logger name
void setDefaultLoggerName(const std::string &name);

} // namespace mlsdk::logging
