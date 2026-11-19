/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <filesystem>

namespace mlsdk::scenariorunner {

// Shared dump writers log file failures and return false without throwing.
bool writeGraphBinaryDump(const std::filesystem::path &path, const void *data, size_t size);
bool writeGraphSessionRAMDump(const std::filesystem::path &path, const unsigned char *data, size_t size);

} // namespace mlsdk::scenariorunner
