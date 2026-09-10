/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scenario_runner/resource_data.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <vector>

namespace mlsdk::scenariorunner::samples {

inline std::filesystem::path assetPath(std::string_view executable, std::string_view asset) {
    return std::filesystem::absolute(executable).parent_path() / asset;
}

inline BufferDataView view(const std::vector<uint32_t> &values) {
    return {values.data(), values.size() * sizeof(uint32_t)};
}

inline bool matches(const BufferData &actual, const std::vector<uint32_t> &expected) {
    const auto expectedSize = expected.size() * sizeof(uint32_t);
    return actual.data.size() == expectedSize && std::memcmp(actual.data.data(), expected.data(), expectedSize) == 0;
}

} // namespace mlsdk::scenariorunner::samples
