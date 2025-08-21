/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "numpy.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {

class RawData {
  public:
    RawData() = default;
    explicit RawData(const std::string &debugName, const std::string &src);

    const char *data() const;
    size_t size() const;
    const std::string &debugName() const;

  private:
    const std::string _debugName{};
    std::unique_ptr<MemoryMap> _mapped{};
    mlsdk::numpy::data_ptr _dataptr{};
};
} // namespace mlsdk::scenariorunner
