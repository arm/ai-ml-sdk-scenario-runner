/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "vgf-utils/memory_map.hpp"
#include "vgf-utils/numpy.hpp"

#include <string>

namespace mlsdk::scenariorunner {

class RawData {
  public:
    explicit RawData(std::string debugName, const std::string &src);

    vgfutils::numpy::DataPtr numpyData() const;
    const MemoryMap &mapping() const;
    const std::string &debugName() const;

  private:
    std::string _debugName;
    MemoryMap _mapped;
};
} // namespace mlsdk::scenariorunner
