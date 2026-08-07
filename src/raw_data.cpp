/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "raw_data.hpp"

#include <utility>

namespace mlsdk::scenariorunner {

RawData::RawData(std::string debugName, const std::string &src)
    : _debugName(std::move(debugName)), _mapped{std::make_unique<MemoryMap>(src)},
      _dataptr(vgfutils::numpy::parse(*_mapped)) {}

const char *RawData::data() const { return _dataptr.ptr; }

size_t RawData::size() const { return _dataptr.size(); }

const std::string &RawData::debugName() const { return _debugName; }
} // namespace mlsdk::scenariorunner
