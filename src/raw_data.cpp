/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "raw_data.hpp"

namespace mlsdk::scenariorunner {

RawData::RawData(const std::string &debugName, const std::string &src)
    : _debugName(debugName), _mapped{std::make_unique<MemoryMap>(src)} {
    mlsdk::numpy::parse(*_mapped, _dataptr);
}

const char *RawData::data() const { return _dataptr._ptr; }

size_t RawData::size() const { return _dataptr.size(); }

const std::string &RawData::debugName() const { return _debugName; }
} // namespace mlsdk::scenariorunner
