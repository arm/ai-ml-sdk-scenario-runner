/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "raw_data.hpp"

namespace mlsdk::scenariorunner {

RawData::RawData(const std::string &debugName, const std::string &src)
    : _debugName(debugName), m_mapped{std::make_unique<MemoryMap>(src)} {
    mlsdk::numpy::parse(*m_mapped, m_dataptr);
}

const char *RawData::data() const { return m_dataptr.ptr; }

size_t RawData::size() const { return m_dataptr.size(); }

const std::string &RawData::debugName() const { return _debugName; }
} // namespace mlsdk::scenariorunner
