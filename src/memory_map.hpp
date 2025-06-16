/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

class MemoryMap {
  public:
    explicit MemoryMap(const std::string &filename);
    MemoryMap(const MemoryMap &) = delete;
    MemoryMap &operator=(const MemoryMap &) = delete;
    MemoryMap(const MemoryMap &&) = delete;
    MemoryMap &operator=(const MemoryMap &&) = delete;

    ~MemoryMap();

    const void *ptr(const size_t offset = 0) const;
    size_t size() const { return m_size; }

  private:
#ifdef _WIN32
    void *m_hFile;
    void *m_hMap;
#else
    int m_fd;
#endif
    void *m_addr;
    size_t m_size;
};
