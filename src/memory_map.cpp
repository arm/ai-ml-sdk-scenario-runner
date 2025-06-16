/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "memory_map.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#    include <windows.h>

#    include <fileapi.h>
#else
#    include <sys/mman.h>
#    include <unistd.h>
#endif

#include <stdexcept>

MemoryMap::MemoryMap(const std::string &filename) {
#ifdef _WIN32
    HANDLE hFile = CreateFile(filename.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Could not open file " + filename);
    }
    m_hFile = reinterpret_cast<void *>(hFile);

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(m_hFile, &fileSize)) {
        throw std::runtime_error("Failed to get file size for " + filename);
    }
    m_size = static_cast<size_t>(fileSize.QuadPart);

    HANDLE hMap = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (hMap == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to create file mapping for file " + filename);
    }
    m_hMap = reinterpret_cast<void *>(hMap);

    m_addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (m_addr == nullptr) {
        throw std::runtime_error("MapViewOfFile failed for file " + filename);
    }
#else
    m_fd = open(filename.c_str(), O_RDONLY);
    if (m_fd < 0) {
        throw std::runtime_error("Could not open file " + filename);
    }
    struct stat st = {};
    if (fstat(m_fd, &st) == -1) {
        throw std::runtime_error("Could not read attributes of file " + filename);
    }
    m_size = size_t(st.st_size);

    m_addr = mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
    if (m_addr == MAP_FAILED) {
        throw std::runtime_error("Failed to memory map the file " + filename);
    }
#endif
}

MemoryMap::~MemoryMap() {
#ifdef _WIN32
    UnmapViewOfFile(m_addr);
    CloseHandle(reinterpret_cast<HANDLE>(m_hMap));
    CloseHandle(reinterpret_cast<HANDLE>(m_hFile));
#else
    if (m_fd > 0) {
        munmap(m_addr, m_size);
        close(m_fd);
    }
#endif
}

const void *MemoryMap::ptr(const size_t offset) const {
    if (offset >= m_size) {
        throw std::runtime_error("offset " + std::to_string(offset) + " exceeds the mapped size " +
                                 std::to_string(m_size));
    }
    return reinterpret_cast<const void *>(static_cast<char *>(m_addr) + offset);
}
