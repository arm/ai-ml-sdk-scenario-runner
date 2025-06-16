/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "memory_map.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace mlsdk::numpy {

namespace {
inline bool is_little_endian() {
    uint16_t num = 1;
    return reinterpret_cast<uint8_t *>(&num)[1] == 0;
}

inline char get_endian_char(uint64_t size) { return size < 2 ? '|' : is_little_endian() ? '<' : '>'; }

} // namespace

struct dtype {
    char byteorder;
    char kind;
    uint64_t itemsize = 0;

    dtype() : byteorder('\0'), kind('\0'), itemsize(0) {}

    dtype(char kind, uint64_t itemsize, char byteorder) : byteorder(byteorder), kind(kind), itemsize(itemsize){};

    dtype(char kind, uint64_t itemsize) : byteorder(get_endian_char(itemsize)), kind(kind), itemsize(itemsize){};

    std::string str() const {
        std::stringstream ss;
        ss << byteorder << kind << itemsize;
        return ss.str();
    }
};

namespace {
inline uint64_t size_of(const std::vector<uint64_t> &shape, const struct dtype &dtype) {
    return std::accumulate(shape.begin(), shape.end(), dtype.itemsize, std::multiplies<uint64_t>());
}
} // namespace

struct data_ptr {
    const char *ptr = nullptr;
    std::vector<uint64_t> shape = {};
    struct dtype dtype = {};

    data_ptr() = default;

    data_ptr(const char *ptr, const std::vector<uint64_t> &shape, const struct dtype &dtype)
        : ptr(ptr), shape(shape), dtype(dtype){};

    uint64_t size() const { return size_of(shape, dtype); }
};

void parse(const MemoryMap &mapped, data_ptr &dataPtr);

void write(const std::string &filename, const data_ptr &data_ptr);

void write(const std::string &filename, const std::vector<uint64_t> &shape, const struct dtype &dtype,
           std::function<uint64_t(std::ostream &)> &&callback);

} // namespace mlsdk::numpy
