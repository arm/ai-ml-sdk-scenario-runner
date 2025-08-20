/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "memory_map.hpp"

#include <cstdint>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace mlsdk::numpy {

struct dtype {
    char _byteorder{'\0'};
    char _kind{'\0'};
    uint64_t _itemsize{0};

    dtype() = default;
    dtype(char kind, uint64_t itemsize, char byteorder);
    dtype(char kind, uint64_t itemsize);

    std::string str() const {
        std::stringstream ss;
        ss << _byteorder << _kind << _itemsize;
        return ss.str();
    }
};

struct data_ptr {
    const char *_ptr = nullptr;
    std::vector<uint64_t> _shape = {};
    dtype _dtype = {};

    data_ptr() = default;

    data_ptr(const char *ptr, const std::vector<uint64_t> &shape, const dtype &dtype)
        : _ptr(ptr), _shape(shape), _dtype(dtype){};

    uint64_t size() const;
};

void parse(const MemoryMap &mapped, data_ptr &dataPtr);

void write(const std::string &filename, const data_ptr &data_ptr);

void write(const std::string &filename, const std::vector<uint64_t> &shape, const dtype &dtype,
           std::function<uint64_t(std::ostream &)> &&callback);

} // namespace mlsdk::numpy
