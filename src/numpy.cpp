/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "numpy.hpp"
#include "memory_map.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

namespace mlsdk::numpy {

namespace {

constexpr std::array<char, 6> numpy_magic_bytes = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

std::string shape_to_str(const std::vector<uint64_t> &shape) {
    std::stringstream shape_ss;
    shape_ss << "(";

    if (shape.empty()) {
        // nothing to do here
    } else if (shape.size() == 1) {
        shape_ss << std::to_string(shape[0]) << ",";
    } else {
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_ss << std::to_string(shape[i]);
            if (i != shape.size() - 1)
                shape_ss << ", ";
        }
    }
    shape_ss << ")";
    return shape_ss.str();
}

std::vector<uint64_t> str_to_shape(const std::string &shape_str) {
    std::stringstream ss(shape_str);
    std::string token;
    std::vector<uint64_t> shape;

    while (std::getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(' '));
        token.erase(token.find_last_not_of(' ') + 1);

        try {
            shape.push_back(std::stoull(token));
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("ml-sdk-numpy: invalid shape: ") + e.what());
        }
    }
    return shape;
}

struct dtype get_dtype(const std::string &dict) {

    size_t descr_start = dict.find("'descr':");
    if (descr_start == std::string::npos)
        throw std::runtime_error("ml-sdk-numpy: missing 'descr' field in header");

    descr_start += 8;
    size_t value_start = dict.find('\'', descr_start);
    size_t value_end = dict.find('\'', value_start + 1);
    if (value_start == std::string::npos || value_end == std::string::npos)
        throw std::runtime_error("ml-sdk-numpy: invalid 'descr' format in header");

    std::string descr_value = dict.substr(value_start + 1, value_end - value_start - 1);
    if (descr_value.size() < 3)
        throw std::runtime_error("ml-sdk-numpy: invalid 'descr' string");

    char byteorder = descr_value[0];
    char kind = descr_value[1];
    uint64_t itemsize;

    try {
        itemsize = std::stoull(descr_value.substr(2));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("ml-sdk-numpy: invalid size in dtype: ") + e.what());
    }

    return dtype(kind, itemsize, byteorder);
}

bool check_fortran_order(const std::string &dict) {
    size_t key_pos = dict.find("'fortran_order':");
    if (key_pos == std::string::npos)
        return false;

    size_t value_pos = dict.find("False", key_pos);
    if (value_pos == std::string::npos || value_pos != key_pos + 17)
        return false;

    return true;
}

void write_header(std::ostream &out, const std::vector<uint64_t> &shape, const std::string &dtype) {
    std::stringstream header_dict;
    header_dict << "{";
    header_dict << "'descr': '" << dtype << "',";
    header_dict << "'fortran_order': False,";
    header_dict << "'shape': " << shape_to_str(shape);
    header_dict << "}";

    // calculate padding len
    std::string header_str = header_dict.str();
    size_t padding_len = 16 - ((10 + header_str.size() + 1) % 16);
    header_str += std::string(padding_len, ' ') + '\n';

    // write magic string
    out.write(numpy_magic_bytes.data(), numpy_magic_bytes.size());

    // write version and HEADER_LEN
    size_t header_len = header_str.size();
    bool use_version_2 = header_len > 65535;

    if (use_version_2) {
        out.put(static_cast<char>(0x02));
        out.put(static_cast<char>(0x00));
        out.put(static_cast<char>(header_len & 0xFF));
        out.put(static_cast<char>((header_len >> 8) & 0xFF));
        out.put(static_cast<char>((header_len >> 16) & 0xFF));
        out.put(static_cast<char>((header_len >> 24) & 0xFF));
    } else {
        out.put(static_cast<char>(0x01));
        out.put(static_cast<char>(0x00));
        out.put(static_cast<char>(header_len & 0xFF));
        out.put(static_cast<char>((header_len >> 8) & 0xFF));
    }

    // write header dict
    out.write(header_str.c_str(), std::streamsize(header_str.size()));
}

} // namespace

void parse(const MemoryMap &mapped, data_ptr &dataPtr) {
    uint8_t major_version;
    uint32_t header_len;
    size_t header_offset = 0;

    // check magic string
    if (std::memcmp(mapped.ptr(), numpy_magic_bytes.data(), numpy_magic_bytes.size()) != 0)
        throw std::runtime_error("ml-sdk-numpy: invalid NumPy file format");

    header_offset += numpy_magic_bytes.size();
    major_version = *reinterpret_cast<const uint8_t *>(mapped.ptr(header_offset));
    header_offset += 2;

    // check version
    if (major_version == 1) {
        uint16_t header_len_v1 = *reinterpret_cast<const uint16_t *>(mapped.ptr(header_offset));
        header_len = header_len_v1;
        header_offset += sizeof(header_len_v1);

    } else if (major_version == 2) {
        header_len = *reinterpret_cast<const uint32_t *>(mapped.ptr(header_offset));
        header_offset += sizeof(header_len);

    } else {
        throw std::runtime_error("ml-sdk-numpy: unsupported NumPy file version");
    }

    // parse header dict
    std::string dict(reinterpret_cast<const char *>(mapped.ptr(header_offset)), header_len);

    // get dtype
    dataPtr.dtype = get_dtype(dict);

    // check byte order
    char byteorder = dataPtr.dtype.byteorder;
    if ((is_little_endian() && byteorder == '>') || (!is_little_endian() && byteorder == '<'))
        throw std::runtime_error("ml-sdk-numpy: mismatch in byte order");

    // check fortran order
    if (!check_fortran_order(dict))
        throw std::runtime_error("ml-sdk-numpy: only supported is fortran_order: False");

    // convert shape str to shape
    size_t shape_start = dict.find('(');
    size_t shape_end = dict.find(')', shape_start);
    std::string shape_str = dict.substr(shape_start + 1, shape_end - shape_start - 1);
    dataPtr.shape = str_to_shape(shape_str);

    // set data_ptr to start of numpy data
    header_offset += header_len;
    dataPtr.ptr = reinterpret_cast<const char *>(mapped.ptr(header_offset));

    // consistency check: verify that all data is mapped
    if (header_offset + dataPtr.size() > mapped.size())
        throw std::runtime_error("ml-sdk-numpy: data size exceeds the mapped memory size");
}

void write(const std::string &filename, const data_ptr &data_ptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("ml-sdk-numpy: cannot open " + filename);
    }

    // write npy header to file
    write_header(file, data_ptr.shape, data_ptr.dtype.str());

    // write data to file
    file.write(reinterpret_cast<const char *>(data_ptr.ptr), std::streamsize(data_ptr.size()));
    file.close();
}

void write(const std::string &filename, const std::vector<uint64_t> &shape, const struct dtype &dtype,
           std::function<uint64_t(std::ostream &)> &&callback) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("ml-sdk-numpy: cannot open " + filename);
    }

    // write npy header to file
    write_header(file, shape, dtype.str());

    // write data to file
    uint64_t size = callback(file);
    if (size_of(shape, dtype) != size) {
        throw std::runtime_error("ml-sdk-numpy: written wrong amount of data");
    }
    file.close();
}

} // namespace mlsdk::numpy
