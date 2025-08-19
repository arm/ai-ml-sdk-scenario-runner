/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "numpy.hpp"

#include "memory_map.hpp"
#include "temp_folder.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(NumPy, Roundtrip) {
    mlsdk::testing::TempFolder tempFolder("scenario_runner_numpy_test");

    std::vector<int32_t> data = {42, 65, 76, 98, 106};
    std::vector<uint64_t> shape = {data.size()};
    mlsdk::numpy::dtype type{'i', sizeof(int32_t)};

    std::string filename = tempFolder.relative("test.npy").string();

    mlsdk::numpy::data_ptr write_ptr{reinterpret_cast<char *>(data.data()), shape, type};
    mlsdk::numpy::write(filename, write_ptr);

    MemoryMap map{filename};

    mlsdk::numpy::data_ptr read_ptr;
    mlsdk::numpy::parse(map, read_ptr);

    ASSERT_TRUE(read_ptr._shape.size() == shape.size());
    for (unsigned i = 0; i < shape.size(); i++) {
        ASSERT_TRUE(read_ptr._shape[i] == shape[i]);
    }
    ASSERT_TRUE(read_ptr._dtype._byteorder == type._byteorder);
    ASSERT_TRUE(read_ptr._dtype._kind == type._kind);
    ASSERT_TRUE(read_ptr._dtype._itemsize == type._itemsize);
    const int32_t *test = reinterpret_cast<const int32_t *>(read_ptr._ptr);
    for (unsigned i = 0; i < data.size(); i++) {
        ASSERT_TRUE(test[i] == data[i]);
    }
}

TEST(NumPy, RoundtripCallbackWrite) {
    mlsdk::testing::TempFolder tempFolder("scenario_runner_numpy_callback_test");

    std::vector<int32_t> data = {42, 65, 76, 98, 106};
    std::vector<uint64_t> shape = {data.size()};
    mlsdk::numpy::dtype type{'i', sizeof(int32_t)};

    std::string filename = tempFolder.relative("test.npy").string();

    mlsdk::numpy::write(filename, shape, type, [&](std::ostream &oss) {
        const char *ptr = reinterpret_cast<const char *>(data.data());
        for (unsigned j = 0; j < data.size() * sizeof(uint32_t); j++) {
            oss << ptr[j];
        }
        return data.size() * sizeof(uint32_t);
    });

    MemoryMap map{filename};

    mlsdk::numpy::data_ptr read_ptr;
    mlsdk::numpy::parse(map, read_ptr);

    ASSERT_TRUE(read_ptr._shape.size() == shape.size());
    for (unsigned i = 0; i < shape.size(); i++) {
        ASSERT_TRUE(read_ptr._shape[i] == shape[i]);
    }
    ASSERT_TRUE(read_ptr._dtype._byteorder == type._byteorder);
    ASSERT_TRUE(read_ptr._dtype._kind == type._kind);
    ASSERT_TRUE(read_ptr._dtype._itemsize == type._itemsize);
    const int32_t *test = reinterpret_cast<const int32_t *>(read_ptr._ptr);
    for (unsigned i = 0; i < data.size(); i++) {
        ASSERT_TRUE(test[i] == data[i]);
    }
}
