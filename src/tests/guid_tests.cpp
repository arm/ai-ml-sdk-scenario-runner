/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <string>

#include <gtest/gtest.h>

#include "guid.hpp"

namespace mlsdk::scenariorunner {

TEST(Guids, Ctor) {
    Guid guid0 = Guid("This is guid 0");
    Guid guid1 = Guid("This is guid 1");
    Guid guid2 = Guid("This is guid 1"); // intentionally the same string. guid2 must equal guid1.

    ASSERT_TRUE(guid0.isValid() == true);
    ASSERT_TRUE(guid1.isValid() == true);
    ASSERT_TRUE(guid2.isValid() == true);

    ASSERT_TRUE(guid0 != guid1);
    ASSERT_TRUE(guid1 == guid2);

    Guid invalid;
    ASSERT_TRUE(invalid.isValid() == false);
}

TEST(Guids, Copy) { // cppcheck-suppress syntaxError
    Guid guid0 = Guid("This is guid 0");
    Guid guid1 = guid0;

    ASSERT_TRUE(guid0.isValid() == true);
    ASSERT_TRUE(guid1.isValid() == true);

    ASSERT_TRUE(guid1 == Guid("This is guid 0"));
    ASSERT_TRUE(guid0 == Guid("This is guid 0"));
    ASSERT_TRUE(guid1 == guid0);
}

} // namespace mlsdk::scenariorunner
