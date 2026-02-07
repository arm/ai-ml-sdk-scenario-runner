/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_formats.hpp"

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

TEST(ImageFormats, FindHandlerByFilename) {
    EXPECT_NE(getImageFormatHandler("image.dds"), nullptr);
    EXPECT_EQ(getImageFormatHandler("image.bmp"), nullptr);
}

TEST(ImageFormats, getFormatThrowsOnUnsupportedFile) {
    EXPECT_THROW(getVkFormatForImage("no_such.ext"), std::runtime_error);
}
