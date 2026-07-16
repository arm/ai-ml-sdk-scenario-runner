/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"
#include "data_manager.hpp"
#include "resource_data.hpp"
#include "scenario.hpp"
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using namespace mlsdk::scenariorunner;

Buffer &prepareBuffer(Context &ctx, DataManager &dm, const Guid &guid, uint32_t sizeBytes) {
    BufferInfo info;
    info.size = sizeBytes;
    dm.createBuffer(guid, info);
    auto &buf = dm.getBufferMut(guid);
    buf.setup(ctx);
    buf.allocateMemory(ctx);
    return buf;
}

TEST(BufferInMemoryTransfer, UploadThrowsOnSizeMismatch) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;

    const Guid guid("buf_mismatch");
    auto &buf = prepareBuffer(ctx, dm, guid, /*sizeBytes*/ 16);

    std::vector<char> small(8, static_cast<char>(0x7B));
    BufferDataView view{small.data(), small.size()};
    EXPECT_THROW(buf.upload(ctx, view), std::runtime_error);
}

TEST(BufferInMemoryTransfer, UploadSucceedsAndPersistsCopy) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;

    const Guid guid("buf_upload_ok");
    auto &buf = prepareBuffer(ctx, dm, guid, /*sizeBytes*/ 16);

    std::vector<char> payload(16);
    std::iota(payload.begin(), payload.end(), static_cast<char>(0));
    BufferDataView view{payload.data(), payload.size()};
    ASSERT_NO_THROW(buf.upload(ctx, view));

    // Mutate source to ensure Buffer keeps a copy, not an alias
    std::fill(payload.begin(), payload.end(), static_cast<char>(0xAA));

    const auto bufferData = buf.download(ctx);
    ASSERT_EQ(bufferData.data.size(), static_cast<size_t>(buf.size()));

    for (size_t i = 0; i < bufferData.data.size(); ++i) {
        EXPECT_EQ(static_cast<unsigned char>(bufferData.data[i]), static_cast<unsigned char>(i))
            << "mismatch at index " << i;
    }
}

TEST(BufferInMemoryTransfer, DownloadReturnsUploadedData) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;

    const Guid guid("buf_download_ok");
    auto &buf = prepareBuffer(ctx, dm, guid, /*sizeBytes*/ 12);

    std::vector<char> payload(12);
    std::iota(payload.begin(), payload.end(), static_cast<char>(0));
    BufferDataView view{payload.data(), payload.size()};
    buf.upload(ctx, view);

    const auto bufferData = buf.download(ctx);
    ASSERT_EQ(bufferData.data.size(), payload.size());
    EXPECT_EQ(bufferData.data, payload);
}
