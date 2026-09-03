/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"
#include "context.hpp"
#include "data_manager.hpp"
#include "resource_data.hpp"
#include "scenario_options.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

Buffer &prepareBuffer(Context &ctx, DataManager &dm, BufferId id, uint32_t sizeBytes) {
    BufferInfo info;
    info.size = sizeBytes;
    dm.createBuffer(id, info);
    auto &buf = dm.getBufferMut(id);
    buf.setup(ctx);
    buf.allocateMemory(ctx);
    return buf;
}

TEST(BufferInMemoryTransfer, UploadThrowsOnSizeMismatch) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;

    auto &buf = prepareBuffer(ctx, dm, BufferId{0}, /*sizeBytes*/ 16);

    std::vector<std::byte> small(8, std::byte{0x7B});
    BufferDataView view{small.data(), small.size()};
    EXPECT_THROW(buf.upload(ctx, view), std::runtime_error);
}

TEST(BufferInMemoryTransfer, UploadSucceedsAndPersistsCopy) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;

    auto &buf = prepareBuffer(ctx, dm, BufferId{0}, /*sizeBytes*/ 16);

    std::vector<std::byte> payload(16);
    uint8_t value = 0;
    std::generate(payload.begin(), payload.end(), [&value] { return std::byte{value++}; });
    BufferDataView view{payload.data(), payload.size()};
    ASSERT_NO_THROW(buf.upload(ctx, view));

    // Mutate source to ensure Buffer keeps a copy, not an alias
    std::fill(payload.begin(), payload.end(), std::byte{0xAA});

    const auto bufferData = buf.download(ctx);
    ASSERT_EQ(bufferData.data.size(), static_cast<size_t>(buf.size()));

    for (size_t i = 0; i < bufferData.data.size(); ++i) {
        EXPECT_EQ(std::to_integer<unsigned char>(bufferData.data[i]), static_cast<unsigned char>(i))
            << "mismatch at index " << i;
    }
}

TEST(BufferInMemoryTransfer, DownloadReturnsUploadedData) {
    ScenarioOptions opts{};
    Context ctx{opts};
    DataManager dm;

    auto &buf = prepareBuffer(ctx, dm, BufferId{0}, /*sizeBytes*/ 12);

    std::vector<std::byte> payload(12);
    uint8_t value = 0;
    std::generate(payload.begin(), payload.end(), [&value] { return std::byte{value++}; });
    BufferDataView view{payload.data(), payload.size()};
    buf.upload(ctx, view);

    const auto bufferData = buf.download(ctx);
    ASSERT_EQ(bufferData.data.size(), payload.size());
    EXPECT_EQ(bufferData.data, payload);
}
