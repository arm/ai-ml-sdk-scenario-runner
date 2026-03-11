/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

[[vk::binding(0, 0)]] ByteAddressBuffer In1Buffer;
[[vk::binding(1, 0)]] RWByteAddressBuffer OutBuffer;

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint offset = tid.x * 4;
    OutBuffer.Store(offset, In1Buffer.Load(offset) + 0x01010101u);
}
