/*
 * SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

StructuredBuffer<TestType> In1Buffer : register(t0);
StructuredBuffer<TestType> In2Buffer : register(t1);
RWStructuredBuffer<TestType> OutBuffer : register(u2, space1);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    OutBuffer[idx] = In1Buffer[idx] + In2Buffer[idx];
}
