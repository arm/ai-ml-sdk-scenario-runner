/*
 * SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
StructuredBuffer<float> In1BufferAA : register(t0);
RWStructuredBuffer<float> OutBufferAddPush : register(u1);

struct PushConstants {
    float data[10];
};

[[vk::push_constant]]
PushConstants pc;

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    OutBufferAddPush[idx] = In1BufferAA[idx] + pc.data[idx];
}
