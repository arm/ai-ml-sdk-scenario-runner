/*
 * SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CONSTANT_0
#define CONSTANT_0 0.0
#endif

StructuredBuffer<float> inBuffer : register(t0);
RWStructuredBuffer<float> outBuffer : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    float value = inBuffer[idx];
    value += (float)CONSTANT_0;
#ifdef DIVIDE_BY_TWO
    value /= 2.0;
#endif
    outBuffer[idx] = value;
}
