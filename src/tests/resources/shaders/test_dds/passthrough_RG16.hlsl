/*
 * SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
RWTexture2D<float2> in01 : register(u0);
RWTexture2D<float2> out1 : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int2 coord = int2(tid.xy);
    float2 test = in01[coord];
    out1[coord] = test;
}
