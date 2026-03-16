/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
Texture2D<float4> in01 : register(t0);
RWTexture2D<float> out1 : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int2 coord = int2(tid.xy);
    float4 test = in01.Load(int3(coord, 0));
    out1[coord] = test.x;
}
