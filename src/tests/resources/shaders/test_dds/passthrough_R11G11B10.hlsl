/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
Texture2D<float3> in01 : register(t0);
RWTexture2D<float3> out1 : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int2 coord = int2(tid.xy);
    float3 value = in01.Load(int3(coord, 0));
    out1[coord] = value;
}
