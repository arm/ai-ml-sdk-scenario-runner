/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
Texture2D<float4>  in01  : register(t0);  // read-only view
RWTexture2D<float4> out1 : register(u1);  // write-only storage

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int2 coord = int2(tid.xy);

    // Exact texel fetch at integer coords (no sampler, no filtering)
    float4 value = in01.Load(int3(coord, 0));

    out1[coord] = value;
}
