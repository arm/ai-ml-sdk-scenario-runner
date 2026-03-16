/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
Texture2D<float4> in01 : register(t0);
SamplerState samp      : register(s0);

RWTexture2D<float4> out1 : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int2 coord0 = int2(tid.xy);
    int2 coord1 = int2(-1, 0);

    float2 uv = float2(coord1); // becomes (-1.0, 0.0)
    float4 test = in01.SampleLevel(samp, uv, 0);

    out1[coord0] = test;
}
