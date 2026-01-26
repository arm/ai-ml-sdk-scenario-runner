/*
* SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
* SPDX-License-Identifier: Apache-2.0
*/

Texture2D<float4> depth : register(t0);
RWTexture2D<float4> out1 : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 coord = DTid.xy;
    float4 temp = depth.Load(int3(coord, 0));
    out1[coord] = temp;
}
