/*
 * SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
StructuredBuffer<float> inBuffer : register(t0);
RWStructuredBuffer<float> outBuffer : register(u1);


struct PushConstants
{
    float4 _offsets;
    float2 _multipliers;
    float _inv;
};

[[vk::push_constant]]
PushConstants pc;

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    float value = inBuffer[idx] + pc._offsets.x + pc._offsets.y + pc._offsets.z + pc._offsets.w;
    value *= pc._inv;
    value *= pc._multipliers.x * pc._multipliers.y;
    outBuffer[idx] = value;
}
