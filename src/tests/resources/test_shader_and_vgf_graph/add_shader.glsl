/*
* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
* SPDX-License-Identifier: Apache-2.0
*/
#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly uniform tensorARM<int8_t, 4> in1Add;
layout(set = 0, binding = 2) readonly uniform tensorARM<int8_t, 4> in2Add;
layout(set = 0, binding = 3) writeonly uniform tensorARM<int8_t, 4> outAdd;

void main()
{
    uint coords[4] = uint[](0, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
    int8_t val1;
    tensorReadARM(in1Add, coords, val1);
    int8_t val2;
    tensorReadARM(in2Add, coords, val2);
    int8_t result  = val1 + val2;
    tensorWriteARM(outAdd, coords, result);
}
