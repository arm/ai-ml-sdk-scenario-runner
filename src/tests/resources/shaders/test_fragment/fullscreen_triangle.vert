/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#version 450

void main() {
    vec2 pos;
    if (gl_VertexIndex == 0) {
        pos = vec2(-1.0, -1.0);
    } else if (gl_VertexIndex == 1) {
        pos = vec2(3.0, -1.0);
    } else {
        pos = vec2(-1.0, 3.0);
    }

    gl_Position = vec4(pos, 0.0, 1.0);
}
