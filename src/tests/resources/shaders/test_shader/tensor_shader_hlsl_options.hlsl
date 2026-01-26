/*
 * SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TEST_OPTION
    #error TEST_OPTION not defined
#endif

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // No-op; compile-time option test only.
}
