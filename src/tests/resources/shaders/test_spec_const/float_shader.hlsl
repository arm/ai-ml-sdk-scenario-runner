/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
  RWStructuredBuffer<float> out_data : register(u0);

  [[vk::constant_id(0)]]
  const float SPEC_CONST = 100500.0f;

  [numthreads(1, 1, 1)]
  void main(uint3 DTid : SV_DispatchThreadID)
  {
      out_data[0] = SPEC_CONST;
  }
