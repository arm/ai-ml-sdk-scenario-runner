// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

void fuzzScenario(const uint8_t *data, size_t size);
void validateScenarioFuzzerEnvironment();
