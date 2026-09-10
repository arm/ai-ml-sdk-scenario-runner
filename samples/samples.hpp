/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

namespace mlsdk::scenariorunner::samples {

int runInMemoryBufferSample();
int runComputeSample(std::string_view executable);
int runVgfInferenceSample(std::string_view executable);
int runJsonSample(std::string_view executable);

} // namespace mlsdk::scenariorunner::samples
