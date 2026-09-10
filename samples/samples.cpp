/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "samples.hpp"

#include <iostream>

int main(int, char **argv) {
    using namespace mlsdk::scenariorunner::samples;

    if (runInMemoryBufferSample() != 0 || runComputeSample(argv[0]) != 0 || runVgfInferenceSample(argv[0]) != 0 ||
        runJsonSample(argv[0]) != 0) {
        return 1;
    }

    std::cout << "Samples execution complete.\n";
    return 0;
}
