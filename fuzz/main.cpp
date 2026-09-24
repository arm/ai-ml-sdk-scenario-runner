// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include "fuzzers.hpp"

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    validateScenarioFuzzerEnvironment();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (data == nullptr || size == 0) {
        return 0;
    }

#ifdef SCENARIO_FUZZ_REJECTIONS
    fuzzScenarioRejections(data, size);
#else
    fuzzScenario(data, size);
#endif
    return 0;
}
