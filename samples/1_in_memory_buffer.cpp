/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "samples.hpp"

#include "scenario_runner/scenario_builder.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>

int mlsdk::scenariorunner::samples::runInMemoryBufferSample() {
    using namespace mlsdk::scenariorunner;

    try {
        // Create a buffer with no input file, then transfer application data
        // directly to and from ML SDK Scenario Runner.
        auto builder = createScenarioBuilder();
        const auto bufferId = builder->addBuffer(BufferInfo{"in-memory-buffer", 16});
        auto scenario = builder->build(ScenarioOptions{});

        const std::array<uint32_t, 4> input{1, 2, 3, 4};
        scenario->upload(bufferId, BufferDataView{input.data(), sizeof(input)});

        const auto output = scenario->download(bufferId);
        if (output.data.size() != sizeof(input) || std::memcmp(output.data.data(), input.data(), sizeof(input)) != 0) {
            std::cerr << "Downloaded buffer does not match the uploaded data\n";
            return 1;
        }

        std::cout << "Transferred buffer " << bufferId.value() << " through the Scenario Runner API\n";
    } catch (const std::exception &error) {
        std::cerr << "Failed to run Scenario Runner sample: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
