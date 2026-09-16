/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_utils.hpp"
#include "samples.hpp"

#include "scenario_runner/scenario_json_factory.hpp"

#include <exception>
#include <iostream>
#include <vector>

int mlsdk::scenariorunner::samples::runJsonSample(std::string_view executable) {
    using namespace mlsdk::scenariorunner;
    using namespace mlsdk::scenariorunner::samples;

    try {
        // API documentation: JSON scenario example begins.
        // JSON memory-resource UIDs remain available for lookup after construction.
        auto scenario = ScenarioJsonFactory::make(assetPath(executable, "increment.json"));
        const auto inputId = scenario->getBufferId("input");
        const auto outputId = scenario->getBufferId("output");

        const std::vector<uint32_t> input{10, 20, 30, 40};
        const std::vector<uint32_t> expected{11, 21, 31, 41};
        scenario->upload(inputId, view(input));
        scenario->run();
        const auto output = scenario->download(outputId);
        // API documentation: JSON scenario example ends.

        if (!matches(output, expected)) {
            std::cerr << "Unexpected JSON scenario output\n";
            return 1;
        }
        std::cout << "JSON scenario output matched the expected values\n";
    } catch (const std::exception &error) {
        std::cerr << "Failed to run JSON sample: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
