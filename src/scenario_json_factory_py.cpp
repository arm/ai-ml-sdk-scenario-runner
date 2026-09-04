/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_desc.hpp"
#include "scenario_json_factory.hpp"
#include "scenario_options.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <filesystem>

namespace py = pybind11;

namespace mlsdk::scenariorunner {
namespace {

std::unique_ptr<IScenario> makeFromFile(const std::filesystem::path &scenarioFile, const ScenarioOptions &options) {
    const auto workDir = scenarioFile.parent_path();
    ScenarioSpec scenarioSpec{scenarioFile, workDir};
    return ScenarioJsonFactory::make(options, scenarioSpec);
}

} // namespace
} // namespace mlsdk::scenariorunner

void pyInitScenarioJsonFactory(py::module_ &m) {
    using namespace mlsdk::scenariorunner;

    py::class_<ScenarioJsonFactory>(m, "ScenarioJsonFactory")
        .def_static("make", &makeFromFile, py::arg("scenario_file"), py::kw_only(),
                    py::arg("options") = ScenarioOptions{});
}
