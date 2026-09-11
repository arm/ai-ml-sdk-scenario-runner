/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scenario_runner/scenario_json_factory.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <filesystem>

namespace py = pybind11;

void pyInitScenarioJsonFactory(py::module_ &m) {
    using namespace mlsdk::scenariorunner;

    py::class_<ScenarioJsonFactory>(m, "ScenarioJsonFactory")
        .def_static("make", &ScenarioJsonFactory::make, py::arg("scenario_file"), py::kw_only(),
                    py::arg("work_dir") = std::filesystem::path{}, py::arg("output_dir") = std::filesystem::path{},
                    py::arg("options") = ScenarioOptions{}, py::call_guard<py::gil_scoped_release>());

    m.def("load_scenario", &ScenarioJsonFactory::make, py::arg("scenario_file"), py::kw_only(),
          py::arg("work_dir") = std::filesystem::path{}, py::arg("output_dir") = std::filesystem::path{},
          py::arg("options") = ScenarioOptions{}, py::call_guard<py::gil_scoped_release>());
}
