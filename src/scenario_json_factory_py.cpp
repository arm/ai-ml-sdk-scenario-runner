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

    py::class_<ScenarioJsonFactory>(m, "ScenarioJsonFactory", "Create scenarios from JSON descriptions.")
        .def_static(
            "make", &ScenarioJsonFactory::make, "Load a JSON scenario and return a ready-to-run :class:`Scenario`.",
            py::arg("scenario_file"), py::kw_only(), py::arg_v("work_dir", std::filesystem::path{}, "Path('.')"),
            py::arg_v("output_dir", std::filesystem::path{}, "Path('.')"),
            py::arg_v("options", ScenarioOptions{}, "ScenarioOptions()"), py::call_guard<py::gil_scoped_release>());

    m.def("load_scenario", &ScenarioJsonFactory::make,
          R"doc(Load a JSON scenario and return a ready-to-run :class:`Scenario`.

Relative resource paths use the scenario file's parent directory unless ``work_dir`` is supplied.)doc",
          py::arg("scenario_file"), py::kw_only(), py::arg_v("work_dir", std::filesystem::path{}, "Path('.')"),
          py::arg_v("output_dir", std::filesystem::path{}, "Path('.')"),
          py::arg_v("options", ScenarioOptions{}, "ScenarioOptions()"), py::call_guard<py::gil_scoped_release>());
}
