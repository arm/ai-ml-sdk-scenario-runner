/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

extern void pyInitScenario(py::module_ &m);
extern void pyInitScenarioBuilder(py::module_ &m);
extern void pyInitScenarioJsonFactory(py::module_ &m);

PYBIND11_MODULE(scenario_runner_py, m) {
    pyInitScenario(m);
    pyInitScenarioBuilder(m);
    pyInitScenarioJsonFactory(m);
}