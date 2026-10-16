#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

if(PYBIND11_FINDPYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
endif()
include(pybind11)
pybind11_add_module(scenario_runner_py
    "${CMAKE_CURRENT_LIST_DIR}/../scenario_runner_py.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../scenario_py.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../scenario_builder_py.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../scenario_json_factory_py.cpp")
target_compile_options(scenario_runner_py PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})
scenario_runner_enable_coverage(scenario_runner_py)
target_link_libraries(scenario_runner_py PRIVATE ScenarioRunnerLib pybind11::module)
