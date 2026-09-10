#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_executable(scenario_runner_samples
    samples.cpp
    1_in_memory_buffer.cpp
    2_run_compute.cpp
    3_run_vgf_inference.cpp
    4_run_json.cpp
)
target_link_libraries(scenario_runner_samples PRIVATE ScenarioRunnerLib)
target_compile_options(scenario_runner_samples PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

function(copy_scenario_runner_sample_asset asset)
    add_custom_command(TARGET scenario_runner_samples POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${asset}"
            "$<TARGET_FILE_DIR:scenario_runner_samples>/${asset}"
    )
endfunction()

copy_scenario_runner_sample_asset(increment.comp)
copy_scenario_runner_sample_asset(increment.json)
