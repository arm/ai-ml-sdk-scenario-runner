# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "scenario_fuzzer requires clang/LLVM libFuzzer.")
endif()

message(STATUS "Building Scenario Runner fuzzers (clang/libFuzzer + ASan + UBSan)")

function(add_scenario_fuzzer target)
    add_executable(${target}
        main.cpp
        fuzz_cpp_api.cpp
    )

    target_link_libraries(${target} PRIVATE
        ScenarioRunnerFuzzerLib
        ${CMAKE_DL_LIBS}
    )
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/src
    )
    target_compile_definitions(${target} PRIVATE
        SCENARIO_FUZZER_RESOURCE_DIR="${PROJECT_SOURCE_DIR}/src/tests/resources"
        SCENARIO_FUZZER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_compile_options(${target} PRIVATE
        ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS}
        -fsanitize=fuzzer,address,undefined
        -fsanitize-recover=address
        -fno-sanitize-recover=undefined
        -fno-omit-frame-pointer
        -fno-sanitize=leak
    )
    target_link_options(${target} PRIVATE
        -fsanitize=fuzzer,address,undefined
        -fsanitize-recover=address
        -fno-sanitize-recover=undefined
        -fno-sanitize=leak
    )
endfunction()

add_scenario_fuzzer(scenario_fuzzer)
add_scenario_fuzzer(scenario_rejection_fuzzer)
target_compile_definitions(scenario_rejection_fuzzer PRIVATE
    SCENARIO_FUZZ_REJECTIONS
)
