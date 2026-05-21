#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_library(vgf_runtime STATIC
    vgf.cpp
)

target_include_directories(vgf_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vgf_runtime
    PUBLIC
        Vulkan::Headers
    PRIVATE
        VGF::vgf
        VGF::vgf-utils
)

target_compile_options(vgf_runtime PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

if(SCENARIO_RUNNER_BUILD_TESTS)
    include(gtest)
    include(CTest)

    add_executable(vgf_runtime_tests
        vgf_runtime_tests.cpp
    )
    target_link_libraries(vgf_runtime_tests PRIVATE
        GTest::gtest_main
        VGF::vgf
        vgf_runtime
    )
    target_compile_options(vgf_runtime_tests PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

    if(NOT ANDROID)
        gtest_discover_tests(vgf_runtime_tests PROPERTIES LABELS vgf_runtime_tests)
    endif()
endif()
