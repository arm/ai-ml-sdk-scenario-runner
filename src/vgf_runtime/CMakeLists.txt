#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_library(vgf_runtime STATIC
    session.cpp
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
        tests/vgf_runtime_test_utils.hpp
        tests/vgf_runtime_full_tests.cpp
        tests/vgf_runtime_vgf_tests.cpp
    )
    target_link_libraries(vgf_runtime_tests PRIVATE
        GTest::gtest_main
        SPIRV-Tools-static
        VGF::vgf
        vgf_runtime
    )
    target_compile_definitions(vgf_runtime_tests PRIVATE
        VGF_RUNTIME_MAXPOOL_16X16_TO_8X8_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/tests/resources/spvasm/maxpool_16x16_to_8x8.spvasm"
        VGF_RUNTIME_MAXPOOL_8X8_TO_4X4_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/tests/resources/spvasm/maxpool_8x8_to_4x4.spvasm"
    )
    target_compile_options(vgf_runtime_tests PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

    if(NOT ANDROID)
        gtest_discover_tests(vgf_runtime_tests PROPERTIES LABELS vgf_runtime_tests)
    endif()
endif()
