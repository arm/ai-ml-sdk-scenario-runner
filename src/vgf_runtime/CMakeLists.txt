#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_library(vgf_runtime STATIC
    session.cpp
    vgf.cpp
)

find_library(VULKAN_LOADER_LIBRARY NAMES vulkan)

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
if(VULKAN_LOADER_LIBRARY)
    target_link_libraries(vgf_runtime PUBLIC ${VULKAN_LOADER_LIBRARY})
endif()

target_compile_options(vgf_runtime PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

if(SCENARIO_RUNNER_BUILD_TESTS)
    include(gtest)
    include(CTest)

    if(NOT VULKAN_LOADER_LIBRARY)
        message(FATAL_ERROR "Vulkan loader library is required to build vgf_runtime_tests")
    endif()

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
        VGF_RUNTIME_MAXPOOL_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/tests/resources/spvasm/maxpool.spvasm"
    )
    target_compile_options(vgf_runtime_tests PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

    if(NOT ANDROID)
        gtest_discover_tests(vgf_runtime_tests PROPERTIES LABELS vgf_runtime_tests)
    endif()
endif()
