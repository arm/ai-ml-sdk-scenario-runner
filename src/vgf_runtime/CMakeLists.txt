#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_library(vgf_runtime STATIC
    src/session.cpp
    src/vgf.cpp
)

get_target_property(VGF_RUNTIME_VULKAN_INCLUDE_DIRS Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(VGF_RUNTIME_VULKAN_COMPILE_DEFINITIONS Vulkan::Headers INTERFACE_COMPILE_DEFINITIONS)
get_target_property(VGF_RUNTIME_VULKAN_COMPILE_OPTIONS Vulkan::Headers INTERFACE_COMPILE_OPTIONS)
get_target_property(VGF_RUNTIME_VGF_INCLUDE_DIRS VGF::vgf INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(VGF_RUNTIME_VGF_COMPILE_DEFINITIONS VGF::vgf INTERFACE_COMPILE_DEFINITIONS)
get_target_property(VGF_RUNTIME_VGF_COMPILE_OPTIONS VGF::vgf INTERFACE_COMPILE_OPTIONS)
get_target_property(VGF_RUNTIME_VGF_UTILS_INCLUDE_DIRS VGF::vgf-utils INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(VGF_RUNTIME_VGF_UTILS_COMPILE_DEFINITIONS VGF::vgf-utils INTERFACE_COMPILE_DEFINITIONS)
get_target_property(VGF_RUNTIME_VGF_UTILS_COMPILE_OPTIONS VGF::vgf-utils INTERFACE_COMPILE_OPTIONS)

foreach(prop
    VGF_RUNTIME_VULKAN_COMPILE_DEFINITIONS
    VGF_RUNTIME_VULKAN_COMPILE_OPTIONS
    VGF_RUNTIME_VGF_COMPILE_DEFINITIONS
    VGF_RUNTIME_VGF_COMPILE_OPTIONS
    VGF_RUNTIME_VGF_UTILS_COMPILE_DEFINITIONS
    VGF_RUNTIME_VGF_UTILS_COMPILE_OPTIONS)
    if("${${prop}}" STREQUAL "${prop}-NOTFOUND")
        set(${prop} "")
    endif()
endforeach()

target_include_directories(vgf_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(vgf_runtime SYSTEM
    PUBLIC
        $<BUILD_INTERFACE:${VGF_RUNTIME_VULKAN_INCLUDE_DIRS}>
    PRIVATE
        ${VGF_RUNTIME_VULKAN_INCLUDE_DIRS}
        ${VGF_RUNTIME_VGF_INCLUDE_DIRS}
        ${VGF_RUNTIME_VGF_UTILS_INCLUDE_DIRS}
)

target_compile_definitions(vgf_runtime PRIVATE
    ${VGF_RUNTIME_VULKAN_COMPILE_DEFINITIONS}
    ${VGF_RUNTIME_VGF_COMPILE_DEFINITIONS}
    ${VGF_RUNTIME_VGF_UTILS_COMPILE_DEFINITIONS}
)

target_compile_options(vgf_runtime PRIVATE
    ${VGF_RUNTIME_VULKAN_COMPILE_OPTIONS}
    ${VGF_RUNTIME_VGF_COMPILE_OPTIONS}
    ${VGF_RUNTIME_VGF_UTILS_COMPILE_OPTIONS}
)

set_property(TARGET vgf_runtime PROPERTY INTERFACE_LINK_LIBRARIES
    "$<BUILD_LOCAL_INTERFACE:Vulkan::Headers;VGF::vgf;VGF::vgf-utils>")

target_compile_options(vgf_runtime PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

add_library(${SCENARIO_RUNNER_NAMESPACE}::vgf_runtime ALIAS vgf_runtime)

install(TARGETS vgf_runtime
    EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vgf_runtime/runtime.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vgf_runtime
)

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
        VGF_RUNTIME_ADD_INT32_BUFFERS_SPVASM="${CMAKE_CURRENT_SOURCE_DIR}/tests/resources/spvasm/add_int32_buffers.spvasm"
    )
    target_compile_options(vgf_runtime_tests PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

    if(NOT ANDROID)
        gtest_discover_tests(vgf_runtime_tests PROPERTIES LABELS vgf_runtime_tests)
    endif()
endif()
