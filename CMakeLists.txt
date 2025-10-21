#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.25)
project(ScenarioRunner
    DESCRIPTION "Scenario Runner"
    LANGUAGES CXX C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(SCENARIO_RUNNER_PACKAGE_NAME ${PROJECT_NAME})
set(SCENARIO_RUNNER_NAMESPACE ${PROJECT_NAME})
if(NOT (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR) AND ML_SDK_GENERATE_CPACK)
    set(SCENARIO_RUNNER_PACKAGE_NAME "ai_ml_sdk_for_vulkan")
    set(SCENARIO_RUNNER_NAMESPACE "ai_ml_sdk_for_vulkan")
else()
    set(CPACK_PACKAGE_NAME "ai_ml_sdk_scenario_runner")
endif()

if(DEFINED ANDROID_PLATFORM)
    message(WARNING "Cross-compiling for Android is currently an experimental feature.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#############################################################################
# Options
#############################################################################
option(SCENARIO_RUNNER_BUILD_TESTS "Build Scenario Runner unit tests" OFF)
option(SCENARIO_RUNNER_ENABLE_CCACHE "Enable CCACHE support" OFF)
option(SCENARIO_RUNNER_ENABLE_RDOC "Build Scenario Runner w/ RDoc support enabled" OFF)
option(SCENARIO_RUNNER_BUILD_DOCS "Build project documentation" OFF)
option(MOLTEN_VK_SUPPORT "Darwin requires MoltenVK" OFF)

# Handle build type, set default as Release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is not defined, building for Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING  "Select the type of build, possible options: None Debug Release RelWithDebInfo and MinSizeRel")
endif()

#############################################################################
# Toolchain configuration
#############################################################################

if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM AND SCENARIO_RUNNER_ENABLE_CCACHE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

include(argparse)
include(json)
include(vgf-lib)
include(vulkan-headers)
include(spirv-headers)
include(spirv-tools)
include(glslang)
include(version)

if(SCENARIO_RUNNER_ENABLE_RDOC)
    find_package(RenderDoc REQUIRED)
endif()

add_subdirectory(src)

add_executable(scenario-runner
    src/main.cpp
)
target_link_libraries(scenario-runner PRIVATE
    ScenarioRunnerLib
    argparse::argparse
    ${CMAKE_DL_LIBS}
)
target_compile_options(scenario-runner PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

mlsdk_generate_version_header(
    TARGET scenario-runner
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/version.hpp.in"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp"
    DEPENDENCIES
        argparse
        glslang
        nlohmann_json
        SPIRV-Headers
        SPIRV-Tools
        VGF
        VulkanHeaders
)

add_executable(${SCENARIO_RUNNER_NAMESPACE}::scenario-runner ALIAS scenario-runner)

install(TARGETS scenario-runner EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)

if(ML_SDK_BUILD_DOCS OR SCENARIO_RUNNER_BUILD_DOCS)
    include(docs/docs.cmake)
endif()
unset(SCENARIO_RUNNER_BUILD_DOCS CACHE)

if(SCENARIO_RUNNER_ENABLE_RDOC)
    add_custom_command(TARGET scenario-runner
        POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
            "${RenderDoc_LIBRARIES}"
            $<TARGET_FILE_DIR:scenario-runner>
            COMMENT "Copying RenderDoc libraries to scenario-runner output directory"
    )
endif()
unset(SCENARIO_RUNNER_ENABLE_RDOC CACHE)

###############################################################################
# Packaging
###############################################################################

include(package)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    mlsdk_package(PACKAGE_NAME ${SCENARIO_RUNNER_PACKAGE_NAME}
        NAMESPACE ${SCENARIO_RUNNER_NAMESPACE}
        LICENSES "${CMAKE_CURRENT_LIST_DIR}/LICENSES/Apache-2.0.txt")
endif()
unset(ANDROID_PLATFORM CACHE)
