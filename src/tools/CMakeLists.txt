#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

add_executable(glslc
    glslc/main.cpp
    ../glsl_compiler.cpp
)
target_include_directories(glslc PRIVATE
    ../
)
target_link_libraries(glslc
  PUBLIC
    glslang::glslang
    glslang::SPIRV
    glslang::glslang-default-resource-limits
    argparse::argparse
)
target_compile_options(glslc PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})


add_executable(dds_utils
    dds_utils/main.cpp)
target_include_directories(dds_utils PRIVATE
    ../)
target_link_libraries(dds_utils
  PUBLIC
    argparse::argparse
    ScenarioRunnerLib
    ${CMAKE_DL_LIBS}
)
target_compile_options(dds_utils PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

install(TARGETS glslc EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
install(TARGETS dds_utils EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
