#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
    add_executable(hlslc
        hlsl/main.cpp
        ../hlsl_compiler.cpp)
    target_include_directories(hlslc SYSTEM PRIVATE
        ../
        ${DXC_PATH}/include
    )
    target_link_libraries(hlslc
        PUBLIC
            dxcompiler
            argparse::argparse
    )
    target_compile_options(hlslc PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

    if(WIN32)
        add_custom_command(TARGET hlslc POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:dxcompiler>"
                "$<TARGET_FILE_DIR:hlslc>"
        )
        if(TARGET dxil)
            add_custom_command(TARGET hlslc POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:dxil>"
                    "$<TARGET_FILE_DIR:hlslc>"
            )
        endif()
    endif()
endif()

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


add_executable(png_utils
    png_utils/main.cpp)
target_include_directories(png_utils PRIVATE
    ../)
target_link_libraries(png_utils
  PUBLIC
    argparse::argparse
    ScenarioRunnerLib
    ${CMAKE_DL_LIBS}
  PRIVATE
    stb::stb
)
target_compile_options(png_utils PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})


if(NOT (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR) AND ML_SDK_GENERATE_CPACK)
    if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
        install(TARGETS hlslc EXPORT ai_ml_sdk_for_vulkanConfig)
    endif()
    install(TARGETS glslc EXPORT ai_ml_sdk_for_vulkanConfig)
    install(TARGETS dds_utils EXPORT ai_ml_sdk_for_vulkanConfig)
    install(TARGETS png_utils EXPORT ai_ml_sdk_for_vulkanConfig)
else()
    if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
        install(TARGETS hlslc EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
    endif()
    install(TARGETS glslc EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
    install(TARGETS dds_utils EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
    install(TARGETS png_utils EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
endif()
