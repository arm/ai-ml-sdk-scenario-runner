#
# SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

#############################################################################
# Add dependencies targets
#############################################################################

set(SCENARIO_RUNNER_LIB_SOURCES
    barrier.cpp
    buffer.cpp
    commands.cpp
    compute.cpp
    context.cpp
    data_manager.cpp
    dds_reader.cpp
    glsl_compiler.cpp
    group_manager.cpp
    image.cpp
    iresource.cpp
    json_reader.cpp
    scenario_json_factory.cpp
    json_writer.cpp
    logging.cpp
    optical_flow_utils.cpp
    pipeline_cache.cpp
    pipeline.cpp
    raw_data.cpp
    resource_desc.cpp
    resource_manager.cpp
    scenario.cpp
    scenario_builder.cpp
    scenario_desc.cpp
    tensor.cpp
    utils.cpp
    vgf_view.cpp
    frame_capturer.cpp
    image_formats.cpp
    png_reader.cpp
)
if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
    list(APPEND SCENARIO_RUNNER_LIB_SOURCES hlsl_compiler.cpp)
endif()

function(srlib_configure_target target_name)
    target_include_directories(${target_name} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(${target_name}
        PUBLIC
            glslang::glslang
            glslang::glslang-default-resource-limits
            glslang::SPIRV
            nlohmann_json::nlohmann_json
            SPIRV-Tools-static
            VGF::vgf
            VGF::vgf-utils
            Vulkan::Headers
            ${CMAKE_DL_LIBS}
        PRIVATE
            stb::stb
    )
    if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
        target_link_libraries(${target_name} PUBLIC scenario_runner_dxc)
        target_compile_definitions(${target_name} PRIVATE SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
    endif()
    if(SCENARIO_RUNNER_ENABLE_RDOC)
        target_include_directories(${target_name} PRIVATE ${RenderDoc_INCLUDE_DIRS})
        target_compile_definitions(${target_name} PRIVATE ML_SDK_ENABLE_RDOC)
    endif()

    if(SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT)
        target_compile_definitions(${target_name} PUBLIC SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT)
    endif()

    target_compile_options(${target_name} PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})
endfunction()

add_library(ScenarioRunnerLib ${SCENARIO_RUNNER_LIB_SOURCES})
srlib_configure_target(ScenarioRunnerLib)

if(SCENARIO_RUNNER_BUILD_SHARED)
    add_library(ScenarioRunnerLib_shared SHARED ${SCENARIO_RUNNER_LIB_SOURCES})
    srlib_configure_target(ScenarioRunnerLib_shared)
    set_target_properties(ScenarioRunnerLib_shared PROPERTIES OUTPUT_NAME ScenarioRunnerLib)
endif()

if(SCENARIO_RUNNER_ENABLE_FUZZER)
    add_library(ScenarioRunnerFuzzerLib STATIC ${SCENARIO_RUNNER_LIB_SOURCES})
    srlib_configure_target(ScenarioRunnerFuzzerLib)
    target_compile_options(ScenarioRunnerFuzzerLib PRIVATE
        -fsanitize=fuzzer-no-link,address,undefined
        -fsanitize-recover=address
        -fno-sanitize-recover=undefined
        -fno-omit-frame-pointer
        -fno-sanitize=leak
        -fno-sanitize=vptr
    )
endif()

if(SCENARIO_RUNNER_BUILD_PYLIB AND NOT ANDROID)
    include(pybind11)
    pybind11_add_module(scenario_runner_py
        scenario_runner_py.cpp
        scenario_py.cpp
        scenario_builder_py.cpp
        scenario_json_factory_py.cpp
    )
    target_compile_options(scenario_runner_py PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})
    target_link_libraries(scenario_runner_py PRIVATE ScenarioRunnerLib pybind11::module)
endif()

add_subdirectory(tools)

if(SCENARIO_RUNNER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
