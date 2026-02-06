#
# SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

#############################################################################
# Add dependencies targets
#############################################################################

add_library(ScenarioRunnerLib
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
    json_writer.cpp
    logging.cpp
    pipeline_cache.cpp
    pipeline.cpp
    raw_data.cpp
    resource_desc.cpp
    scenario.cpp
    scenario_desc.cpp
    tensor.cpp
    utils.cpp
    vgf_view.cpp
    frame_capturer.cpp
    image_formats.cpp
)
target_link_libraries(ScenarioRunnerLib PUBLIC
    glslang::glslang
    glslang::glslang-default-resource-limits
    glslang::SPIRV
    nlohmann_json::nlohmann_json
    SPIRV-Tools-static
    VGF::vgf
    VGF::vgf-utils
    Vulkan::Headers
    PRIVATE
        stb::stb
)

if(SCENARIO_RUNNER_ENABLE_RDOC)
    target_include_directories(ScenarioRunnerLib PRIVATE ${RenderDoc_INCLUDE_DIRS})
    target_compile_definitions(ScenarioRunnerLib PRIVATE ML_SDK_ENABLE_RDOC)
endif()

if(SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT)
    target_compile_definitions(ScenarioRunnerLib PUBLIC SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT)
endif()

if(MOLTEN_VK_SUPPORT)
    target_compile_definitions(ScenarioRunnerLib PRIVATE MOLTEN_VK_SUPPORT)
endif()

target_compile_options(ScenarioRunnerLib PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

add_subdirectory(tools)

if(SCENARIO_RUNNER_BUILD_TESTS)
    include(gtest)

    add_subdirectory(tests)
endif()
