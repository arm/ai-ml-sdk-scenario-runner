#
# SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(gtest)
include(CTest)

add_executable(ScenarioRunnerTests
  buffer_tests.cpp
  data_graph_binary_tests.cpp
  dds_reader_tests.cpp
  device_selection_tests.cpp
  graph_diagnostics_tests.cpp
  group_manager_tests.cpp
  guid_tests.cpp
  image_formats_tests.cpp
  image_tests.cpp
  iresource_tests.cpp
  json_parser_tests.cpp
  json_writer_tests.cpp
  logging_tests.cpp
  perf_counter_tests.cpp
  png_reader_tests.cpp
  public_api_tests.cpp
  queue_selection_tests.cpp
  raw_data_tests.cpp
  resource_manager_tests.cpp
  scenario_tests.cpp
  tensor_tests.cpp
  vgf_binding_map_tests.cpp
  vgf_configuration_integration_tests.cpp
  vgf_configuration_tests.cpp
  vgf_view_tests.cpp
  vgf_workload_tests.cpp
  vulkan_startup_tests.cpp
  workload_compute_tests.cpp
  workload_data_graph_tests.cpp
  workload_execution_tests.cpp
)
target_link_libraries(ScenarioRunnerTests PRIVATE
  GTest::gtest_main
  ScenarioRunnerLib
  ${CMAKE_DL_LIBS}
  PRIVATE
      stb::stb
)
target_include_directories(ScenarioRunnerTests PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)
if(ANDROID)
  set(SCENARIO_RUNNER_TEST_SPVASM_DIR "tests/spvasm")
else()
  set(SCENARIO_RUNNER_TEST_SPVASM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources/spvasm")
endif()
target_compile_definitions(ScenarioRunnerTests PRIVATE
  SCENARIO_RUNNER_TEST_SPVASM_DIR="${SCENARIO_RUNNER_TEST_SPVASM_DIR}"
)
target_compile_options(ScenarioRunnerTests PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})
scenario_runner_enable_coverage(ScenarioRunnerTests)

if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
  target_sources(ScenarioRunnerTests PRIVATE hlsl_compiler_tests.cpp)
endif()

if(NOT ANDROID)
  gtest_discover_tests(ScenarioRunnerTests PROPERTIES LABELS ScenarioRunnerTests)

  enable_testing()

  # pytest tests depend on spirv tools and vgfpy
  add_custom_target(scenario-runner-tests ALL DEPENDS spirv-as spirv-val vgfpy)
  if(NOT (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR) AND ML_SDK_GENERATE_CPACK)
    install(TARGETS spirv-as EXPORT ai_ml_sdk_for_vulkanConfig)
    install(TARGETS spirv-val EXPORT ai_ml_sdk_for_vulkanConfig)
  else()
    install(TARGETS spirv-as EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
    install(TARGETS spirv-val EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
  endif()
endif()
