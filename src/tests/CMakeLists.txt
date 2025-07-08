#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
if(NOT ANDROID)
  include(CTest)

  add_executable(ScenarioRunnerTests
    numpy_test.cpp
    guid_tests.cpp
    json_parser_tests.cpp
    vulkan_startup_tests.cpp
    logging_tests.cpp
  )
  target_link_libraries(ScenarioRunnerTests PRIVATE
    GTest::gtest_main
    ScenarioRunnerLib
    ${CMAKE_DL_LIBS}
  )
  target_include_directories(ScenarioRunnerTests PRIVATE
    ${PROJECT_SOURCE_DIR}/src
  )
  target_compile_options(ScenarioRunnerTests PRIVATE ${ML_SDK_SCENARIO_RUNNER_COMPILE_OPTIONS})

  gtest_discover_tests(ScenarioRunnerTests PROPERTIES LABELS ScenarioRunnerTests)

  enable_testing()

  # pytest tests depend on spirv tools and vgfpy
  add_custom_target(scenario-runner-tests ALL DEPENDS spirv-as spirv-val)
  if(TARGET vgfpy)
    add_dependencies(scenario-runner-tests vgfpy)
  endif()
  if(ML_SDK_GENERATE_CPACK)
    install(TARGETS spirv-as EXPORT ml-sdkConfig)
    install(TARGETS spirv-val EXPORT ml-sdkConfig)
  else()
    install(TARGETS spirv-as EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
    install(TARGETS spirv-val EXPORT ${SCENARIO_RUNNER_PACKAGE_NAME}Config)
  endif()
endif()
