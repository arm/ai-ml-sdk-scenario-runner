#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Shared by the native project and the standalone documentation project.
get_filename_component(SCENARIO_RUNNER_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(SCENARIO_RUNNER_DOCS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(SCENARIO_RUNNER_DOCS_DIR "${SCENARIO_RUNNER_DOCS_BINARY_DIR}/docs")
list(APPEND CMAKE_MODULE_PATH "${SCENARIO_RUNNER_REPO_ROOT}/cmake")

if(SCENARIO_RUNNER_DOCS_STANDALONE)
    find_package(Doxygen 1.9.1 REQUIRED)
    find_package(Sphinx REQUIRED)
else()
    find_package(Doxygen)
    find_package(Sphinx)
    if(NOT DOXYGEN_FOUND OR NOT SPHINX_FOUND)
        message(STATUS "Doxygen or Sphinx is not available. Cannot generate documentation.")
        return()
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(WARNING "Cannot build the documentation when cross-compiling. Skipping.")
        return()
    endif()
    if(NOT TARGET scenario_runner_py)
        message(FATAL_ERROR "Python API documentation requires SCENARIO_RUNNER_BUILD_PYLIB.")
    endif()
endif()

set(SPHINX_SRC_DIR_IN "${SCENARIO_RUNNER_REPO_ROOT}/docs/source")
set(SPHINX_SRC_DIR "${SCENARIO_RUNNER_DOCS_DIR}/in")
set(SPHINX_GEN_DIR "${SCENARIO_RUNNER_DOCS_DIR}/generated")
set(DOC_STAGED_FILES)

foreach(SRC_IN README.md CONTRIBUTING.md SECURITY.md LICENSES/Apache-2.0.txt)
    set(DOC_SOURCE_FILE "${SPHINX_GEN_DIR}/${SRC_IN}")
    configure_file("${SCENARIO_RUNNER_REPO_ROOT}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

set(DOC_SOURCE_FILE "${SPHINX_GEN_DIR}/scenario_runner_help.txt")
configure_file("${SCENARIO_RUNNER_REPO_ROOT}/docs/help/scenario_runner_help.rst"
    "${DOC_SOURCE_FILE}" COPYONLY)
list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")

file(GLOB_RECURSE DOC_SRC_FILES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${SPHINX_SRC_DIR_IN}" "${SPHINX_SRC_DIR_IN}/*")
foreach(SRC_IN IN LISTS DOC_SRC_FILES)
    set(DOC_SOURCE_FILE "${SPHINX_SRC_DIR}/${SRC_IN}")
    configure_file("${SPHINX_SRC_DIR_IN}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

# Preserve the relative paths used by the API guides' literalinclude directives.
foreach(SAMPLE_FILE 2_run_compute.cpp 4_run_json.cpp
        python/1_run_json.py python/2_run_compute.py)
    set(DOC_SOURCE_FILE "${SPHINX_SRC_DIR}/samples/${SAMPLE_FILE}")
    configure_file("${SCENARIO_RUNNER_REPO_ROOT}/samples/${SAMPLE_FILE}"
        "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

# Doxygen reads checked-in top-level public inputs directly (RECURSIVE=NO).
file(GLOB DOC_DOXYGEN_FILES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    "${SCENARIO_RUNNER_REPO_ROOT}/include/scenario_runner/*.c"
    "${SCENARIO_RUNNER_REPO_ROOT}/include/scenario_runner/*.cpp"
    "${SCENARIO_RUNNER_REPO_ROOT}/include/scenario_runner/*.h"
    "${SCENARIO_RUNNER_REPO_ROOT}/include/scenario_runner/*.hpp"
    "${SCENARIO_RUNNER_REPO_ROOT}/include/scenario_runner/*.md"
    "${SCENARIO_RUNNER_REPO_ROOT}/include/scenario_runner/*.dox")

set(DOC_STAGED_MANIFEST "${SCENARIO_RUNNER_DOCS_DIR}/staged-files.txt")
if(EXISTS "${DOC_STAGED_MANIFEST}")
    file(STRINGS "${DOC_STAGED_MANIFEST}" DOC_PREVIOUS_FILES)
    foreach(DOC_PREVIOUS_FILE IN LISTS DOC_PREVIOUS_FILES)
        cmake_path(IS_PREFIX SCENARIO_RUNNER_DOCS_DIR "${DOC_PREVIOUS_FILE}" NORMALIZE DOC_IS_STAGED)
        if(DOC_IS_STAGED AND NOT DOC_PREVIOUS_FILE IN_LIST DOC_STAGED_FILES)
            file(REMOVE "${DOC_PREVIOUS_FILE}")
        endif()
    endforeach()
endif()
list(JOIN DOC_STAGED_FILES "\n" DOC_STAGED_CONTENT)
file(CONFIGURE OUTPUT "${DOC_STAGED_MANIFEST}" CONTENT "${DOC_STAGED_CONTENT}\n" @ONLY)
set(DOC_DOXYGEN_MANIFEST "${SCENARIO_RUNNER_DOCS_DIR}/doxygen-inputs.txt")
list(JOIN DOC_DOXYGEN_FILES "\n" DOC_DOXYGEN_CONTENT)
file(CONFIGURE OUTPUT "${DOC_DOXYGEN_MANIFEST}" CONTENT "${DOC_DOXYGEN_CONTENT}\n" @ONLY)

include("${SCENARIO_RUNNER_REPO_ROOT}/cmake/doxygen.cmake")
include("${SCENARIO_RUNNER_REPO_ROOT}/cmake/sphinx.cmake")

add_custom_target(scenario_runner_doc ALL
    DEPENDS scenario_runner_doxy_doc scenario_runner_sphx_doc
    SOURCES "${SPHINX_SRC_DIR}/index.rst")
