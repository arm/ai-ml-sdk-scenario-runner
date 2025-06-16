#
# SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
find_package(Sphinx)
if(NOT SPHINX_FOUND)
    message("Sphinx is not available. Cannot generate documentation.")
    unset(SPHINX_EXECUTABLE CACHE)
    return()
endif()

set(SCEN_RUN_DOCS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/source)
set(SPHINX_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/in)
set(SPHINX_BLD_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/out)
set(SPHINX_INDEX_HTML ${SPHINX_BLD_DIR}/index.html)
set(DOXYGEN_XML_GEN ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen/xml)
set(SPHINX_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/generated)

add_custom_target(scenario_runner_sphx_doc DEPENDS ${SPHINX_INDEX_HTML} scenario_runner_doxy_doc)
