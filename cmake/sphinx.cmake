#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(SPHINX_BLD_DIR "${SCENARIO_RUNNER_DOCS_DIR}/out")
set(SPHINX_INDEX_HTML "${SPHINX_BLD_DIR}/index.html")
set(SPHINX_STAMP "${SCENARIO_RUNNER_DOCS_DIR}/sphinx.stamp")
set(DOXYGEN_XML_GEN "${DOXYGEN_OUTPUT_DIR}/xml")

add_custom_command(
    OUTPUT "${SPHINX_STAMP}" "${SPHINX_INDEX_HTML}"
    DEPENDS ${DOC_STAGED_FILES} "${DOC_STAGED_MANIFEST}"
        "${DOXYGEN_STAMP}" "${DOXYGEN_INDEX_XML}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${SPHINX_BLD_DIR}"
    COMMAND "${SPHINX_EXECUTABLE}" -b html -W
        "-Dbreathe_projects.ScenarioRunner=${DOXYGEN_XML_GEN}"
        "${SPHINX_SRC_DIR}" "${SPHINX_BLD_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${SPHINX_STAMP}"
    WORKING_DIRECTORY "${SCENARIO_RUNNER_DOCS_BINARY_DIR}"
    COMMENT "Generating API documentation with Sphinx"
    VERBATIM)

add_custom_target(scenario_runner_sphx_doc
    DEPENDS "${SPHINX_STAMP}" "${SPHINX_INDEX_HTML}" scenario_runner_doxy_doc)
