#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
include(cmake/doxygen.cmake)
include(cmake/sphinx.cmake)

if(NOT DOXYGEN_FOUND OR NOT SPHINX_FOUND)
    return()
endif()

file(MAKE_DIRECTORY ${SPHINX_GEN_DIR})

set(DOC_SRC_FILES_FULL_PATHS "")
# Copy MD files for inclusion into the published docs
set(MD_FILES CONTRIBUTING.md README.md SECURITY.md)
foreach(MD_FILE IN LISTS MD_FILES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${MD_FILE} ${SPHINX_GEN_DIR}/${MD_FILE} COPYONLY)
    list(APPEND DOC_SRC_FILES_FULL_PATHS ${SPHINX_GEN_DIR}/${MD_FILE})
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/Apache-2.0.txt ${SPHINX_GEN_DIR}/LICENSES/Apache-2.0.txt COPYONLY)

file(GLOB_RECURSE DOC_SRC_FILES CONFIGURE_DEPENDS RELATIVE ${SCEN_RUN_DOCS_SRC_DIR} ${SCEN_RUN_DOCS_SRC_DIR}/*)
foreach(SRC_IN IN LISTS DOC_SRC_FILES)
    set(DOC_SRC_FILE "${SCEN_RUN_DOCS_SRC_DIR}/${SRC_IN}")
    set(DOC_IN_FILE "${SPHINX_SRC_DIR}/${SRC_IN}")
    configure_file(${DOC_SRC_FILE} ${DOC_IN_FILE} COPYONLY)
    list(APPEND DOC_SRC_FILES_FULL_PATHS ${DOC_IN_FILE})
endforeach()

add_custom_command(
    OUTPUT ${SPHINX_INDEX_HTML}
    DEPENDS ${DOC_SRC_FILES_FULL_PATHS}
    COMMAND ${SPHINX_EXECUTABLE} -b html -W -Dbreathe_projects.MLSDK=${DOXYGEN_XML_GEN} ${SPHINX_SRC_DIR} ${SPHINX_BLD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Sphinx"
    VERBATIM
)

# Main target to build the docs
add_custom_target(scenario_runner_doc ALL DEPENDS scenario_runner_doxy_doc scenario_runner_sphx_doc)
