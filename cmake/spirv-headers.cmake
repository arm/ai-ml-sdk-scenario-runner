#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(SPIRV_HEADERS_PATH "SPIRV_HEADERS-NOTFOUND" CACHE PATH "Path to SPIR-V Headers")
set(SPIRV-Headers_VERSION "unknown")

if(EXISTS ${SPIRV_HEADERS_PATH}/CMakeLists.txt)
    if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
        add_subdirectory(${SPIRV_HEADERS_PATH} spirv-headers SYSTEM EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision(${SPIRV_HEADERS_PATH} SPIRV-Headers_VERSION)
else()
    find_package(SPIRV-Headers REQUIRED CONFIG)

    get_target_property(SPIRV-Headers_SOURCE_DIR SPIRV-Headers::SPIRV-Headers INTERFACE_INCLUDE_DIRECTORIES)
    list(GET SPIRV-Headers_SOURCE_DIR 0 SPIRV-Headers_SOURCE_DIR)
    set(SPIRV-Headers_SOURCE_DIR "${SPIRV-Headers_SOURCE_DIR}/..")
endif()
