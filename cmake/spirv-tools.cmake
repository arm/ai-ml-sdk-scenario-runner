#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(SPIRV_TOOLS_PATH "SPIRV_TOOLS-NOTFOUND" CACHE PATH "Path to SPIR-V Tools")
set(SPIRV-Tools_VERSION "unknown")

if(EXISTS ${SPIRV_TOOLS_PATH}/CMakeLists.txt)
    if(NOT TARGET SPIRV-Tools)
        option(SPIRV_SKIP_TESTS "" ON)
        option(SPIRV_WERROR "" OFF)

        add_subdirectory(${SPIRV_TOOLS_PATH} spirv-tools SYSTEM EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision(${SPIRV_TOOLS_PATH} SPIRV-Tools_VERSION)
else()
    find_package(SPIRV-Tools REQUIRED CONFIG)
    set(SPIRV-Tools_VERSION "unknown")
endif()
