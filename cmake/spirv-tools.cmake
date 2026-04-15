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

        if(MSVC AND SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
            # DXC forces _ITERATOR_DEBUG_LEVEL=0 on MSVC builds unless
            # HLSL_ENABLE_DEBUG_ITERATORS is enabled. Because DXC reuses the
            # parent SPIRV-Tools targets when they already exist, align the
            # reused SPIRV-Tools build with DXC to avoid LNK2038.
            set(SPIRV_TOOLS_EXTRA_DEFINITIONS
                "${SPIRV_TOOLS_EXTRA_DEFINITIONS} /D_ITERATOR_DEBUG_LEVEL=0")
        endif()
        add_subdirectory(${SPIRV_TOOLS_PATH} spirv-tools SYSTEM EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision(${SPIRV_TOOLS_PATH} SPIRV-Tools_VERSION)
else()
    find_package(SPIRV-Tools REQUIRED CONFIG)
    set(SPIRV-Tools_VERSION "unknown")
endif()
