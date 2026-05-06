#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(ML_SDK_VGF_LIB_PATH "ML_SDK_VGF_LIB-NOTFOUND" CACHE PATH "Path to VGF Library")
set(VGF_VERSION "unknown")

if(EXISTS ${ML_SDK_VGF_LIB_PATH}/CMakeLists.txt)
    if(NOT TARGET vgf)
        set(ML_SDK_VGF_LIB_BUILD_TOOLS OFF CACHE BOOL "Build VGF tools" FORCE)
        if(SCENARIO_RUNNER_BUILD_TESTS AND NOT ANDROID)
            set(ML_SDK_VGF_LIB_BUILD_PYLIB ON CACHE BOOL "Build VGF Python Lib")
        endif()
        add_subdirectory(${ML_SDK_VGF_LIB_PATH} vgf-lib EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision(${ML_SDK_VGF_LIB_PATH} VGF_VERSION)
else()
    find_package(VGF REQUIRED CONFIG)
endif()
