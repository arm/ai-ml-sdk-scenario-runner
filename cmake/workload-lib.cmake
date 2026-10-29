#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(ML_SDK_WORKLOAD_LIB_PATH "ML_SDK_WORKLOAD_LIB-NOTFOUND" CACHE PATH
    "Path to ML Workload Library for Vulkan")
set(MLWorkloadLibraryForVulkan_VERSION "unknown")

if(EXISTS "${ML_SDK_WORKLOAD_LIB_PATH}/CMakeLists.txt")
    if(NOT TARGET MLWorkloadLibraryForVulkan::mlworkloadlib)
        add_subdirectory("${ML_SDK_WORKLOAD_LIB_PATH}" workload-lib EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision("${ML_SDK_WORKLOAD_LIB_PATH}" MLWorkloadLibraryForVulkan_VERSION)
else()
    find_package(MLWorkloadLibraryForVulkan REQUIRED CONFIG)
endif()
