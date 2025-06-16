#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(VULKAN_HEADERS_PATH "VULKAN_HEADERS-NOTFOUND" CACHE PATH "Path to Vulkan Headers")
set(VulkanHeaders_VERSION "unknown")

if(EXISTS ${VULKAN_HEADERS_PATH}/CMakeLists.txt)
    if(NOT TARGET Vulkan::Headers)
        add_subdirectory(${VULKAN_HEADERS_PATH} vulkan-headers SYSTEM EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision(${VULKAN_HEADERS_PATH} VulkanHeaders_VERSION)
else()
    find_package(VulkanHeaders CONFIG REQUIRED HINTS ${VULKAN_HEADERS_PATH})
endif()
