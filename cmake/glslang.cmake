#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(GLSLANG_PATH "GLSLANG-NOTFOUND" CACHE PATH "Path to GLSLang")
set(glslang_VERSION "unknown")

if(EXISTS ${GLSLANG_PATH}/CMakeLists.txt)
    if(NOT TARGET glslang)
        add_subdirectory(${GLSLANG_PATH} glslang SYSTEM EXCLUDE_FROM_ALL)
    endif()

    mlsdk_get_git_revision(${GLSLANG_PATH} glslang_VERSION)
else()
    find_package(glslang REQUIRED CONFIG)

    get_target_property(SPIRV_INCLUDE_DIRECTORIES glslang::SPIRV INTERFACE_INCLUDE_DIRECTORIES)
    foreach(include_directory ${SPIRV_INCLUDE_DIRECTORIES})
        target_include_directories(glslang::SPIRV INTERFACE "${include_directory}/glslang")
    endforeach()
endif()
