#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

set(SPIRV_TOOLS_PATH "SPIRV_TOOLS-NOTFOUND" CACHE PATH "Path to SPIR-V Tools")
set(SPIRV-Tools_VERSION "unknown")

if(EXISTS "${SPIRV_TOOLS_PATH}/CMakeLists.txt")
    set(SPIRV_TOOLS_PATCH_FILES
        "${CMAKE_CURRENT_LIST_DIR}/../patches/spirv-tools-graph-shape-inference-pass.patch"
    )

    if(SPIRV_TOOLS_PATCH_FILES)
        find_package(Git REQUIRED)

        foreach(SPIRV_TOOLS_PATCH_FILE IN LISTS SPIRV_TOOLS_PATCH_FILES)
            execute_process(
                COMMAND
                    "${GIT_EXECUTABLE}"
                    -C
                    "${SPIRV_TOOLS_PATH}"
                    apply
                    --reverse
                    --check
                    "${SPIRV_TOOLS_PATCH_FILE}"
                RESULT_VARIABLE SPIRV_TOOLS_PATCH_REVERSE_CHECK
                OUTPUT_VARIABLE SPIRV_TOOLS_PATCH_REVERSE_CHECK_OUTPUT
                ERROR_VARIABLE SPIRV_TOOLS_PATCH_REVERSE_CHECK_ERROR
            )

            if(SPIRV_TOOLS_PATCH_REVERSE_CHECK EQUAL 0)
                message(STATUS "SPIR-V Tools patch ${SPIRV_TOOLS_PATCH_FILE} is already applied")
                continue()
            endif()

            execute_process(
                COMMAND
                    "${GIT_EXECUTABLE}"
                    -C
                    "${SPIRV_TOOLS_PATH}"
                    -c
                    user.name=svc_sdk
                    -c
                    user.email=svc_sdk@arm.com
                    am
                    "${SPIRV_TOOLS_PATCH_FILE}"
                RESULT_VARIABLE SPIRV_TOOLS_APPLY_AND_COMMIT_PATCH
                OUTPUT_VARIABLE SPIRV_TOOLS_APPLY_AND_COMMIT_PATCH_OUTPUT
                ERROR_VARIABLE SPIRV_TOOLS_APPLY_AND_COMMIT_PATCH_ERROR
            )
            if(SPIRV_TOOLS_APPLY_AND_COMMIT_PATCH EQUAL 0)
                execute_process(
                    COMMAND "${GIT_EXECUTABLE}" -C "${SPIRV_TOOLS_PATH}" log -1 --oneline
                    OUTPUT_VARIABLE SPIRV_TOOLS_PATCH_COMMIT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                message(STATUS "SPIR-V Tools patch ${SPIRV_TOOLS_PATCH_COMMIT} applied")
            else()
                execute_process(
                    COMMAND "${GIT_EXECUTABLE}" -C "${SPIRV_TOOLS_PATH}" am --abort
                    OUTPUT_QUIET
                    ERROR_QUIET
                )
                message(STATUS "${SPIRV_TOOLS_APPLY_AND_COMMIT_PATCH}")
                message(STATUS "${SPIRV_TOOLS_APPLY_AND_COMMIT_PATCH_OUTPUT}")
                message(STATUS "${SPIRV_TOOLS_APPLY_AND_COMMIT_PATCH_ERROR}")
                message(FATAL_ERROR "Failed to apply SPIR-V Tools patch ${SPIRV_TOOLS_PATCH_FILE}")
            endif()
        endforeach()
    endif()

    if(NOT TARGET SPIRV-Tools)
        option(SPIRV_SKIP_TESTS "" ON)
        option(SPIRV_WERROR "" OFF)

        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            add_subdirectory("${SPIRV_TOOLS_PATH}" spirv-tools EXCLUDE_FROM_ALL)
        else()
            add_subdirectory("${SPIRV_TOOLS_PATH}" spirv-tools SYSTEM EXCLUDE_FROM_ALL)
        endif()
    endif()

    # AppleClang searches /usr/local/include before command-line system include
    # paths. Use normal include ordering for the patched checkout while retaining
    # system-header diagnostics for SPIR-V Tools headers.
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        foreach(SPIRV_TOOLS_INTERNAL_TARGET IN ITEMS
                SPIRV-Tools SPIRV-Tools-static SPIRV-Tools-shared SPIRV-Tools-opt)
            if(TARGET "${SPIRV_TOOLS_INTERNAL_TARGET}")
                get_target_property(SPIRV_TOOLS_ALIASED_TARGET
                    "${SPIRV_TOOLS_INTERNAL_TARGET}" ALIASED_TARGET)
                if(NOT SPIRV_TOOLS_ALIASED_TARGET)
                    target_compile_options("${SPIRV_TOOLS_INTERNAL_TARGET}" INTERFACE
                        "$<$<COMPILE_LANGUAGE:CXX>:--system-header-prefix=spirv-tools/>")
                endif()
            endif()
        endforeach()
    endif()

    mlsdk_get_git_revision("${SPIRV_TOOLS_PATH}" SPIRV-Tools_VERSION)
else()
    find_package(SPIRV-Tools REQUIRED CONFIG)
    set(SPIRV-Tools_VERSION "unknown")
endif()
