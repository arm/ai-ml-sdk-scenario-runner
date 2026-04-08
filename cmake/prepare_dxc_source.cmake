#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
if(NOT DEFINED DXC_SOURCE_DIR OR DXC_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "DXC_SOURCE_DIR must be defined")
endif()

if(NOT DEFINED DXC_STAGE_DIR OR DXC_STAGE_DIR STREQUAL "")
    message(FATAL_ERROR "DXC_STAGE_DIR must be defined")
endif()

if(NOT DEFINED GIT_EXECUTABLE OR GIT_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "GIT_EXECUTABLE must be defined")
endif()

set(expected_stamp "source=${DXC_SOURCE_DIR}\n")
if(DEFINED DXC_PATCHES AND NOT DXC_PATCHES STREQUAL "")
    foreach(patch_file IN LISTS DXC_PATCHES)
        file(SHA256 "${patch_file}" patch_hash)
        string(APPEND expected_stamp "patch=${patch_file}:${patch_hash}\n")
    endforeach()
endif()

if(EXISTS "${DXC_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${DXC_SOURCE_DIR}" rev-parse HEAD
        RESULT_VARIABLE git_head_result
        OUTPUT_VARIABLE git_head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(git_head_result EQUAL 0)
        string(APPEND expected_stamp "head=${git_head}\n")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${DXC_SOURCE_DIR}" status --porcelain --untracked-files=no
        RESULT_VARIABLE git_status_result
        OUTPUT_VARIABLE git_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(git_status_result EQUAL 0)
        string(APPEND expected_stamp "status=${git_status}\n")
    endif()
endif()

set(stamp_file "${DXC_STAGE_DIR}/.scenario_runner_dxc_stamp")
set(need_refresh TRUE)
if(EXISTS "${stamp_file}")
    file(READ "${stamp_file}" current_stamp)
    if(current_stamp STREQUAL expected_stamp)
        set(need_refresh FALSE)
    endif()
endif()

if(need_refresh)
    file(REMOVE_RECURSE "${DXC_STAGE_DIR}")
    file(MAKE_DIRECTORY "${DXC_STAGE_DIR}")

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${DXC_SOURCE_DIR}" "${DXC_STAGE_DIR}"
        RESULT_VARIABLE copy_result
    )
    if(NOT copy_result EQUAL 0)
        message(FATAL_ERROR "Failed to copy DXC source tree to ${DXC_STAGE_DIR}")
    endif()

    if(DEFINED DXC_PATCHES AND NOT DXC_PATCHES STREQUAL "")
        foreach(patch_file IN LISTS DXC_PATCHES)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace --whitespace=nowarn "${patch_file}"
                WORKING_DIRECTORY "${DXC_STAGE_DIR}"
                RESULT_VARIABLE apply_result
                OUTPUT_VARIABLE apply_stdout
                ERROR_VARIABLE apply_stderr
            )
            if(NOT apply_result EQUAL 0)
                message(FATAL_ERROR
                    "Failed to apply patch ${patch_file}\n"
                    "stdout:\n${apply_stdout}\n"
                    "stderr:\n${apply_stderr}"
                )
            endif()
        endforeach()
    endif()

    file(WRITE "${stamp_file}" "${expected_stamp}")
endif()
