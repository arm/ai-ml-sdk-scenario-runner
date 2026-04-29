#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
    set(DXC_PATH "DXC-NOTFOUND" CACHE PATH "Path to DXC (DirectXShaderCompiler)")
    set(dxc_VERSION "unknown")

    if(NOT EXISTS "${DXC_PATH}/CMakeLists.txt")
        message(FATAL_ERROR
            "SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT requires DXC_PATH to point to a DirectXShaderCompiler source tree"
        )
    endif()

    find_package(Git REQUIRED)

    set(DXC_PATCH_COMMIT_MESSAGE "dxc-static-link-for-scenario-runner-28-04-2026")
    set(DXC_PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/../patches/dxc-static-link.patch")

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}"
            -C
            "${DXC_PATH}"
            log
            -1
            --fixed-strings
            --format=%H
            --grep=${DXC_PATCH_COMMIT_MESSAGE}
        RESULT_VARIABLE DXC_PATCH_SEARCH_RESULT
        OUTPUT_VARIABLE DXC_PATCH_SEARCH_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE DXC_PATCH_SEARCH_ERROR
    )
    if(NOT DXC_PATCH_SEARCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to query DXC patch state:\n${DXC_PATCH_SEARCH_ERROR}")
    endif()

    if(DXC_PATCH_SEARCH_OUTPUT)
        execute_process(
            COMMAND
                "${GIT_EXECUTABLE}"
                -C
                "${DXC_PATH}"
                apply
                --reverse
                --check
                "${DXC_PATCH_FILE}"
            RESULT_VARIABLE DXC_PATCH_REVERSE_CHECK
            OUTPUT_VARIABLE DXC_PATCH_REVERSE_CHECK_OUTPUT
            ERROR_VARIABLE DXC_PATCH_REVERSE_CHECK_ERROR
        )
        if(DXC_PATCH_REVERSE_CHECK EQUAL 0)
            message(STATUS "DXC patch is already applied")
        else()
            message(FATAL_ERROR
                "Existing DXC commit ${DXC_PATCH_COMMIT_MESSAGE} doesnt match DXC patch file\n"
                "${DXC_PATCH_REVERSE_CHECK_OUTPUT}\n"
                "${DXC_PATCH_REVERSE_CHECK_ERROR}"
            )
        endif()
    else()
        execute_process(
            COMMAND
                "${GIT_EXECUTABLE}"
                -C
                "${DXC_PATH}"
                -c
                user.name=svc_sdk
                -c
                user.email=svc_sdk@arm.com
                am
                "${DXC_PATCH_FILE}"
            RESULT_VARIABLE DXC_APPLY_AND_COMMIT_PATCH
            OUTPUT_VARIABLE DXC_APPLY_AND_COMMIT_PATCH_OUTPUT
            ERROR_VARIABLE DXC_APPLY_AND_COMMIT_PATCH_ERROR
        )
        if(DXC_APPLY_AND_COMMIT_PATCH EQUAL 0)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${DXC_PATH}" log -1 --oneline
                OUTPUT_VARIABLE DXC_PATCH_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            message(STATUS "DXC patch ${DXC_PATCH_COMMIT} applied")
        else()
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${DXC_PATH}" am --abort
                OUTPUT_QUIET
                ERROR_QUIET
            )
            message(STATUS "${DXC_APPLY_AND_COMMIT_PATCH}")
            message(STATUS "${DXC_APPLY_AND_COMMIT_PATCH_OUTPUT}")
            message(STATUS "${DXC_APPLY_AND_COMMIT_PATCH_ERROR}")
            message(FATAL_ERROR "Failed to apply DXC patch")
        endif()
    endif()

    set(DXC_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/dxc-build-worktree")

    include("${DXC_PATH}/cmake/caches/PredefinedParams.cmake")

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
    set(LLVM_APPEND_VC_REV ON CACHE BOOL "" FORCE)
    set(LLVM_DEFAULT_TARGET_TRIPLE "dxil-ms-dx" CACHE STRING "" FORCE)
    set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLVM_OPTIMIZED_TABLEGEN OFF CACHE BOOL "" FORCE)
    set(LLVM_TARGETS_TO_BUILD "None" CACHE STRING "" FORCE)
    set(LIBCLANG_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(CLANG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CLANG_CL OFF CACHE BOOL "" FORCE)
    set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "" FORCE)
    set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "" FORCE)
    set(ENABLE_SPIRV_CODEGEN ON CACHE BOOL "" FORCE)
    set(SPIRV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)

    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(LLVM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CLANG_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(CLANG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(HLSL_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(HLSL_DISABLE_SOURCE_GENERATION ON CACHE BOOL "" FORCE)
    if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # DXC disables checked iterators by default on MSVC. Keep the
        # embedded DXC/LLVM build aligned with the rest of the Debug link.
        set(HLSL_ENABLE_DEBUG_ITERATORS ON CACHE BOOL "" FORCE)
    endif()
    if(MSVC)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded CACHE STRING "" FORCE)
    endif()

    add_subdirectory("${DXC_PATH}" "${DXC_BUILD_DIR}" EXCLUDE_FROM_ALL)

    add_library(scenario_runner_dxc INTERFACE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_libraries(scenario_runner_dxc INTERFACE
            "-Wl,--start-group"
            dxcompiler_static
            LLVMHLSL
            LLVMScalarOpts
            LLVMPassPrinters
            LLVMipo
            "-Wl,--end-group"
        )
    else()
        target_link_libraries(scenario_runner_dxc INTERFACE dxcompiler_static)
    endif()
    target_include_directories(scenario_runner_dxc SYSTEM INTERFACE
        "${DXC_PATH}/include"
        "${DXC_BUILD_DIR}/include"
    )

    if(NOT TARGET dxc)
        add_custom_target(dxc DEPENDS dxcompiler_static)
    endif()

    mlsdk_get_git_revision(${DXC_PATH} dxc_VERSION)
else()
    message(WARNING "DirectXShaderCompiler is not supported on the current platform")
endif()
