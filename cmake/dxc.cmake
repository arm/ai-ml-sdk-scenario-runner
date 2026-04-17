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

    set(DXC_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/dxc-src")
    set(DXC_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/dxc-build")
    set(DXC_PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/../patches/dxc-static-link.patch"
    )

    execute_process(
        COMMAND
            "${CMAKE_COMMAND}"
            -DDXC_SOURCE_DIR=${DXC_PATH}
            -DDXC_STAGE_DIR=${DXC_SOURCE_DIR}
            -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
            -DDXC_PATCHES=${DXC_PATCHES}
            -P "${CMAKE_CURRENT_LIST_DIR}/prepare_dxc_source.cmake"
        COMMAND_ERROR_IS_FATAL ANY
    )

    include("${DXC_SOURCE_DIR}/cmake/caches/PredefinedParams.cmake")

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

    add_subdirectory("${DXC_SOURCE_DIR}" "${DXC_BUILD_DIR}" EXCLUDE_FROM_ALL)

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
        "${DXC_SOURCE_DIR}/include"
        "${DXC_BUILD_DIR}/include"
    )

    if(NOT TARGET dxc)
        add_custom_target(dxc DEPENDS dxcompiler_static)
    endif()

    mlsdk_get_git_revision(${DXC_PATH} dxc_VERSION)
else()
    message(WARNING "DirectXShaderCompiler is not supported on the current platform")
endif()
