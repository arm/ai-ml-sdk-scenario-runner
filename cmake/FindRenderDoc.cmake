#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

set(RenderDoc_ROOT "" CACHE PATH "Path to the RenderDoc installation root")
set(RenderDoc_INCLUDE_DIRS "NOTFOUND" CACHE PATH "Path to RenderDoc includes directories")
set(RenderDoc_LIBRARIES "NOTFOUND" CACHE PATH "Path to RenderDoc libraries directories")
set(RenderDoc_BINARY "NOTFOUND" CACHE PATH "Path to RenderDoc binary")

if(WIN32)
    set(RenderDoc_SEARCH_PATHS)
    if(RenderDoc_ROOT)
        list(APPEND RenderDoc_SEARCH_PATHS "${RenderDoc_ROOT}")
    endif()
    list(APPEND RenderDoc_SEARCH_PATHS
        "$ENV{ProgramFiles}/RenderDoc_For_Arm_GPUs_Windows_64"
        "$ENV{ProgramFiles}/RenderDoc"
    )
    foreach(path ${RenderDoc_SEARCH_PATHS})
        if(EXISTS "${path}/qrenderdoc.exe")
            set(RenderDoc_INCLUDE_DIRS ${path} CACHE PATH "Path to RenderDoc include directories" FORCE)
            set(RenderDoc_LIBRARIES "${path}/renderdoc.dll" CACHE PATH "Path to RenderDoc libraries directories" FORCE)
            set(RenderDoc_BINARY "${path}/qrenderdoc.exe" CACHE PATH "Path to RenderDoc binary" FORCE)
            break()
        endif()
    endforeach()
elseif(LINUX)
    if(RenderDoc_ROOT)
        find_path(RenderDoc_INCLUDE_DIRS
            NAMES renderdoc_app.h
            PATHS "${RenderDoc_ROOT}"
            PATH_SUFFIXES include ""
            NO_DEFAULT_PATH
        )
        find_file(RenderDoc_LIBRARIES
            NAMES librenderdoc.so
            PATHS "${RenderDoc_ROOT}"
            PATH_SUFFIXES lib lib64 ""
            NO_DEFAULT_PATH
        )
        find_program(RenderDoc_BINARY
            NAMES qrenderdoc
            PATHS "${RenderDoc_ROOT}"
            PATH_SUFFIXES bin ""
            NO_DEFAULT_PATH
        )
    endif()
    if(NOT RenderDoc_INCLUDE_DIRS AND EXISTS "/usr/include/renderdoc_app.h")
        set(RenderDoc_INCLUDE_DIRS "/usr/include" CACHE PATH "Path to RenderDoc include directories" FORCE)
    endif()
    if(NOT RenderDoc_LIBRARIES AND EXISTS "/usr/lib/librenderdoc.so")
        set(RenderDoc_LIBRARIES "/usr/lib/librenderdoc.so" CACHE PATH "Path to RenderDoc libraries directories" FORCE)
    endif()
    if(NOT RenderDoc_BINARY AND EXISTS "/usr/bin/qrenderdoc")
        set(RenderDoc_BINARY "/usr/bin/qrenderdoc" CACHE PATH "Path to RenderDoc binary" FORCE)
    endif()
else()
    message(FATAL "Don't know how to find RenderDoc.")
endif()

find_package_handle_standard_args(RenderDoc
    REQUIRED_VARS RenderDoc_INCLUDE_DIRS RenderDoc_LIBRARIES RenderDoc_BINARY
    FAIL_MESSAGE "RenderDoc package not found. Set RenderDoc_ROOT or the RenderDoc_* cache variables manually."
)
