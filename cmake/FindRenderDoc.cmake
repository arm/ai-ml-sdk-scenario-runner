#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(RenderDoc_INCLUDE_DIRS "" CACHE PATH "Path to RenderDoc includes directories")
set(RenderDoc_LIBRARIES "" CACHE PATH "Path to RenderDoc libraries directories")
set(RenderDoc_BINARY "" CACHE PATH "Path to RenderDoc binary")

if(WIN32)
    set(RenderDoc_SEARCH_PATHS
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
    if(EXISTS "/usr/include/renderdoc_app.h")
        set(RenderDoc_INCLUDE_DIRS "/usr/include" CACHE PATH "Path to RenderDoc include directories" FORCE)
    endif()
    if(EXISTS "/usr/lib/librenderdoc.so")
        set(RenderDoc_LIBRARIES "/usr/lib/librenderdoc.so" CACHE PATH "Path to RenderDoc libraries directories" FORCE)
    endif()
    if(EXISTS "/usr/bin/qrenderdoc")
        set(RenderDoc_BINARY "/usr/bin/qrenderdoc" CACHE PATH "Path to RenderDoc binary" FORCE)
    endif()
else()
    message(FATAL "Don't know how to find RenderDoc.")
endif()

find_package_handle_standard_args(RenderDoc
    REQUIRED_VARS RenderDoc_INCLUDE_DIRS RenderDoc_LIBRARIES RenderDoc_BINARY
    FAIL_MESSAGE "RenderDoc package not found. Please install it and set RenderDoc variables manually."
)
