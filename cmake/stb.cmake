#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

set(STB_VERSION "f1c79c02822848a9bed4315b12c8c8f3761e1296" CACHE STRING "Git ref (branch, tag, or commit) for stb headers")
set(STB_BASE_URL "https://raw.githubusercontent.com/nothings/stb/${STB_VERSION}" CACHE STRING "Base URL used to download stb headers")
set(STB_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/stb" CACHE PATH "Directory where stb headers are stored")
set(STB_IMAGE_REQUIRED_VERSION "v2.30")
set(STB_IMAGE_WRITE_REQUIRED_VERSION "v1.16")

function(stb_download_header header_name)
    set(target_path "${STB_DOWNLOAD_DIR}/${header_name}")
    if(NOT EXISTS "${target_path}")
        message(STATUS "Downloading ${header_name} from ${STB_BASE_URL}")
        file(DOWNLOAD
            "${STB_BASE_URL}/${header_name}"
            "${target_path}"
            STATUS download_status
            LOG download_log
        )
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "Failed to download ${header_name} from ${STB_BASE_URL} (status: ${download_status}). Log: ${download_log}")
        endif()
    endif()
endfunction()

file(MAKE_DIRECTORY "${STB_DOWNLOAD_DIR}")

function(stb_extract_version HEADER_BASENAME OUT_VAR)
    set(header_path "${STB_DOWNLOAD_DIR}/${HEADER_BASENAME}")
    if(NOT EXISTS "${header_path}")
        message(FATAL_ERROR "Missing ${HEADER_BASENAME} at ${header_path}")
    endif()
    # Expect version on the first line
    file(STRINGS "${header_path}" first_line LIMIT_COUNT 1)
    string(REGEX MATCH "v[0-9]+\\.[0-9]+" extracted_version "${first_line}")
    if(NOT extracted_version)
        set(extracted_version "unknown")
    endif()
    set(${OUT_VAR} "${extracted_version}" PARENT_SCOPE)
endfunction()

function(stb_ensure_header HEADER_BASENAME EXPECTED_VERSION OUT_VAR)
    stb_download_header("${HEADER_BASENAME}")
    stb_extract_version("${HEADER_BASENAME}" detected_version)
    if(NOT detected_version STREQUAL EXPECTED_VERSION)
        message(FATAL_ERROR
            "Expected ${HEADER_BASENAME} version ${EXPECTED_VERSION}, but found ${detected_version} in ${STB_DOWNLOAD_DIR}/${HEADER_BASENAME}.")
    endif()
    set(${OUT_VAR} "${detected_version}" PARENT_SCOPE)
endfunction()

stb_ensure_header("stb_image.h" "${STB_IMAGE_REQUIRED_VERSION}" stb_image_VERSION)
stb_ensure_header("stb_image_write.h" "${STB_IMAGE_WRITE_REQUIRED_VERSION}" stb_image_write_VERSION)

add_library(stb INTERFACE)
add_library(stb::stb ALIAS stb)
target_include_directories(stb SYSTEM INTERFACE "${STB_DOWNLOAD_DIR}")
