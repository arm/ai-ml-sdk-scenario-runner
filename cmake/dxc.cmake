#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

include(version)

if(SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT)
    set(DXC_PATH "DXC-NOTFOUND" CACHE PATH "Path to DXC (DirectXShaderCompiler)")
    set(dxc_VERSION "unknown")

    if(EXISTS ${DXC_PATH}/CMakeLists.txt)
        include(ExternalProject)

        set(DXC_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/dxc-build")
        set(DXC_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/dxc-install")

        if(WIN32)
            set(DXC_DXCOMPILER_IMPORTED_LOCATION "${DXC_BUILD_DIR}/bin/dxcompiler.dll")
            set(DXC_DXCOMPILER_IMPLIB "${DXC_BUILD_DIR}/lib/dxcompiler.lib")
            set(DXC_DXIL_IMPORTED_LOCATION "${DXC_BUILD_DIR}/bin/dxil.dll")
            set(DXC_DXIL_IMPLIB "${DXC_BUILD_DIR}/lib/dxil.lib")
            set(DXC_BYPRODUCTS
                "${DXC_DXCOMPILER_IMPORTED_LOCATION}"
                "${DXC_DXCOMPILER_IMPLIB}"
            )
        else()
            if(DEFINED DXC_BUILD_DIR)
                list(APPEND CMAKE_BUILD_RPATH "${DXC_BUILD_DIR}/lib;${DXC_INSTALL_DIR}/lib")
                list(APPEND CMAKE_INSTALL_RPATH "${DXC_BUILD_DIR}/lib;;${DXC_INSTALL_DIR}/lib")

            endif()
            set(DXC_DXCOMPILER_IMPORTED_LOCATION "${DXC_BUILD_DIR}/lib/libdxcompiler.so")
            set(DXC_DXIL_IMPORTED_LOCATION "${DXC_BUILD_DIR}/lib/libdxil.so")
            set(DXC_BYPRODUCTS
                "${DXC_DXCOMPILER_IMPORTED_LOCATION}"
                "${DXC_DXIL_IMPORTED_LOCATION}"
            )
        endif()

        ExternalProject_Add(dxc_ep
            SOURCE_DIR "${DXC_PATH}"
            BINARY_DIR "${DXC_BUILD_DIR}"
            CMAKE_ARGS
                -C${DXC_PATH}/cmake/caches/PredefinedParams.cmake
                -DCMAKE_INSTALL_PREFIX=${DXC_INSTALL_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DLLVM_INCLUDE_TESTS=OFF
                -DLLVM_BUILD_TESTS=OFF
                -DCLANG_INCLUDE_TESTS=OFF
                -DCLANG_BUILD_TESTS=OFF
                -DHLSL_INCLUDE_TESTS=OFF
                -DHLSL_DISABLE_SOURCE_GENERATION=ON
                $<$<BOOL:${MSVC}>:-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded>
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS
                ${DXC_BYPRODUCTS}
        )

        if(WIN32)
            add_library(dxcompiler SHARED IMPORTED GLOBAL)
            set_target_properties(dxcompiler PROPERTIES
                IMPORTED_LOCATION             "${DXC_DXCOMPILER_IMPORTED_LOCATION}"
                IMPORTED_IMPLIB               "${DXC_DXCOMPILER_IMPLIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${DXC_PATH}/include"
            )

            add_library(dxil SHARED IMPORTED GLOBAL)
            set_target_properties(dxil PROPERTIES
                IMPORTED_LOCATION             "${DXC_DXIL_IMPORTED_LOCATION}"
                IMPORTED_IMPLIB               "${DXC_DXIL_IMPLIB}"
            )
        else()
            add_library(dxcompiler SHARED IMPORTED GLOBAL)
            set_target_properties(dxcompiler PROPERTIES
                IMPORTED_LOCATION             "${DXC_DXCOMPILER_IMPORTED_LOCATION}"
                INTERFACE_INCLUDE_DIRECTORIES "${DXC_PATH}/include"
            )

            add_library(dxil SHARED IMPORTED GLOBAL)
            set_target_properties(dxil PROPERTIES
                IMPORTED_LOCATION             "${DXC_DXIL_IMPORTED_LOCATION}"
            )
        endif()

        add_dependencies(dxcompiler dxc_ep)
        add_dependencies(dxil dxc_ep)

        if(NOT TARGET dxc)
            add_custom_target(dxc)
            add_dependencies(dxc dxc_ep)
        endif()

        mlsdk_get_git_revision(${DXC_PATH} dxc_VERSION)
    else()
        find_package(dxc REQUIRED CONFIG)
    endif()
else()
    message(WARNING "DirectXShaderCompiler is not supported on the current platform")
endif()
