#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Builds the Scenario Runner App"""
import argparse
import pathlib
import platform
import subprocess
import sys

try:
    import argcomplete
except:
    argcomplete = None

SCENARIO_RUNNER_DIR = pathlib.Path(__file__).resolve().parent / ".."
DEPENDENCY_DIR = SCENARIO_RUNNER_DIR / ".." / ".." / "dependencies"
CMAKE_TOOLCHAIN_PATH = SCENARIO_RUNNER_DIR / "cmake" / "toolchain"


def absolute(path):
    return pathlib.Path(path).resolve().as_posix()


class Builder:
    """
    A  class that builds the Scenario Runner.

    Parameters
    ----------
    args : 'dict'
        Dictionary with arguments to build the Scenario Runner.
    """

    def __init__(self, args) -> None:
        self.build_dir = args.build_dir
        self.prefix_path = args.prefix_path
        self.test_dir = pathlib.Path(self.build_dir) / "src" / "tests"
        self.threads = args.threads
        self.run_tests = args.test
        self.build_type = args.build_type
        self.target_platform = args.target_platform
        self.cmake_toolchain_for_android = args.cmake_toolchain_for_android
        self.force_no_debug_symbols_android_build = (
            args.force_no_debug_symbols_android_build
        )
        self.vulkan_headers_path = absolute(args.vulkan_headers_path)
        self.flatc_path = args.flatc_path
        self.vgf_lib_path = absolute(args.vgf_lib_path)
        self.json_path = absolute(args.json_path)
        self.gtest_path = absolute(args.gtest_path)
        self.flatbuffers_path = absolute(args.flatbuffers_path)
        self.glslang_path = absolute(args.glslang_path)
        self.spirv_tools_path = absolute(args.spirv_tools_path)
        self.spirv_headers_path = absolute(args.spirv_headers_path)
        self.argparse_path = absolute(args.argparse_path)
        self.pybind11_path = absolute(args.pybind11_path)
        self.doc = args.doc
        self.enable_gcc_sanitizers = args.enable_gcc_sanitizers
        self.run_linting = args.lint
        self.install = args.install
        self.package = args.package
        self.package_type = args.package_type
        self.package_source = args.package_source
        self.emulation_layer = args.emulation_layer
        self.enable_rdoc = args.enable_rdoc

    def setup_platform_build(self, cmake_cmd):
        if self.target_platform == "host":
            system = platform.system()
            if system == "Linux":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'linux-gcc.cmake'}"
                )
                return True

            if system == "Windows":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'windows-msvc.cmake'}"
                )
                cmake_cmd.append("-DMSVC=ON")
                return True

            print(f"Unsupported host platform {system}", file=sys.stderr)
            return False

        if self.target_platform == "aarch64":
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'linux-aarch64-gcc.cmake'}"
            )
            cmake_cmd.append("-DHAVE_CLONEFILE=0")
            cmake_cmd.append("-DBUILD_TOOLS=OFF")
            cmake_cmd.append("-DBUILD_REGRESS=OFF")
            cmake_cmd.append("-DBUILD_EXAMPLES=OFF")
            cmake_cmd.append("-DBUILD_DOC=OFF")

            cmake_cmd.append("-DBUILD_WSI_WAYLAND_SUPPORT=OFF")
            cmake_cmd.append("-DBUILD_WSI_XLIB_SUPPORT=OFF")
            cmake_cmd.append("-DBUILD_WSI_XLIB_SUPPORT=OFF")
            cmake_cmd.append("-DBUILD_WSI_XCB_SUPPORT=OFF")
            return True

        if self.target_platform == "android":
            print(
                "WARNING: Cross-compiling Scenario Runner for Android is currently an experimental feature."
            )
            if not self.cmake_toolchain_for_android:
                print(
                    "No toolchain path specified for Android cross-compilation",
                    file=sys.stderr,
                )
                return False

            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={self.cmake_toolchain_for_android}"
            )
            cmake_cmd.append("-DCMAKE_FIND_ROOT_PATH=/")
            cmake_cmd.append("-DANDROID_ABI=arm64-v8a")
            cmake_cmd.append("-DANDROID_PLATFORM=android-21")
            cmake_cmd.append("-DANDROID_ALLOW_UNDEFINED_SYMBOLS=ON")
            cmake_cmd.append("-DANDROID_PIE=ON")
            if self.force_no_debug_symbols_android_build:
                cmake_cmd.append("-DCMAKE_CXX_FLAGS_RELEASE=-g0")
            return True

        print(
            f"Incorrect target platform option: {self.target_platform}", file=sys.stderr
        )
        return False

    def run(self):
        cmake_setup_cmd = [
            "cmake",
            "-S",
            str(SCENARIO_RUNNER_DIR),
            "-B",
            self.build_dir,
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            "-DSCENARIO_RUNNER_ENABLE_CCACHE=ON",
            f"-DVULKAN_HEADERS_PATH={self.vulkan_headers_path}",
            f"-DML_SDK_VGF_LIB_PATH={self.vgf_lib_path}",
            f"-DJSON_PATH={self.json_path}",
            f"-DFLATBUFFERS_PATH={self.flatbuffers_path}",
            f"-DSPIRV_TOOLS_PATH={self.spirv_tools_path}",
            f"-DSPIRV_HEADERS_PATH={self.spirv_headers_path}",
            f"-DGLSLANG_PATH={self.glslang_path}",
            f"-DARGPARSE_PATH={self.argparse_path}",
        ]
        if self.prefix_path:
            cmake_setup_cmd.append(f"-DCMAKE_PREFIX_PATH={self.prefix_path}")

        if self.flatc_path:
            cmake_setup_cmd.append(f"-DFLATC_PATH={self.flatc_path}")

        if self.run_tests:
            cmake_setup_cmd.append("-DSCENARIO_RUNNER_BUILD_TESTS=ON")
            cmake_setup_cmd.append(f"-DGTEST_PATH={self.gtest_path}")
            cmake_setup_cmd.append("-DML_SDK_VGF_LIB_BUILD_PYLIB=ON")
            cmake_setup_cmd.append(f"-DPYBIND11_PATH={self.pybind11_path}")

        if self.doc:
            cmake_setup_cmd.append("-DSCENARIO_RUNNER_BUILD_DOCS=ON")

        if self.run_linting:
            cmake_setup_cmd.append("-DSCENARIO_RUNNER_ENABLE_LINT=ON")

        if self.enable_gcc_sanitizers:
            cmake_setup_cmd.append("-DSCENARIO_RUNNER_GCC_SANITIZERS=ON")

        if self.enable_rdoc:
            cmake_setup_cmd.append("-DSCENARIO_RUNNER_ENABLE_RDOC=ON")

        if not self.setup_platform_build(cmake_setup_cmd):
            return 1

        cmake_build_cmd = [
            "cmake",
            "--build",
            self.build_dir,
            "-j",
            str(self.threads),
            "--config",
            self.build_type,
        ]

        try:
            subprocess.run(cmake_setup_cmd, check=True)
            subprocess.run(cmake_build_cmd, check=True)

            if self.run_tests:
                test_cmd = [
                    "ctest",
                    "--test-dir",
                    str(self.test_dir),
                    "-j",
                    str(self.threads),
                    "--output-on-failure",
                ]
                subprocess.run(test_cmd, check=True)

                exe_ext, build_type_dir = (
                    (".exe", self.build_type)
                    if platform.system() == "Windows"
                    else ("", "")
                )

                cmake_build_vgf_pylib = [
                    "cmake",
                    "--build",
                    f"{self.build_dir}/vgf-lib/src",
                    "--target",
                    "vgfpy",
                    "--config",
                    self.build_type,
                ]
                subprocess.run(cmake_build_vgf_pylib, check=True)

                pytest_cmd = [
                    "python",
                    "-m",
                    "pytest",
                    "-n",
                    str(self.threads),
                    "--tb=short",
                    f"{SCENARIO_RUNNER_DIR / 'src' / 'tests'}",
                    "--vgf-pylib-dir",
                    f"{self.build_dir}/vgf-lib/src/{build_type_dir}",
                ]

                if self.install:
                    pytest_cmd += [
                        "--scenario-runner",
                        f"{self.install}/bin/scenario-runner{exe_ext}",
                        "--glsl-compiler",
                        f"{self.install}/bin/glslc{exe_ext}",
                        "--dds-utils",
                        f"{self.install}/bin/dds_utils{exe_ext}",
                        "--spirv-as",
                        f"{self.install}/bin/spirv-as{exe_ext}",
                        "--spirv-val",
                        f"{self.install}/bin/spirv-val{exe_ext}",
                    ]
                else:
                    pytest_cmd += [
                        "--scenario-runner",
                        f"{self.build_dir}/{build_type_dir}/scenario-runner{exe_ext}",
                        "--glsl-compiler",
                        f"{self.build_dir}/src/tools/{build_type_dir}/glslc{exe_ext}",
                        "--dds-utils",
                        f"{self.build_dir}/src/tools/{build_type_dir}/dds_utils{exe_ext}",
                        "--spirv-as",
                        f"{self.build_dir}/spirv-tools/tools/{build_type_dir}/spirv-as{exe_ext}",
                        "--spirv-val",
                        f"{self.build_dir}/spirv-tools/tools/{build_type_dir}/spirv-val{exe_ext}",
                    ]

                if self.emulation_layer:
                    pytest_cmd.append("--emulation-layer")
                subprocess.run(pytest_cmd, cwd=SCENARIO_RUNNER_DIR, check=True)

            if self.install:
                cmake_install_cmd = [
                    "cmake",
                    "--install",
                    self.build_dir,
                    "--prefix",
                    self.install,
                    "--config",
                    self.build_type,
                ]
                subprocess.run(cmake_install_cmd, check=True)

            if self.package:
                package_type = self.package_type or "tgz"
                cpack_generator = package_type.upper()

                cmake_package_cmd = [
                    "cpack",
                    "--config",
                    f"{self.build_dir}/CPackConfig.cmake",
                    "-C",
                    self.build_type,
                    "-G",
                    cpack_generator,
                    "-B",
                    self.package,
                    "-D",
                    "CPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF",
                ]
                subprocess.run(cmake_package_cmd, check=True)

            if self.package_source:
                package_type = self.package_type or "tgz"
                cpack_generator = package_type.upper()

                cmake_package_cmd = [
                    "cpack",
                    "--config",
                    f"{self.build_dir}/CPackSourceConfig.cmake",
                    "-C",
                    self.build_type,
                    "-G",
                    cpack_generator,
                    "-B",
                    self.package_source,
                    "-D",
                    "CPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF",
                ]
                subprocess.run(cmake_package_cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERROR: Build failed with error: {e}", file=sys.stderr)
            return 1

        return 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        help="Name of folder where to build the Scenario Runner. Default: %(default)s",
        default=f"{SCENARIO_RUNNER_DIR / 'build'}",
    )
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        help="Number of threads to use for building. Default: %(default)s",
        default=16,
    )
    parser.add_argument(
        "--prefix-path",
        help="Path to prefix directory.",
    )
    parser.add_argument(
        "-t",
        "--test",
        help="Run unit tests after build. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--build-type",
        help="Type of build to perform. Default: %(default)s",
        default="Release",
    )
    parser.add_argument(
        "--target-platform",
        help="Specify the target build platform. Default: %(default)s",
        choices=["host", "android", "aarch64"],
        default="host",
    )
    parser.add_argument(
        "--cmake-toolchain-for-android",
        help="Path to the cmake compiler toolchain. Default: %(default)s",
        default="",
    )
    parser.add_argument(
        "--force-no-debug-symbols-android-build",
        help=(
            "Force no debug symbols when building with Android toolchain. "
            "Default on Android is RelWithDebInfo."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--vulkan-headers-path",
        help="Path to the vulkan headers folder. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'Vulkan-Headers'}",
    )
    parser.add_argument(
        "--flatc-path",
        help="Path to the flatc compiler. Default: %(default)s",
        default="",
    )
    parser.add_argument(
        "--vgf-lib-path",
        help="Path to the ml-sdk-vgf-lib repo. Default: %(default)s",
        default=f"{SCENARIO_RUNNER_DIR / '..' / 'vgf-lib'}",
    )
    parser.add_argument(
        "--glslang-path",
        help="Path to the glslang repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'glslang'}",
    )
    parser.add_argument(
        "--spirv-headers-path",
        help="Path to the spirv-headers repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'SPIRV-Headers'}",
    )
    parser.add_argument(
        "--spirv-tools-path",
        help="Path to the spirv-tools repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'SPIRV-Tools'}",
    )
    parser.add_argument(
        "--argparse-path",
        help="Path to argparse repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'argparse'}",
    )
    parser.add_argument(
        "--json-path",
        help="Path to json repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'json'}",
    )
    parser.add_argument(
        "--gtest-path",
        help="Path to googletest repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'googletest'}",
    )
    parser.add_argument(
        "--flatbuffers-path",
        help="Path to flatbuffers repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'flatbuffers'}",
    )
    parser.add_argument(
        "--pybind11-path",
        help="Path to pybind11 repo. Default: %(default)s",
        default=f"{DEPENDENCY_DIR / 'pybind11'}",
    )
    parser.add_argument(
        "--doc",
        help="Build documentation. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--enable-gcc-sanitizers",
        help="Enable GCC sanitizers. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--lint",
        help="Run linter. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--install",
        help="Install build artifacts into a provided location",
    )
    parser.add_argument(
        "--package",
        help="Create a package with build artifacts and store it in a provided location",
    )
    parser.add_argument(
        "--package-type",
        choices=["zip", "tgz"],
        help="Package type",
    )
    parser.add_argument(
        "--package-source",
        help="Create a source code package and store it in a provided location",
    )
    parser.add_argument(
        "--emulation-layer",
        help="Specifies if emulation layer is used. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--enable-rdoc",
        help=("Enable Rdoc support"),
        action="store_true",
        default=False,
    )

    if argcomplete:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()
    return args


def main():
    builder = Builder(parse_arguments())
    sys.exit(builder.run())


if __name__ == "__main__":
    main()
