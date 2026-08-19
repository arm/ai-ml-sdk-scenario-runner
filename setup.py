#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pathlib
import platform
import shutil
import sys

from setuptools import setup
from setuptools.command.build import build as setuptools_build
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


SCENARIO_RUNNER_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCENARIO_RUNNER_DIR))

from scripts.build import build as build_scenario_runner  # noqa: E402

SKIP_NATIVE_BUILD_ENV = "SCENARIO_RUNNER_SKIP_NATIVE_BUILD"


class Build(setuptools_build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(pathlib.Path("build") / "python")


class BuildPy(build_py):
    def run(self):
        super().run()

        binary_name = (
            "scenario-runner.exe"
            if platform.system() == "Windows"
            else "scenario-runner"
        )
        staged_binary = (
            SCENARIO_RUNNER_DIR
            / "pip_package"
            / "scenario_runner"
            / "binaries"
            / "bin"
            / binary_name
        )
        if os.environ.get(SKIP_NATIVE_BUILD_ENV) == "1" or staged_binary.is_file():
            return

        dependency_dir = SCENARIO_RUNNER_DIR.parent.parent / "dependencies"
        vgf_lib_dir = SCENARIO_RUNNER_DIR.parent / "vgf-lib"
        missing_paths = [
            path for path in (dependency_dir, vgf_lib_dir) if not path.is_dir()
        ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(
                "The Scenario Runner native build requires an ML SDK checkout. "
                f"Missing: {missing}"
            )

        missing_tools = [tool for tool in ("cmake", "ninja") if not shutil.which(tool)]
        if missing_tools:
            raise RuntimeError(
                "The Scenario Runner native build requires: " + ", ".join(missing_tools)
            )

        build_command = self.get_finalized_command("build")
        native_build_dir = pathlib.Path(build_command.build_temp) / "scenario_runner"
        native_install_dir = (
            pathlib.Path(self.build_lib) / "scenario_runner" / "binaries"
        )

        result = build_scenario_runner(
            [
                "--build-dir",
                str(native_build_dir),
                "--install",
                str(native_install_dir),
            ]
        )
        if result:
            raise RuntimeError(
                f"Scenario Runner native build failed with code {result}"
            )


class BDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        system = platform.system()
        machine = platform.machine()
        if system == "Windows":
            assert machine == "AMD64"
            platformName = "win_amd64"
        elif system == "Linux":
            if machine == "aarch64":
                platformName = "manylinux2014_aarch64"
            else:
                assert machine == "x86_64"
                platformName = "manylinux2014_x86_64"
        elif system == "Darwin":
            assert machine == "arm64"
            platformName = "macosx_11_0_arm64"
        return ("py3", "none", platformName)


setup(cmdclass={"build": Build, "build_py": BuildPy, "bdist_wheel": BDistWheel})
