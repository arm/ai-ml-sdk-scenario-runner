#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import platform
import sys

from setuptools import Extension
from setuptools import setup
from setuptools.command.build import build as setuptools_build
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "pip_package"))

from build_native import build_native  # noqa: E402


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class Build(setuptools_build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(pathlib.Path("build") / "python")


class BuildPy(build_py):
    def run(self):
        self.run_command("build_ext")
        super().run()


class BuildExt(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        output_path = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        install_dir = None
        if self.editable_mode:
            build_py = self.get_finalized_command("build_py")
            package_dir = pathlib.Path(build_py.get_package_dir("scenario_runner"))
            install_dir = package_dir / "binaries"
        build_native(
            output_path,
            install_dir=install_dir,
            package_version=self.distribution.get_version(),
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
        else:
            raise RuntimeError(f"Unsupported platform: {system} {machine}")
        pythonTag, abiTag, _ = super().get_tag()
        return (pythonTag, abiTag, platformName)


setup(
    cmdclass={
        "build": Build,
        "build_ext": BuildExt,
        "build_py": BuildPy,
        "bdist_wheel": BDistWheel,
    },
    ext_modules=[CMakeExtension("scenario_runner.scenario_runner_py")],
)
