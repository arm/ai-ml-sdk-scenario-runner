#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_native


def test_build_native_limits_build_and_install_to_wheel_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extension = tmp_path / "setuptools" / "scenario_runner_py.so"
    commands: list[list[str]] = []

    def run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        if "--build" in command:
            extension.parent.mkdir(parents=True, exist_ok=True)
            extension.touch()

    monkeypatch.setattr(build_native.subprocess, "run", run)

    build_native.build_native(extension)

    configure_command = commands[0]
    assert configure_command[configure_command.index("-B") + 1] == str(
        build_native.ROOT_DIR
        / "build"
        / "pip"
        / f"{build_native.platform.system().lower()}-{build_native.platform.machine().lower()}"
        / "release"
    )

    build_command = commands[1]
    assert build_command[build_command.index("--target") + 1 : -2] == [
        "scenario_runner_py",
        "scenario-runner",
        "glslc",
        "dds_utils",
        "png_utils",
    ]
    install_command = commands[2]
    assert install_command[install_command.index("--component") + 1] == "mlsdk_wheel"


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_build_native_enables_hlsl_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    system: str,
) -> None:
    extension = tmp_path / "setuptools" / "scenario_runner_py.so"
    commands: list[list[str]] = []

    def run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        if "--build" in command:
            extension.parent.mkdir(parents=True, exist_ok=True)
            extension.touch()

    monkeypatch.setattr(build_native.subprocess, "run", run)
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(
        build_native, "_native_generator_python", lambda _: Path(sys.executable)
    )
    monkeypatch.setenv("SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT", "ON")

    build_native.build_native(extension)

    assert "-DSCENARIO_RUNNER_ENABLE_HLSL_SUPPORT=ON" in commands[0]
    build_command = commands[1]
    targets = build_command[
        build_command.index("--target") + 1 : build_command.index("--parallel")
    ]
    assert "hlslc" in targets


def test_build_native_reuses_native_tree_across_python_abis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)

    monkeypatch.setattr(build_native.subprocess, "run", run)
    monkeypatch.setenv("SCENARIO_RUNNER_PIP_BUILD_DIR", str(tmp_path / "native"))
    monkeypatch.setenv("AUDITWHEEL_PLAT", "manylinux_2_28_x86_64")
    monkeypatch.setenv("SCENARIO_RUNNER_PIP_BUILD_TYPE", "Debug")

    for abi in ("cpython-310-x86_64-linux-gnu", "cpython-314-x86_64-linux-gnu"):
        extension = tmp_path / abi / "scenario_runner_py.so"
        extension.parent.mkdir()
        extension.touch()
        monkeypatch.setattr(build_native, "PYTHON_ABI_TAG", abi)
        monkeypatch.setattr(
            build_native.sys, "executable", str(tmp_path / abi / "python")
        )

        build_native.build_native(extension)

        configure = commands[-3]
        assert configure[configure.index("-B") + 1] == str(
            tmp_path / "native" / "manylinux_2_28_x86_64" / "debug"
        )
        assert f"-DMLSDK_WHEEL_PYTHON_ABI={abi}" in configure
        assert f"-DPython_EXECUTABLE={tmp_path / abi / 'python'}" in configure
        assert "-UPython_*" in configure
        assert "-U_Python_*" in configure
        assert "-UPYTHON_*" in configure
        assert "-DPYBIND11_FINDPYTHON=ON" in configure


def test_build_native_keeps_native_generator_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extension = tmp_path / "setuptools" / "scenario_runner_py.so"
    extension.parent.mkdir()
    extension.touch()
    commands: list[list[str]] = []

    def run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)
        if "-B" in command:
            build_dir = Path(command[command.index("-B") + 1])
            build_dir.mkdir(parents=True, exist_ok=True)
            native_python = next(
                arg.split("=", 1)[1]
                for arg in command
                if arg.startswith("-DPython3_EXECUTABLE=")
            )
            (build_dir / "CMakeCache.txt").write_text(
                f"Python3_EXECUTABLE:FILEPATH={native_python}\n", encoding="utf-8"
            )

    monkeypatch.setattr(build_native.subprocess, "run", run)
    monkeypatch.setenv("SCENARIO_RUNNER_PIP_BUILD_DIR", str(tmp_path / "native"))
    monkeypatch.setattr(build_native.platform, "system", lambda: "Linux")
    python_name = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for version in ("310", "314"):
        prefix = tmp_path / f"python{version}"
        native_python = prefix / "bin" / python_name
        native_python.parent.mkdir(parents=True)
        native_python.touch()
        isolated_python = tmp_path / f"build-env-{version}" / "python"
        isolated_python.parent.mkdir()
        isolated_python.touch()
        monkeypatch.setattr(build_native.sys, "base_prefix", str(prefix))
        monkeypatch.setattr(build_native.sys, "executable", str(isolated_python))
        monkeypatch.setattr(build_native.sys, "_base_executable", str(isolated_python))

        build_native.build_native(extension)
        configure = commands[-3]
        assert f"-DPython_EXECUTABLE={isolated_python}" in configure
        assert "-U_Python3_*" in configure
        assert (
            f"-DPython3_EXECUTABLE={tmp_path / 'python310' / 'bin' / python_name}"
            in configure
        )
        isolated_python.unlink()


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_native_generator_python_recovers_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    system: str,
) -> None:
    prefix = tmp_path / "base-python"
    python_name = (
        "python.exe"
        if system == "Windows"
        else f"bin/python{sys.version_info.major}.{sys.version_info.minor}"
    )
    native_python = prefix / python_name
    native_python.parent.mkdir(parents=True)
    native_python.touch()
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(build_native.sys, "base_prefix", str(prefix))
    (tmp_path / "CMakeCache.txt").write_text(
        f"Python3_EXECUTABLE:FILEPATH={tmp_path / 'removed-build-env' / 'python'}\n",
        encoding="utf-8",
    )

    assert build_native._native_generator_python(tmp_path) == native_python


def test_build_native_rejects_staged_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLSDK_WHEEL_BUILD_MODE", "staged")

    with pytest.raises(RuntimeError, match="require source mode"):
        build_native.build_native(tmp_path / "scenario_runner_py.so")
