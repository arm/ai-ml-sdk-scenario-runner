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
        build_native.ROOT_DIR / "build" / "pip" / build_native.PYTHON_ABI_TAG
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


def test_build_native_enables_hlsl_from_environment(
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
    monkeypatch.setattr(build_native.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SCENARIO_RUNNER_ENABLE_HLSL_SUPPORT", "ON")

    build_native.build_native(extension)

    assert "-DSCENARIO_RUNNER_ENABLE_HLSL_SUPPORT=ON" in commands[0]
    build_command = commands[1]
    targets = build_command[
        build_command.index("--target") + 1 : build_command.index("--parallel")
    ]
    assert "hlslc" in targets


def test_build_native_rejects_staged_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLSDK_WHEEL_BUILD_MODE", "staged")

    with pytest.raises(RuntimeError, match="require source mode"):
        build_native.build_native(tmp_path / "scenario_runner_py.so")
