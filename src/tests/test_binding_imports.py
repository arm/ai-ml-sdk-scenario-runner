#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "modules",
    [("vgfpy", "scenario_runner_py"), ("scenario_runner_py", "vgfpy")],
)
def test_python_bindings_can_share_an_interpreter(request, modules):
    if request.config.getoption("--sanitizers"):
        pytest.skip("incompatible with --sanitizers")
    for module in modules:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not built")

    # Use a fresh process for each order: collection may already import vgfpy.
    # Both import and interpreter shutdown must use compatible allocators for
    # the internal data that pybind11 shares between these extensions.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            "; ".join(f"import {m}" for m in modules),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
