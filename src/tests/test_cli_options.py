#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import subprocess

import pytest


def test_graph_profiling_dump_dir_requires_existing_directory(
    scenario_runner, resources_helper
):
    scenario = resources_helper.get_scenario_path("test_shader/add_shader.json")
    invalid_dump_dir = resources_helper.get_testenv_path("missing_graph_profiling_dir")

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        scenario_runner.run(
            "--scenario",
            scenario,
            "--emulation-layer-profiling-dump-dir",
            invalid_dump_dir,
        )

    assert "Invalid graph profiling dump directory:" in (exc_info.value.stderr or "")
