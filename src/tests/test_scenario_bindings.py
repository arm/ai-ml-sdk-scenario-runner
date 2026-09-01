#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

sr = pytest.importorskip("scenario_runner_py")


def test_python_interfaces_and_builder_types():
    builder = sr.ScenarioBuilder()
    options = sr.ScenarioOptions()

    buffer_info = sr.BufferInfo()
    buffer_info.debug_name = "buffer"
    buffer_info.size = 4
    buffer_id = builder.add_buffer(buffer_info)

    assert isinstance(builder, sr.IScenarioBuilder)
    assert isinstance(buffer_id, sr.BufferId)
    assert buffer_id.value == 0

    scenario = builder.build(options=options)
    assert isinstance(scenario, sr.IScenario)
