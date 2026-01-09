#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024,2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for marking boundary. """
import numpy as np
import pytest

pytestmark = pytest.mark.mark_boundary


def test_mark_boundary(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("add_shader.comp", {"TestType": "float"})
    sdk_tools.run_scenario("test_mark_boundary/mark_boundary.json")

    result = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result, input1 + input2 + input2)


def test_mark_boundary_command_only(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.run_scenario("test_mark_boundary/mark_boundary_command_only.json")


def test_mark_boundary_and_shader(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("add_shader.comp", {"TestType": "float"})
    sdk_tools.run_scenario("test_mark_boundary/mark_boundary_and_shader.json")

    result = numpy_helper.load("outBufferAdd.npy", np.float32)
    assert np.array_equal(result, input1 + input2)
