#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

"""Tests specialization constants support on compute shaders."""

pytestmark = pytest.mark.spec_const


@pytest.mark.parametrize(
    "expected_result, dtype, shader, scenario",
    [
        (
            42.0,
            np.float32,
            "float_shader.comp",
            "float_passthrough_glsl.json",
        ),
        (
            42,
            np.uint32,
            "uint_shader.comp",
            "uint_passthrough_glsl.json",
        ),
        (
            -42,
            np.int32,
            "int_shader.comp",
            "int_passthrough_glsl.json",
        ),
        (
            100500,
            np.uint32,
            "uint_shader.comp",
            "default_passthrough_glsl.json",
        ),
        (
            42.0,
            np.float32,
            "float_shader.hlsl",
            "float_passthrough_hlsl.json",
        ),
        (
            42,
            np.uint32,
            "uint_shader.hlsl",
            "uint_passthrough_hlsl.json",
        ),
        (
            -42,
            np.int32,
            "int_shader.hlsl",
            "int_passthrough_hlsl.json",
        ),
        (
            100500,
            np.uint32,
            "uint_shader.hlsl",
            "default_passthrough_hlsl.json",
        ),
    ],
)
def test_compute_shader_glsl_passthrough_spec_const_value(
    sdk_tools, numpy_helper, expected_result, dtype, shader, scenario
):
    if shader.endswith(".hlsl") and sdk_tools.hlsl_compiler.path is None:
        pytest.skip("HLSL compiler not provided; skipping HLSL-dependent tests.")
    sdk_tools.compile_shader(f"test_spec_const/{shader}")
    sdk_tools.run_scenario(f"test_spec_const/{scenario}")
    result = numpy_helper.load("out_data.npy", dtype)

    assert result == expected_result


@pytest.mark.parametrize(
    "expected_result, dtype, shader, scenario",
    [
        (
            42.0,
            np.float32,
            "float_shader.comp",
            "float_passthrough_spirv.json",
        ),
        (
            42,
            np.uint32,
            "uint_shader.comp",
            "uint_passthrough_spirv.json",
        ),
        (
            -42,
            np.int32,
            "int_shader.comp",
            "int_passthrough_spirv.json",
        ),
        (
            100500,
            np.uint32,
            "uint_shader.comp",
            "default_passthrough_spirv.json",
        ),
        (
            42.0,
            np.float32,
            "float_shader.hlsl",
            "float_passthrough_spirv.json",
        ),
        (
            42,
            np.uint32,
            "uint_shader.hlsl",
            "uint_passthrough_spirv.json",
        ),
        (
            -42,
            np.int32,
            "int_shader.hlsl",
            "int_passthrough_spirv.json",
        ),
        (
            100500,
            np.uint32,
            "uint_shader.hlsl",
            "default_passthrough_spirv.json",
        ),
    ],
)
def test_compute_shader_spirv_passthrough_spec_const_value(
    sdk_tools, numpy_helper, expected_result, dtype, shader, scenario
):
    if shader.endswith(".hlsl") and sdk_tools.hlsl_compiler.path is None:
        pytest.skip("HLSL compiler not provided; skipping HLSL-dependent tests.")
    sdk_tools.compile_shader(f"test_spec_const/{shader}", output="shader.spv")
    sdk_tools.run_scenario(f"test_spec_const/{scenario}")
    result = numpy_helper.load("out_data.npy", dtype)

    assert result == expected_result
