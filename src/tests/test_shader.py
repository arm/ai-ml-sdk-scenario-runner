#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024,2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for shaders execution. """
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.shaders


@pytest.mark.parametrize(
    "numpy_type, shader_type",
    [
        (np.float32, "float"),
        (np.int8, "int8_t"),
        (np.int32, "uint"),
    ],
)
def test_single_shader_execution(
    sdk_tools,
    numpy_helper,
    numpy_type,
    shader_type,
):
    input1 = numpy_helper.generate([10], dtype=numpy_type, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=numpy_type, filename="inBufferB.npy")

    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": shader_type})
    sdk_tools.run_scenario(
        "test_shader/add_shader.json", {"{DATA_SIZE}": str(input1.nbytes)}
    )

    result = numpy_helper.load("outBufferAdd.npy", numpy_type)
    assert np.array_equal(result, input1 + input2)


def test_chained_shaders_execution(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})
    sdk_tools.run_scenario("test_shader/chained_shaders.json")

    result = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result, input1 + input2 + input2)


def test_shader_push_constants_execution(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferAA.npy")
    push_constants = numpy_helper.generate(10, dtype=np.float32, filename="data.npy")

    sdk_tools.compile_shader("test_shader/add_shader_with_push_constants.comp")
    sdk_tools.run_scenario("test_shader/shader_push_constants.json")

    result = numpy_helper.load("outBufferAddPush.npy", np.float32)
    assert np.array_equal(result, input1 + push_constants)


def test_shader_unstructured_push_constants_execution(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferAA.npy")
    data = [1.0, 2.0, 3.0, 4.0, 0.1, 2.0, 4.0]
    push_constants = numpy_helper.generate(
        7, dtype=np.float32, filename="data.npy", data=data
    )

    expected = input1 + sum(push_constants[:4])
    expected *= push_constants[4]
    expected *= push_constants[5]
    expected *= push_constants[6]

    sdk_tools.compile_shader("test_shader/add_shader_unstructured_push_constants.comp")
    sdk_tools.run_scenario("test_shader/shader_unstructured_push_constants.json")

    result = numpy_helper.load("outBufferUnstructuredPush.npy", np.float32)
    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_shader_build_options_execution(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBuffer.npy")

    sdk_tools.compile_shader(
        "test_shader/shader_build_opts.comp",
        compile_opts="-DCONSTANT_0=10.0 -DDIVIDE_BY_TWO",
    )

    sdk_tools.run_scenario("test_shader/shader_build_opts.json")
    result = numpy_helper.load("outBuffer.npy", np.float32)

    assert np.array_equal(result, (input1 + 10.0) / 2)


def test_shader_with_tensor_execution(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate(
        [1, 10, 1, 1], dtype=np.int8, filename="inTensor.npy"
    )

    sdk_tools.compile_shader("test_shader/tensor_shader.comp")
    sdk_tools.run_scenario("test_shader/tensor_shader.json")

    result = numpy_helper.load("outTensor.npy")
    assert np.array_equal(result, input1 + 10)


def test_glsl_preprocessor_options(sdk_tools):
    with pytest.raises(Exception):
        sdk_tools.compile_shader("test_shader/tensor_shader_glsl_options.comp")

    sdk_tools.compile_shader(
        "test_shader/tensor_shader_glsl_options.comp", compile_opts="-DTEST_OPTION"
    )
