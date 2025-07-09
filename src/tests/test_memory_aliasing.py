#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""  Tests for memory aliasing. """
import os

import numpy as np
import pytest


pytestmark = pytest.mark.memory_aliasing


def equal_cmp_as_fp16(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """Compare two ndarray as they have type fp16."""
    arr1_flatten = arr1.view(np.uint16).flatten()
    arr2_flatten = arr2.view(np.uint16).flatten()

    if arr1_flatten.nbytes != arr2_flatten.nbytes:
        return False

    FP16_EXPONENT = 0x7C00
    isNan = lambda value: (value & FP16_EXPONENT) == FP16_EXPONENT

    for a, b in zip(arr1_flatten, arr2_flatten):
        if isNan(a) and isNan(b):
            continue

        if a != b:
            return False

    return True


@pytest.mark.parametrize(
    "input_type, output_type, input_width, input_height, input_depth, data_type",
    [
        (
            "buffer",
            "buffer",
            256,
            1,
            1,
            np.int8,
        ),
        (
            "buffer",
            "tensor",
            8,
            8,
            2,
            np.int16,
        ),
        (
            "buffer",
            "image",
            16384,
            1,
            1,
            np.int8,
        ),
        (
            "tensor",
            "buffer",
            8,
            8,
            2,
            np.int16,
        ),
        (
            "tensor",
            "tensor",
            8,
            8,
            2,
            np.int16,
        ),
        (
            "tensor",
            "image",
            64,
            64,
            2,
            np.int16,
        ),
        (
            "image",
            "buffer",
            64,
            64,
            2,
            np.int16,
        ),
        (
            "image",
            "tensor",
            64,
            64,
            2,
            np.int16,
        ),
        (
            "image",
            "image",
            64,
            64,
            2,
            np.int16,
        ),
    ],
)
def test_generic_resources_no_compute(
    sdk_tools,
    numpy_helper,
    input_type,
    output_type,
    input_width,
    input_height,
    input_depth,
    data_type,
):
    scenario_name = f"test_memory_aliasing/input_{input_type}_output_{output_type}.json"

    if input_type == "buffer":
        data_size = (
            input_width * input_height * input_depth * np.dtype(data_type).itemsize
        )
        input_npy = numpy_helper.generate(
            [data_size], dtype=np.int8, filename="inBuffer.npy"
        )
    if input_type == "tensor":
        input_npy = numpy_helper.generate(
            [input_width, input_height, input_depth],
            dtype=data_type,
            filename="inTensor.npy",
        )
    if input_type == "image":
        dds_file = sdk_tools.generate_dds_file(
            input_height,
            input_width,
            "fp16",
            4,
            "DXGI_FORMAT_R16G16_FLOAT",
            "inImage.dds",
        )
        sdk_tools.convert_dds_to_npy(dds_file, "temp.npy", 4)
        input_npy = numpy_helper.load("temp.npy", np.uint16)

    sdk_tools.run_scenario(scenario_name)

    if output_type == "buffer":
        result = numpy_helper.load("outBuffer.npy", np.int8)
    if output_type == "tensor":
        result = numpy_helper.load("outTensor.npy", data_type)
    if output_type == "image":
        sdk_tools.convert_dds_to_npy(
            sdk_tools.resources_helper.get_testenv_path("outImage.dds"), "temp.npy", 4
        )
        result = numpy_helper.load("temp.npy", data_type)

    assert input_npy.nbytes == result.nbytes
    assert input_npy.tobytes() == result.tobytes()


@pytest.mark.parametrize(
    "input_type, output0_type, output1_type, input_width, input_height, input_depth, data_type",
    [
        (
            "buffer",
            "buffer",
            "buffer",
            256,
            1,
            1,
            np.int8,
        ),
        (
            "tensor",
            "buffer",
            "image",
            64,
            64,
            2,
            np.int16,
        ),
        (
            "image",
            "tensor",
            "tensor",
            64,
            64,
            2,
            np.int16,
        ),
    ],
)
def test_generic_resources_no_compute_two_outputs(
    sdk_tools,
    numpy_helper,
    input_type,
    output0_type,
    output1_type,
    input_width,
    input_height,
    input_depth,
    data_type,
):
    scenario_name = f"test_memory_aliasing/input_{input_type}_output_{output0_type}_{output1_type}.json"

    if input_type == "buffer":
        data_size = (
            input_width * input_height * input_depth * np.dtype(data_type).itemsize
        )
        input_npy = numpy_helper.generate(
            [data_size], dtype=np.int8, filename="inBuffer.npy"
        )
    if input_type == "tensor":
        input_npy = numpy_helper.generate(
            [input_width, input_height, input_depth],
            dtype=data_type,
            filename="inTensor.npy",
        )
    if input_type == "image":
        dds_file = sdk_tools.generate_dds_file(
            input_height,
            input_width,
            "fp16",
            4,
            "DXGI_FORMAT_R16G16_FLOAT",
            "inImage.dds",
        )
        sdk_tools.convert_dds_to_npy(dds_file, "temp.npy", 4)
        input_npy = numpy_helper.load("temp.npy", np.uint16)

    sdk_tools.run_scenario(scenario_name)

    if output0_type == "buffer":
        result0 = numpy_helper.load("outBuffer0.npy", np.int8)
    if output0_type == "tensor":
        result0 = numpy_helper.load("outTensor0.npy", data_type)
    if output0_type == "image":
        sdk_tools.convert_dds_to_npy(
            sdk_tools.resources_helper.get_testenv_path("outImage0.dds"), "temp0.npy", 4
        )
        result0 = numpy_helper.load("temp0.npy", data_type)

    if output1_type == "buffer":
        result1 = numpy_helper.load("outBuffer1.npy", np.int8)
    if output1_type == "tensor":
        result1 = numpy_helper.load("outTensor1.npy", data_type)
    if output1_type == "image":
        sdk_tools.convert_dds_to_npy(
            sdk_tools.resources_helper.get_testenv_path("outImage1.dds"), "temp1.npy", 4
        )
        result1 = numpy_helper.load("temp1.npy", data_type)

    assert input_npy.nbytes == result0.nbytes
    assert input_npy.tobytes() == result0.tobytes()
    assert input_npy.nbytes == result1.nbytes
    assert input_npy.tobytes() == result1.tobytes()


@pytest.mark.parametrize(
    "width, height, dsize, dxgi_format, data_type, scenario",
    [
        (
            64,
            10,
            4,
            "DXGI_FORMAT_R16G16_FLOAT",
            "fp16",
            "image_to_tensor_aliasing_no_compute.json",
        ),
        (
            71,
            3,
            2,
            "DXGI_FORMAT_R8G8_SINT",
            "uint8",
            "image_to_tensor_aliasing_no_compute_8bit.json",
        ),
        (
            17,
            31,
            4,
            "DXGI_FORMAT_R32_FLOAT",
            "fp32",
            "image_to_tensor_aliasing_no_compute_32bit.json",
        ),
    ],
)
def test_image_to_tensor_aliasing_no_compute(
    sdk_tools,
    numpy_helper,
    width,
    height,
    dsize,
    dxgi_format,
    data_type,
    scenario,
):
    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        data_type,
        dsize,
        dxgi_format,
        "temp.dds",
    )

    sdk_tools.run_scenario(f"test_memory_aliasing/{scenario}")

    sdk_tools.convert_dds_to_npy(dds_file, "temp.dds.npy", 2)
    dds_data = numpy_helper.load("temp.dds.npy", np.uint16)

    result = numpy_helper.load("output.npy")
    assert result.tobytes() == dds_data.tobytes()


def test_image_to_tensor_aliasing_tensor_plus_ten_shader(sdk_tools, numpy_helper):
    width, height, dsize = 64, 10, 4

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "fp16",
        dsize,
        "DXGI_FORMAT_R16G16_FLOAT",
        "temp.dds",
    )

    sdk_tools.compile_shader("test_memory_aliasing/plus_ten_tensor.comp")
    sdk_tools.run_scenario(
        "test_memory_aliasing/image_to_tensor_aliasing_aliased_tensor_plus_ten_shader.json"
    )

    sdk_tools.convert_dds_to_npy(dds_file, "temp.dds.npy", 2)
    dds_data = numpy_helper.load("temp.dds.npy", np.uint16)

    input_npy = numpy_helper.load("input.npy")
    assert np.array_equal(input_npy, dds_data.reshape(input_npy.shape))

    result = numpy_helper.load("output.npy")
    assert np.array_equal(result, dds_data.reshape(result.shape) + 10)


def test_image_to_tensor_aliasing_copy_image_shader_copy_tensor_shader(
    sdk_tools, numpy_helper, resources_helper
):
    width, height, dsize = 64, 64, 8

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "fp16",
        dsize,
        "DXGI_FORMAT_R16G16B16A16_FLOAT",
        "input_image.dds",
    )

    sdk_tools.compile_shader("test_memory_aliasing/copy_img_shader.comp")
    sdk_tools.compile_shader("test_memory_aliasing/copy_tensor_shader.comp")

    sdk_tools.run_scenario(
        "test_memory_aliasing/image_to_tensor_aliasing_copy_image_shader_copy_tensor_shader.json"
    )

    dds_file_npy_path = sdk_tools.convert_dds_to_npy(dds_file, "input_image.dds.npy", 1)
    dds_file_npy = numpy_helper.load(dds_file_npy_path)

    output_dds_npy_path = sdk_tools.convert_dds_to_npy(
        resources_helper.get_testenv_path("output_image.dds"), "output_image.dds.npy", 1
    )
    output_dds_npy = numpy_helper.load(output_dds_npy_path)
    assert dds_file_npy.nbytes == output_dds_npy.nbytes
    assert equal_cmp_as_fp16(dds_file_npy, output_dds_npy)

    input_npy = numpy_helper.load("input_tensor.npy")
    assert equal_cmp_as_fp16(dds_file_npy, input_npy)

    output_npy = numpy_helper.load("output_tensor.npy")
    assert equal_cmp_as_fp16(dds_file_npy, output_npy)


def test_image_to_tensor_aliasing_copy_tensor_shader_copy_image_shader(
    sdk_tools, numpy_helper, resources_helper
):
    width, height, dsize = 64, 64, 8

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "fp16",
        dsize,
        "DXGI_FORMAT_R16G16B16A16_FLOAT",
        "input_image.dds",
    )

    dds_file_npy_path = sdk_tools.convert_dds_to_npy(dds_file, "input_image.dds.npy", 2)
    dds_file_npy = numpy_helper.load(dds_file_npy_path, np.uint16)
    numpy_helper.save(dds_file_npy, "input_tensor.npy")

    sdk_tools.compile_shader("test_memory_aliasing/copy_tensor_shader.comp")
    sdk_tools.compile_shader("test_memory_aliasing/copy_img_shader.comp")

    sdk_tools.run_scenario(
        "test_memory_aliasing/image_to_tensor_aliasing_copy_tensor_shader_copy_image_shader.json"
    )

    output_npy = numpy_helper.load("output_tensor.npy")
    assert output_npy.tobytes() == dds_file_npy.tobytes()
    assert equal_cmp_as_fp16(output_npy, dds_file_npy)

    output_image_dds = resources_helper.get_testenv_path("output_image.dds")
    sdk_tools.convert_dds_to_npy(output_image_dds, "output_image.dds.npy", 2)
    output_image_dds_npy = numpy_helper.load("output_image.dds.npy", np.uint16)
    assert equal_cmp_as_fp16(output_npy, output_image_dds_npy)
