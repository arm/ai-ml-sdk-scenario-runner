#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""  Tests for tensor aliasing. """
import numpy as np
import pytest


pytestmark = pytest.mark.tensor_aliasing


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

    sdk_tools.run_scenario(f"test_old_tensor_aliasing/{scenario}")

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

    sdk_tools.compile_shader("test_old_tensor_aliasing/plus_ten_tensor.comp")
    sdk_tools.run_scenario(
        "test_old_tensor_aliasing/image_to_tensor_aliasing_aliased_tensor_plus_ten_shader.json"
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

    sdk_tools.compile_shader("test_old_tensor_aliasing/copy_img_shader.comp")
    sdk_tools.compile_shader("test_old_tensor_aliasing/copy_tensor_shader.comp")

    sdk_tools.run_scenario(
        "test_old_tensor_aliasing/image_to_tensor_aliasing_copy_image_shader_copy_tensor_shader.json"
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

    sdk_tools.compile_shader("test_old_tensor_aliasing/copy_tensor_shader.comp")
    sdk_tools.compile_shader("test_old_tensor_aliasing/copy_img_shader.comp")

    sdk_tools.run_scenario(
        "test_old_tensor_aliasing/image_to_tensor_aliasing_copy_tensor_shader_copy_image_shader.json"
    )

    output_npy = numpy_helper.load("output_tensor.npy")
    assert output_npy.tobytes() == dds_file_npy.tobytes()
    assert equal_cmp_as_fp16(output_npy, dds_file_npy)

    output_image_dds = resources_helper.get_testenv_path("output_image.dds")
    sdk_tools.convert_dds_to_npy(output_image_dds, "output_image.dds.npy", 2)
    output_image_dds_npy = numpy_helper.load("output_image.dds.npy", np.uint16)
    assert equal_cmp_as_fp16(output_npy, output_image_dds_npy)
