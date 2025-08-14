#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests Images and Tensors with tiling support.  """
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.tiling


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
    "width, height, dsize, data_type, dxgi_format, shader, scenario",
    [
        (
            64,
            64,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "image_shader.comp",
            "image_linear_to_linear.json",
        ),
        (
            256,
            100,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "image_shader.comp",
            "image_linear_to_optimal.json",
        ),
        (
            1,
            16,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "image_shader.comp",
            "image_optimal_to_linear.json",
        ),
        (
            16,
            1,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "image_shader.comp",
            "image_optimal_to_optimal.json",
        ),
    ],
)
def test_image_tiling_passthrough(
    sdk_tools,
    resources_helper,
    width,
    height,
    dsize,
    data_type,
    dxgi_format,
    shader,
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

    sdk_tools.compile_shader(f"test_tiling/{shader}", output="outputInput.spv")
    sdk_tools.run_scenario(f"test_tiling/{scenario}")

    result_dds_path = resources_helper.get_testenv_path("output.dds")
    assert sdk_tools.compare_dds(dds_file, result_dds_path, data_type)


@pytest.mark.parametrize(
    "scenario",
    [
        "tensor_linear_to_linear.json",
        "tensor_linear_to_optimal.json",
        "tensor_optimal_to_linear.json",
        "tensor_optimal_to_optimal.json",
    ],
)
def test_tensor_tiling_passthrough(sdk_tools, numpy_helper, scenario):
    input = numpy_helper.generate([1, 10, 1, 1], dtype=np.int8, filename="inTensor.npy")

    sdk_tools.compile_shader("test_tiling/tensor_shader.comp")
    sdk_tools.run_scenario(f"test_tiling/{scenario}")

    result = numpy_helper.load("outTensor.npy")
    assert np.array_equal(result, input)


@pytest.mark.parametrize(
    "width, height, dsize, dxgi_format, data_type, scenario",
    [
        (
            64,
            10,
            4,
            "DXGI_FORMAT_R16G16_FLOAT",
            "fp16",
            "aliasing_linear_tiling_no_compute.json",
        ),
        (
            17,
            31,
            4,
            "DXGI_FORMAT_R32_FLOAT",
            "fp32",
            "aliasing_linear_tiling_no_compute_32bit.json",
        ),
    ],
)
def test_aliasing_linear_tiling_no_compute(
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

    sdk_tools.run_scenario(f"test_tiling/{scenario}")

    sdk_tools.convert_dds_to_npy(dds_file, "temp.dds.npy", 2)
    dds_data = numpy_helper.load("temp.dds.npy", np.uint16)

    result = numpy_helper.load("output.npy")
    assert result.tobytes() == dds_data.tobytes()


@pytest.mark.parametrize(
    "width, height, dsize, dxgi_format, data_type, scenario",
    [
        (
            64,
            10,
            4,
            "DXGI_FORMAT_R16G16_FLOAT",
            "fp16",
            "test_aliasing_optimal_tiling_no_compute.json",
        ),
    ],
)
def test_aliasing_optimal_tiling_no_compute(
    sdk_tools, numpy_helper, width, height, dsize, dxgi_format, data_type, scenario
):
    dds_file = sdk_tools.generate_dds_file(
        height, width, data_type, dsize, dxgi_format, "temp.dds"
    )

    sdk_tools.run_scenario(f"test_tiling/{scenario}")
    sdk_tools.convert_dds_to_npy(dds_file, "temp.dds.npy", 2)

    dds_data = numpy_helper.load("temp.dds.npy", np.uint16)
    result = numpy_helper.load("output.npy", np.uint16)

    assert (
        result.tobytes() == dds_data.tobytes()
    ), "Alias + layout transition modified the data!"


def test_aliasing_linear_tiling_copy_image_shader(
    sdk_tools, numpy_helper, resources_helper
):
    """Test flow:
    - input_image (DDS) (OPTIMAL) -> copied to output_image (LINEAR) by image_shader
    - output_tensor (npy) aliasing output_image (LINEAR tiling)
    """
    width, height, dsize = 64, 64, 4

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "fp16",
        dsize,
        "DXGI_FORMAT_R16G16_FLOAT",
        "input_image.dds",
    )

    sdk_tools.compile_shader("test_tiling/image_shader.comp")

    sdk_tools.run_scenario("test_tiling/aliasing_linear_tiling_copy_image_shader.json")

    dds_file_npy_path = sdk_tools.convert_dds_to_npy(dds_file, "input_image.dds.npy", 2)
    dds_file_npy = numpy_helper.load(dds_file_npy_path, np.uint16)

    output_image_dds = resources_helper.get_testenv_path("output_image.dds")
    sdk_tools.convert_dds_to_npy(output_image_dds, "output_image.dds.npy", 2)
    output_image_dds_npy = numpy_helper.load("output_image.dds.npy", np.uint16)
    assert dds_file_npy.nbytes == output_image_dds_npy.nbytes
    assert equal_cmp_as_fp16(dds_file_npy, output_image_dds_npy)

    output_npy = numpy_helper.load("output_tensor.npy")
    assert equal_cmp_as_fp16(dds_file_npy, output_npy)


def test_aliasing_optimal_tiling_copy_tensor_shader(
    sdk_tools, numpy_helper, resources_helper
):
    """Test flow:
    - input_tensor aliasing input_image (DDS) (LINEAR tiling)
    - input_tensor (LINEAR) -> copied to output_tensor (OPTIMAL) by tensor_shader
    - output_tensor aliasing output_image (DDS) (OPTIMAL tiling)
    """
    width, height, dsize = 64, 64, 8

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "fp16",
        dsize,
        "DXGI_FORMAT_R16G16B16A16_FLOAT",
        "input_image.dds",
    )

    sdk_tools.compile_shader("test_tiling/tensor_shader.comp")

    sdk_tools.run_scenario(
        f"test_tiling/aliasing_optimal_tiling_copy_tensor_shader.json"
    )


def test_tensor_write_fixed(sdk_tools, numpy_helper):
    width, height = 64, 64

    # Compile the fixed tensor writer shader
    sdk_tools.compile_shader(
        "test_tiling/tensor_write_fixed.comp", output="tensor_write_fixed.spv"
    )

    # Run the scenario
    sdk_tools.run_scenario("test_tiling/tensor_write_fixed.json")

    print("Scenario run completed. Validating tensor values.")

    # Load tensor output
    output_npy = numpy_helper.load("output_tensor.npy")

    # Print top-left values
    print("Top-left pixel channel 0 (R):", output_npy[0, 0, 0, 0])
    print("Top-left pixel channel 1 (G):", output_npy[0, 1, 0, 0])

    # Assertions to validate fixed values were written and preserved
    assert np.all(output_npy[:, 0, :, :] == 12345)
    assert np.all(output_npy[:, 1, :, :] == 54321)


def test_tensor_image_tiling_mismatch_should_fail(sdk_tools, capfd):
    scenario_file = "test_tiling/invalid_tiling_aliasing.json"

    try:
        sdk_tools.run_scenario(scenario_file)
        assert False, "Expected CalledProcessError due to tiling mismatch"
    except subprocess.CalledProcessError:
        # Capture both stdout and stderr that pytest sees
        out, err = capfd.readouterr()
        combined = out + err

        print("Captured combined output:\n", combined)
        assert "Aliased resources must have identical tiling" in combined


def test_image_format_and_mipmap_tiling_support(sdk_tools, capfd):
    scenario_file = "test_tiling/invalid_optimal_tiling_mip_level.json"

    try:
        sdk_tools.run_scenario(scenario_file)
        assert False, "Expected CalledProcessError due to high mip level"
    except subprocess.CalledProcessError:
        # Capture both stdout and stderr that pytest sees
        out, err = capfd.readouterr()
        combined = out + err

        print("Captured combined output:\n", combined)
        assert "mip level provided is not supported" in combined
