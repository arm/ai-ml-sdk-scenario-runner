#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Tests PNG file processing."""
import subprocess

import numpy as np
import pytest


pytestmark = pytest.mark.png


def expected_identity(arr: np.ndarray) -> np.ndarray:
    return arr


def expected_swizzle(arr: np.ndarray) -> np.ndarray:
    return arr[:, :, :, [2, 1, 0, 3]]


@pytest.mark.parametrize(
    "width, height, shader, scenario, expected_raw_fn",
    [
        (
            64,
            64,
            "passthrough_png.comp",
            "passthrough_png.json",
            expected_identity,
        ),
        (
            32,
            32,
            "swizzle_png.comp",
            "swizzle_png.json",
            expected_swizzle,
        ),
    ],
)
def test_png(
    sdk_tools,
    resources_helper,
    numpy_helper,
    width,
    height,
    shader,
    scenario,
    expected_raw_fn,
):
    input_arr = np.arange(width * height * 4, dtype=np.uint8).reshape(
        (1, height, width, 4)
    )
    sdk_tools.generate_png_file(height, width, "input.png", data=input_arr[0].tobytes())

    sdk_tools.compile_shader(f"test_png/{shader}", output="shader.spv")
    sdk_tools.run_scenario(f"test_png/{scenario}")

    result_png_path = resources_helper.get_testenv_path("output.png")
    sdk_tools.convert_png_to_npy(result_png_path, "output.npy")
    output_arr = numpy_helper.load("output.npy")
    expected = expected_raw_fn(input_arr)
    assert np.array_equal(output_arr, expected)


def test_png_utils_generate_and_compare(sdk_tools, resources_helper):
    png_a = resources_helper.get_testenv_path("a.png")
    png_b = resources_helper.get_testenv_path("b.png")

    sdk_tools.generate_png_file(2, 3, png_a)
    sdk_tools.generate_png_file(2, 3, png_b)

    assert sdk_tools.compare_png(png_a, png_b)


def test_png_utils_to_npy(sdk_tools, resources_helper, numpy_helper):
    png_file = resources_helper.get_testenv_path("a.png")
    npy_file = resources_helper.get_testenv_path("a.npy")

    sdk_tools.generate_png_file(2, 2, png_file)
    sdk_tools.convert_png_to_npy(png_file, npy_file)

    arr = numpy_helper.load("a.npy")
    assert arr.shape == (1, 2, 2, 4)
    assert np.all(arr == 0)


def test_png_save_roundtrip_to_png(sdk_tools, resources_helper):
    input_png = resources_helper.get_testenv_path("input.png")
    input_arr = np.arange(64 * 64 * 4, dtype=np.uint8).reshape((1, 64, 64, 4))
    sdk_tools.generate_png_file(
        64,
        64,
        input_png,
        input_arr[0].tobytes(),
    )

    sdk_tools.compile_shader("test_png/passthrough_png.comp", output="shader.spv")
    sdk_tools.run_scenario("test_png/passthrough_png.json")

    result_png_path = resources_helper.get_testenv_path("output.png")
    assert sdk_tools.compare_png(input_png, result_png_path)


def test_png_corrupt_fails_to_npy(sdk_tools, resources_helper):
    png_file = resources_helper.get_testenv_path("a.png")
    png_file.write_text("not a png")
    npy_file = resources_helper.get_testenv_path("a.npy")

    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.convert_png_to_npy(png_file, npy_file)


def test_png_dims_mismatch_fails(sdk_tools, resources_helper):
    sdk_tools.compile_shader("test_png/passthrough_png.comp", output="shader.spv")
    sdk_tools.generate_png_file(
        16,
        16,
        "input.png",
    )

    with pytest.raises(Exception):
        sdk_tools.run_scenario("test_png/passthrough_png_dims_mismatch.json")
