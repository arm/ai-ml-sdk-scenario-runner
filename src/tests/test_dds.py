#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests DDS files processing.  """
import numpy as np
import pytest


pytestmark = pytest.mark.dds


@pytest.mark.parametrize(
    "width, height, dsize, data_type, dxgi_format, shader, scenario",
    [
        (
            256,
            100,
            8,
            "fp16",
            "DXGI_FORMAT_R16G16B16A16_FLOAT",
            "passthrough_RGBA16.comp",
            "passthrough_RGBA16.json",
        ),
        (
            64,
            64,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "passthrough_RG16.comp",
            "passthrough_R16G16_FLOAT.json",
        ),
        (
            64,
            64,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "passthrough_glsl_sampler.comp",
            "passthrough_sampler.json",
        ),
        (
            64,
            64,
            2,
            "fp16",
            "DXGI_FORMAT_R16_FLOAT",
            "passthrough_glsl_sampler_R16.comp",
            "passthrough_sampler_R16.json",
        ),
        (
            1,
            16,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "passthrough_glsl_sampler_small_width.comp",
            "passthrough_sampler_small_width.json",
        ),
        (
            16,
            1,
            4,
            "fp16",
            "DXGI_FORMAT_R16G16_FLOAT",
            "passthrough_glsl_sampler_small_height.comp",
            "passthrough_sampler_small_height.json",
        ),
        (
            256,
            100,
            8,
            "fp16",
            "DXGI_FORMAT_R16G16B16A16_FLOAT",
            "passthrough_glsl_sampler_RGBA16.comp",
            "passthrough_sampler_RGBA16.json",
        ),
    ],
)
def test_image_passthrough(
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

    sdk_tools.compile_shader(f"test_dds/{shader}", output="outputInput.spv")
    sdk_tools.run_scenario(f"test_dds/{scenario}")

    result_dds_path = resources_helper.get_testenv_path("output.dds")
    assert sdk_tools.compare_dds(dds_file, result_dds_path, data_type)


def test_image_passthrough_d32_type(sdk_tools, resources_helper, numpy_helper):
    def create_input(size):
        values = [
            0x00,  # 10
            0x3C,
            0x40,  # 100
            0x56,
            0xD0,  # 1000
            0x63,
            0xE2,  # 10,000
            0x70,
        ]
        output = values * (size // 8)
        return np.array(output).astype(np.uint8)

    def create_ref_output(size):
        values = [
            0x00,  # 10
            0x3C,
            0x40,  # 100
            0x56,
            0x00,  # 10
            0x3C,
            0x40,  # 100
            0x56,
        ]
        output = values * (size // 8)
        return np.array(output).astype(np.uint8)

    width, height, dsize = 16, 16, 8
    data = create_input(width * height * dsize).tobytes()

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        dsize,
        "DXGI_FORMAT_D32_FLOAT_S8X24_UINT",
        "depth.dds",
        data,
    )

    sdk_tools.compile_shader(
        "test_dds/passthrough_depth.comp", output="outputInput.spv"
    )
    sdk_tools.run_scenario("test_dds/passthrough_depth.json")

    ref_output = create_ref_output(width * height * dsize // 2)

    output_dds = resources_helper.get_testenv_path("output.dds")
    output_dds_npy = sdk_tools.convert_dds_to_npy(output_dds, "output.dds.npy", 1)

    assert numpy_helper.load(output_dds_npy).tobytes() == ref_output.tobytes()


def create_repeating_reference_output(size, pixels):
    n_copies = size // len(pixels)
    output = pixels * n_copies
    return np.array(output).astype(np.uint8)


@pytest.mark.parametrize(
    "width, height, dsize, data_type, dxgi_format, shader, scenario, ref_output",
    [
        (
            64,
            4,
            8,
            "fp16",
            "DXGI_FORMAT_R16G16B16A16_FLOAT",
            "access_float_border.comp",
            "color_border_float.json",
            create_repeating_reference_output(64 * 4 * 8, [0, 0, 0, 0, 0, 0, 0, 60]),
        ),
        (
            64,
            4,
            8,
            "fp16",
            "DXGI_FORMAT_R16G16B16A16_FLOAT",
            "access_float_border.comp",
            "custom_color_border_float.json",
            create_repeating_reference_output(64 * 4 * 8, [248, 91, 0, 0, 0, 0, 0, 60]),
        ),
        (
            64,
            4,
            4,
            "fp16",
            "DXGI_FORMAT_R8G8B8A8_SINT",
            "access_int_border.comp",
            "custom_color_border_int.json",
            create_repeating_reference_output(64 * 4 * 4, [20, 13, 110, 4]),
        ),
    ],
)
def test_border_access(
    sdk_tools,
    numpy_helper,
    resources_helper,
    width,
    height,
    dsize,
    data_type,
    dxgi_format,
    shader,
    scenario,
    ref_output,
):
    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        data_type,
        dsize,
        dxgi_format,
        "temp.dds",
    )

    sdk_tools.compile_shader(f"test_dds/{shader}", output="outputInput.spv")
    sdk_tools.run_scenario(f"test_dds/{scenario}")

    output_dds_file = resources_helper.get_testenv_path("output.dds")
    output_dds_npy_file = resources_helper.get_testenv_path("output.dds.npy")

    sdk_tools.convert_dds_to_npy(output_dds_file, output_dds_npy_file, 1)
    output_dds_npy = numpy_helper.load(output_dds_npy_file)

    assert output_dds_npy.tobytes() == ref_output.tobytes()
