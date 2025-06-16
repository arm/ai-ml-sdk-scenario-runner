#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for mipmaps writing. """
import numpy as np
import pytest

pytestmark = pytest.mark.mipmaps_writing


def test_generate_mipmaps_by_blits(sdk_tools, resources_helper, numpy_helper):
    width, height, element_size, mip_levels = 64, 32, 4, 3

    data = np.zeros([width * height * 4], np.float32)
    for x in range(width):
        for y in range(height):
            pos = (x * height + y) * 4
            data[pos] = float(x)
            data[pos + 1] = float(y)
            data[pos + 2] = float(x * y)
            data[pos + 3] = 128

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "fp32",
        element_size * 4,
        "DXGI_FORMAT_R32G32B32A32_FLOAT",
        "base_layer.dds",
        data=data.tobytes(),
    )

    sdk_tools.compile_shader("test_mipmap_writing/read_from_mipmaps.comp")
    sdk_tools.run_scenario("test_mipmap_writing/gen_mipmaps_by_blits.json")

    mip_width, mip_height = width, height
    sample_window_size = 1

    for cur_mip_level in range(mip_levels):
        dds_filename = f"lod{cur_mip_level}.dds"
        dds_file = resources_helper.get_testenv_path(dds_filename)

        dds_npy_file = sdk_tools.convert_dds_to_npy(
            dds_file, f"{dds_filename}.npy", element_size
        )
        dds_npy = numpy_helper.load(dds_npy_file, np.float32)
        assert dds_npy.shape == (1, mip_height, mip_width, 4)
        dds_npy_flatten = dds_npy.flatten()

        # Notice that this test might not work with other settings. The sampling process is highly
        # implementation-dependent.
        for x in range(mip_width):
            for y in range(mip_height):
                index = (x * mip_height + y) * 4

                r = dds_npy_flatten[index]
                g = dds_npy_flatten[index + 1]
                b = dds_npy_flatten[index + 2]
                a = dds_npy_flatten[index + 3]

                x_src_start = float(x * sample_window_size)
                x_src_end = float((x + 1) * sample_window_size)
                y_src_start = float(y * sample_window_size)
                y_src_end = float((y + 1) * sample_window_size)

                assert (x_src_start <= r) and (r <= x_src_end)
                assert (y_src_start <= g) and (g <= y_src_end)
                assert (x_src_start * y_src_start <= b) and (b <= x_src_end * y_src_end)
                assert a == 128

        mip_width //= 2
        mip_height //= 2
        sample_window_size *= 2


def test_write_to_mipmaps(sdk_tools, resources_helper, numpy_helper):
    width, height, element_size, mip_levels = 256, 256, 4, 3

    data = np.zeros([width * height * 4], np.float32)
    for x in range(width):
        for y in range(height):
            pos = (x * height + y) * 4
            data[pos] = float(x)
            data[pos + 1] = float(y)
            data[pos + 2] = float(x * y)
            data[pos + 3] = 128

    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "fp32",
        element_size * 4,
        "DXGI_FORMAT_R32G32B32A32_FLOAT",
        "base_layer.dds",
        data=data.tobytes(),
    )

    # Save colors as push constants
    for color, rgba in [
        ("red", [255.0, 0, 0, 0]),
        ("green", [0, 255.0, 0, 0]),
        ("blue", [0, 0, 255.0, 0]),
    ]:
        npy_data = numpy_helper.generate(
            (4,),
            np.float32,
            f"{color}.npy",
            rgba,
        )

    sdk_tools.compile_shader("test_mipmap_writing/write_to_mipmaps.comp")
    sdk_tools.compile_shader("test_mipmap_writing/read_from_mipmaps.comp")
    sdk_tools.run_scenario("test_mipmap_writing/write_to_mipmaps.json")

    mip_width, mip_height = width, height
    sample_window_size = 1

    for cur_mip_level in range(mip_levels):
        dds_filename = f"lod{cur_mip_level}.dds"
        dds_file = resources_helper.get_testenv_path(dds_filename)

        dds_npy_file = sdk_tools.convert_dds_to_npy(dds_file, f"{dds_filename}.npy", 4)
        dds_npy = numpy_helper.load(dds_npy_file, np.float32)
        assert dds_npy.shape == (1, mip_height, mip_width, 4)

        dds_npy = np.reshape(dds_npy, (-1, 4))

        assert np.all(dds_npy[:, cur_mip_level] == 255.0)
        assert np.all(dds_npy[:, :cur_mip_level] == 0)
        assert np.all(dds_npy[:, cur_mip_level + 1 :] == 0)

        mip_width //= 2
        mip_height //= 2
        sample_window_size *= 2
