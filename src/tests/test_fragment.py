#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import subprocess

import numpy as np
import pytest
from conftest import ResourcesHelper
from conftest import SDKTools

"""Runtime tests for dispatch_fragment command."""

pytestmark = pytest.mark.shaders


def _prepare_fragment_shaders(sdk_tools: SDKTools) -> None:
    sdk_tools.compile_shader("test_fragment/fullscreen_triangle.vert")
    sdk_tools.compile_shader("test_fragment/sampled_copy.frag")


def test_fragment_dispatch_samples_input_image(
    sdk_tools: SDKTools,
    resources_helper: ResourcesHelper,
) -> None:
    width = 64
    height = 64

    input_arr = np.arange(width * height * 4, dtype=np.uint8).reshape(
        (1, height, width, 4)
    )

    sdk_tools.generate_png_file(height, width, "input.png", data=input_arr[0].tobytes())
    _prepare_fragment_shaders(sdk_tools)

    sdk_tools.run_scenario("test_fragment/sampled_fragment.json")

    output_png = resources_helper.get_testenv_path("output.png")
    input_png = resources_helper.get_testenv_path("input.png")
    assert sdk_tools.compare_png(input_png, output_png)


def test_fragment_dispatch_rejects_wrong_shader_stage(sdk_tools: SDKTools) -> None:
    width = 64
    height = 64

    input_arr = np.arange(width * height * 4, dtype=np.uint8).reshape(
        (1, height, width, 4)
    )

    sdk_tools.generate_png_file(height, width, "input.png", data=input_arr[0].tobytes())
    _prepare_fragment_shaders(sdk_tools)

    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario(
            "test_fragment/sampled_fragment.json",
            {
                '"stage": "vertex"': '"stage": "compute"',
            },
        )


def test_fragment_dispatch_rejects_mismatched_color_attachment_extent(
    sdk_tools: SDKTools,
) -> None:
    width = 64
    height = 64

    input_arr = np.arange(width * height * 4, dtype=np.uint8).reshape(
        (1, height, width, 4)
    )

    sdk_tools.generate_png_file(height, width, "input.png", data=input_arr[0].tobytes())
    _prepare_fragment_shaders(sdk_tools)

    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario("test_fragment/sampled_fragment_mismatched_extent.json")
