#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.optical_flow


def _generate_hint_motion_vectors(numpy_helper, height, width):
    numpy_helper.generate(
        [1, height, width, 2], dtype=np.float16, filename="input_mv.npy"
    )


def _assert_outputs(resources_helper):
    output_flow = resources_helper.get_testenv_path("output_flow.dds")
    output_cost = resources_helper.get_testenv_path("output_cost.dds")
    assert output_flow.is_file()
    assert output_cost.is_file()
    assert output_flow.stat().st_size > 0
    assert output_cost.stat().st_size > 0


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Optical flow is not supported on Darwin"
)
def test_optical_flow_png(sdk_tools, resources_helper, numpy_helper):
    width, height = 64, 64
    out_w, out_h = 16, 16

    input_data = np.arange(width * height * 4, dtype=np.uint8).reshape(
        (height, width, 4)
    )
    sdk_tools.generate_png_file(height, width, "input_search.png", input_data.tobytes())
    sdk_tools.generate_png_file(
        height, width, "input_template.png", input_data[::-1].tobytes()
    )

    _generate_hint_motion_vectors(numpy_helper, out_h, out_w)

    sdk_tools.run_scenario("test_optical_flow/optical_flow_png.json")
    _assert_outputs(resources_helper)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Optical flow is not supported on Darwin"
)
def test_optical_flow_dds(sdk_tools, resources_helper, numpy_helper):
    width, height = 64, 64
    out_w, out_h = 16, 16

    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_search.DDS",
    )
    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_template.DDS",
    )

    _generate_hint_motion_vectors(numpy_helper, out_h, out_w)

    sdk_tools.run_scenario("test_optical_flow/optical_flow_dds.json")
    _assert_outputs(resources_helper)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Optical flow is not supported on Darwin"
)
def test_optical_flow_search_image_lod_invalid(sdk_tools, numpy_helper):
    width, height = 64, 64
    out_w, out_h = 16, 16

    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_search.DDS",
    )
    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_template.DDS",
    )

    _generate_hint_motion_vectors(numpy_helper, out_h, out_w)

    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario("test_optical_flow/optical_flow_search_lod_invalid.json")


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Optical flow is not supported on Darwin"
)
def test_optical_flow_search_image_lod_valid(sdk_tools, resources_helper, numpy_helper):
    width, height = 128, 128
    out_w, out_h = 16, 16

    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_search.DDS",
    )
    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_template.DDS",
    )

    _generate_hint_motion_vectors(numpy_helper, out_h, out_w)

    sdk_tools.run_scenario("test_optical_flow/optical_flow_search_lod_valid.json")
    _assert_outputs(resources_helper)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Optical flow is not supported on Darwin"
)
def test_optical_flow_lod_all_bindings(sdk_tools, resources_helper, numpy_helper):
    width, height = 128, 128
    out_w, out_h = 32, 32

    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_search.DDS",
    )
    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_template.DDS",
    )

    _generate_hint_motion_vectors(numpy_helper, out_h, out_w)

    sdk_tools.run_scenario("test_optical_flow/optical_flow_lod_all_bindings.json")
    _assert_outputs(resources_helper)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Optical flow is not supported on Darwin"
)
def test_optical_flow_lod_mixed_bindings(sdk_tools, resources_helper, numpy_helper):
    width, height = 128, 128
    out_w, out_h = 16, 16

    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_search.DDS",
    )
    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_template.DDS",
    )

    _generate_hint_motion_vectors(numpy_helper, out_h, out_w)

    sdk_tools.run_scenario("test_optical_flow/optical_flow_lod_mixed_bindings.json")
    _assert_outputs(resources_helper)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Optical flow is not supported on Darwin"
)
def test_optical_flow_mips_no_lod_control(sdk_tools, resources_helper, numpy_helper):
    width, height = 128, 128
    out_w, out_h = 32, 32

    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_search.DDS",
    )
    sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        4,
        "DXGI_FORMAT_R8G8B8A8_UNORM",
        "input_template.DDS",
    )

    _generate_hint_motion_vectors(numpy_helper, out_h, out_w)

    sdk_tools.run_scenario("test_optical_flow/optical_flow_mips_no_lod.json")
    _assert_outputs(resources_helper)
