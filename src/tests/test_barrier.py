#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for barrier functionality.  """
import numpy as np
import pytest


pytestmark = pytest.mark.barrier


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


def create_repeating_reference_output(size, pixels):
    n_copies = size // len(pixels)
    output = pixels * n_copies
    return np.array(output).astype(np.uint8)


def scenario_repacements(implicit_barrier_enabled, default_stages, barrier_json):
    replacements = {}

    if implicit_barrier_enabled:
        replacements["{IMPL_BARRIER}"] = ""
        replacements["{BARRIER_DISPATCH}"] = ""
    else:
        replacements["{IMPL_BARRIER}"] = '"implicit_barrier": false,'
        replacements["{BARRIER_DISPATCH}"] = barrier_json

    stages = """
    "src_stage": ["compute"],
    "dst_stage": ["compute"],
""".strip()

    if default_stages:
        replacements["{STAGES}"] = ""
    else:
        replacements["{STAGES}"] = stages

    return replacements


def image_scenario_repacements(implicit_barrier_enabled, default_stages):
    barrier_json = """
{
    "dispatch_barrier": {
        "image_barrier_refs": ["inDDSImageBarrier"],
        "memory_barrier_refs": [],
        "buffer_barrier_refs": []
    }
},
""".strip()

    return scenario_repacements(
        implicit_barrier_enabled,
        default_stages,
        barrier_json,
    )


def buffer_scenario_repacements(implicit_barrier_enabled, default_stages):
    barrier_json = """
{
    "dispatch_barrier": {
        "image_barrier_refs": [],
        "memory_barrier_refs": [],
        "buffer_barrier_refs": ["bufferBarrier"]
    }
},
""".strip()

    return scenario_repacements(
        implicit_barrier_enabled,
        default_stages,
        barrier_json,
    )


@pytest.mark.parametrize(
    "implicit_barrier_enabled, default_stages",
    [
        (False, False),
        (False, True),
        (True, True),
    ],
)
def test_image_memory_barrier(
    sdk_tools,
    resources_helper,
    numpy_helper,
    implicit_barrier_enabled,
    default_stages,
):
    width, height, dsize = 64, 4, 8

    data = create_input(height * width * dsize).tobytes()
    dds_file = sdk_tools.generate_dds_file(
        height,
        width,
        "uint8",
        dsize,
        "DXGI_FORMAT_R16G16B16A16_FLOAT",
        "input.dds",
        data,
    )

    sdk_tools.compile_shader("test_barrier/apply_offset.comp", output="outputInput.spv")
    sdk_tools.run_scenario(
        "test_barrier/image_barrier.json",
        image_scenario_repacements(implicit_barrier_enabled, default_stages),
    )

    output_ref = create_repeating_reference_output(
        height * width * dsize, [0, 66, 64, 86, 208, 99, 226, 112]
    )

    output_dds = resources_helper.get_testenv_path("output.dds")
    output_dds_npy = sdk_tools.convert_dds_to_npy(output_dds, "output.dds.npy", 1)

    output = numpy_helper.load(output_dds_npy)
    assert output.tobytes() == output_ref.tobytes()


@pytest.mark.parametrize(
    "implicit_barrier_enabled, default_stages",
    [
        (False, False),
        (False, True),
        (True, True),
    ],
)
def test_buffer_memory_barrier(
    sdk_tools,
    resources_helper,
    numpy_helper,
    implicit_barrier_enabled,
    default_stages,
):
    input1 = numpy_helper.generate(
        [256], dtype=np.uint8, filename="input.npy", data=[42] * 256
    )

    sdk_tools.compile_shader("test_barrier/add_one.comp", output="addOne.spv")

    sdk_tools.run_scenario(
        "test_barrier/buffer_barrier.json",
        buffer_scenario_repacements(implicit_barrier_enabled, default_stages),
    )

    output = numpy_helper.load("output.npy")
    assert np.array_equal(output, input1 + 1 + 1)
