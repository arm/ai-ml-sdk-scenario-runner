#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for pipeline caching. """
import numpy as np
import pytest

pytestmark = pytest.mark.pipeline_cache


def test_enable_pipeline_cache(sdk_tools, resources_helper, numpy_helper, capfd):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("add_shader.comp", {"TestType": "float"})

    cache_path = resources_helper.get_testenv_path("pipeline_cache")
    cache_path.mkdir()

    sdk_tools.run_scenario(
        "test_pipeline_cache/enable_pipeline_cache.json",
        options=[
            "--pipeline-caching",
            "--cache-path",
            cache_path,
        ],
    )

    captured = capfd.readouterr()
    assert "[Scenario-Runner] INFO: Pipeline Cache cleared" not in captured.out
    assert "[Scenario-Runner] INFO: Pipeline Cache loaded" not in captured.out

    result_first = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result_first, input1 + input2 + input2)

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"

    cache_data_first = files[0].read_bytes()

    # run the second time and see if cache is loaded
    sdk_tools.run_scenario(
        "test_pipeline_cache/enable_pipeline_cache.json",
        options=[
            "--pipeline-caching",
            "--cache-path",
            cache_path,
        ],
    )

    captured = capfd.readouterr()
    assert "[Scenario-Runner] INFO: Pipeline Cache cleared" not in captured.out
    assert "[Scenario-Runner] INFO: Pipeline Cache loaded" in captured.out

    result_second = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result_second, result_first)

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"

    cache_data_second = files[0].read_bytes()
    assert cache_data_second == cache_data_first

    # run the third time and see that cache is cleared
    sdk_tools.run_scenario(
        "test_pipeline_cache/enable_pipeline_cache.json",
        options=[
            "--pipeline-caching",
            "--clear-pipeline-cache",
            "--cache-path",
            cache_path,
        ],
    )

    captured = capfd.readouterr()
    assert "[Scenario-Runner] INFO: Pipeline Cache cleared" in captured.out
    assert "[Scenario-Runner] INFO: Pipeline Cache loaded" not in captured.out

    result_third = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result_third, result_first)

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"

    cache_data_third = files[0].read_bytes()
    assert cache_data_third == cache_data_first
