#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import json
import os
import subprocess
import sys

import numpy as np
import pytest

"""Tests for pipeline caching."""


pytestmark = pytest.mark.pipeline_cache
PIPELINE_CACHE_HEADER_SIZE = 32


def _seed_real_pipeline_cache_file(
    sdk_tools, resources_helper, numpy_helper, cache_path, target_scenario
):
    warmup_scenario = "test_shader/shader_build_opts.json"

    numpy_helper.generate([10], dtype=np.float32, filename="inBuffer.npy")
    sdk_tools.compile_shader(
        "test_shader/shader_build_opts.comp",
        compile_opts="-DCONSTANT_0=10.0 -DDIVIDE_BY_TWO",
    )

    warmup_scenario_path = resources_helper.prepare_scenario(warmup_scenario)
    target_scenario_path = resources_helper.get_testenv_path(
        os.path.basename(target_scenario)
    )
    target_scenario_path.write_text(warmup_scenario_path.read_text())

    sdk_tools.scenario_runner.run(
        "--scenario",
        target_scenario_path,
        "--pipeline-caching",
        "--cache-path",
        cache_path,
        "--dry-run",
    )

    target_cache_name = (
        f"{os.path.splitext(os.path.basename(target_scenario))[0]}.cache"
    )
    target_cache_file = cache_path / target_cache_name
    assert target_cache_file.is_file()
    assert target_cache_file.stat().st_size > 0


def _setup_compute_pipeline_cache_miss(sdk_tools, resources_helper, numpy_helper):
    numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})

    return "test_pipeline_cache/enable_pipeline_cache.json", None


def _setup_graph_pipeline_cache_miss(sdk_tools, resources_helper, numpy_helper):
    conv2d_spv_path = sdk_tools.assemble_spirv(
        "test_vgf_graph/conv2d.spvasm",
        {
            "INPUT_SET": "0",
            "INPUT_BINDING": "0",
            "OUTPUT_SET": "0",
            "OUTPUT_BINDING": "1",
        },
    )
    sdk_tools.validate_spirv(conv2d_spv_path)

    const_shape = [16, 2, 2, 16]
    numpy_helper.generate([1, 16, 16, 16], dtype=np.int8, filename="conv2dInput.npy")
    numpy_helper.save(np.full(const_shape, 1, dtype=np.int8), "graphConstant0.npy")

    return (
        "test_spv_graph/conv2d_spv.json",
        {
            "{SPV}": json.dumps(conv2d_spv_path.as_posix()),
            "{OUT}": json.dumps(
                resources_helper.get_testenv_path("conv2dOutput_spv.npy").as_posix()
            ),
            "{CONST}": json.dumps(
                resources_helper.get_testenv_path("graphConstant0.npy").as_posix()
            ),
            "{CONST_DIMS}": json.dumps(const_shape),
        },
    )


def _setup_graphics_pipeline_cache_miss(sdk_tools, resources_helper, numpy_helper):
    width = 64
    height = 64
    input_arr = np.arange(width * height * 4, dtype=np.uint8).reshape(
        (1, height, width, 4)
    )

    sdk_tools.generate_png_file(height, width, "input.png", input_arr[0].tobytes())
    sdk_tools.compile_shader("test_fragment/fullscreen_triangle.vert")
    sdk_tools.compile_shader("test_fragment/sampled_copy.frag")

    return "test_fragment/sampled_fragment.json", None


def _setup_optical_flow_pipeline_cache_miss(sdk_tools, resources_helper, numpy_helper):
    width = 64
    height = 64
    output_width = 16
    output_height = 16
    input_data = np.arange(width * height * 4, dtype=np.uint8).reshape(
        (height, width, 4)
    )

    sdk_tools.generate_png_file(height, width, "input_search.png", input_data.tobytes())
    sdk_tools.generate_png_file(
        height, width, "input_template.png", input_data[::-1].tobytes()
    )
    numpy_helper.generate(
        [1, output_height, output_width, 2],
        dtype=np.float16,
        filename="input_mv.npy",
    )

    return "test_optical_flow/optical_flow_png.json", None


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(_setup_compute_pipeline_cache_miss, id="compute"),
        pytest.param(_setup_graph_pipeline_cache_miss, id="graph_compute"),
        pytest.param(_setup_graphics_pipeline_cache_miss, id="graphics"),
        pytest.param(
            _setup_optical_flow_pipeline_cache_miss,
            marks=pytest.mark.skipif(
                sys.platform == "darwin",
                reason="Optical flow is not supported on Darwin",
            ),
            id="optical_flow",
        ),
    ],
)
def test_fail_on_pipeline_cache_miss_triggers_for_all_pipeline_types(
    sdk_tools, resources_helper, numpy_helper, capfd, setup
):
    scenario, replacements = setup(sdk_tools, resources_helper, numpy_helper)

    cache_path = resources_helper.get_testenv_path(
        f"{scenario.replace('/', '_')}_miss_cache"
    )
    cache_path.mkdir()
    # Using an empty cachefile does not trigger a cache miss, so we need to seed the cache with a valid cache file
    _seed_real_pipeline_cache_file(
        sdk_tools, resources_helper, numpy_helper, cache_path, scenario
    )
    capfd.readouterr()

    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario(
            scenario,
            replacements,
            options=[
                "--pipeline-caching",
                "--fail-on-pipeline-cache-miss",
                "--cache-path",
                cache_path,
                "--dry-run",
            ],
        )

    captured = capfd.readouterr()
    assert "INFO: Pipeline Cache loaded and validated." in captured.out
    assert "ERROR: Pipeline cache miss for pipeline:" in (captured.out + captured.err)


def test_pass_on_empty_pipeline_cache(sdk_tools, resources_helper, numpy_helper, capfd):
    scenario, replacements = _setup_compute_pipeline_cache_miss(
        sdk_tools, resources_helper, numpy_helper
    )

    cache_path = resources_helper.get_testenv_path(
        f"{scenario.replace('/', '_')}_miss_cache"
    )
    cache_path.mkdir()
    capfd.readouterr()

    sdk_tools.run_scenario(
        scenario,
        replacements,
        options=[
            "--pipeline-caching",
            "--fail-on-pipeline-cache-miss",
            "--cache-path",
            cache_path,
            "--dry-run",
        ],
    )

    captured = capfd.readouterr()
    assert "INFO: Pipeline Cache loaded and validated." not in captured.out
    assert "ERROR: Pipeline cache miss for pipeline:" not in (
        captured.out + captured.err
    )


def test_enable_pipeline_cache(sdk_tools, resources_helper, numpy_helper, capfd):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})

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

    first_store_idx = captured.out.find("[Scenario-Runner] INFO: Pipeline Cache stored")
    first_dispatch_idx = captured.out.find("[Scenario-Runner] INFO: Dispatch compute")
    assert (
        first_store_idx != -1
        and first_dispatch_idx != -1
        and first_store_idx < first_dispatch_idx
    )
    assert captured.out.count("[Scenario-Runner] INFO: Pipeline Cache stored") == 2

    result_first = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result_first, input1 + input2 + input2)

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"

    cache_data_first = files[0].read_bytes()
    assert len(cache_data_first) >= PIPELINE_CACHE_HEADER_SIZE

    # Run the second time and verify the cache file is accepted.
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
    assert captured.out.count("[Scenario-Runner] INFO: Pipeline Cache stored") == 2

    result_second = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result_second, result_first)

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"

    cache_data_second = files[0].read_bytes()
    assert len(cache_data_second) > 0

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
    assert captured.out.count("[Scenario-Runner] INFO: Pipeline Cache stored") == 2

    result_third = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result_third, result_first)

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"

    cache_data_third = files[0].read_bytes()
    assert len(cache_data_third) > 0


def test_pipeline_cache_hit(sdk_tools, resources_helper, numpy_helper, capfd):
    scenario, replacements = _setup_compute_pipeline_cache_miss(
        sdk_tools, resources_helper, numpy_helper
    )
    cache_path = resources_helper.get_testenv_path(
        f"{scenario.replace('/', '_')}_hit_cache"
    )
    cache_path.mkdir()

    options = ["--pipeline-caching", "--cache-path", cache_path, "--dry-run"]
    sdk_tools.run_scenario(scenario, replacements, options=options)

    cache_files = list(cache_path.glob("*.cache"))
    assert len(cache_files) == 1
    cache_data = cache_files[0].read_bytes()
    assert len(cache_data) >= PIPELINE_CACHE_HEADER_SIZE
    header_size = int.from_bytes(cache_data[:4], byteorder=sys.byteorder)
    assert header_size == PIPELINE_CACHE_HEADER_SIZE
    if len(cache_data) == header_size:
        pytest.skip("Vulkan driver did not serialize a pipeline cache payload")

    capfd.readouterr()
    sdk_tools.run_scenario(
        scenario,
        replacements,
        options=[*options, "--fail-on-pipeline-cache-miss"],
    )

    captured = capfd.readouterr()
    assert "INFO: Pipeline Cache loaded and validated." in captured.out
    assert "ERROR: Pipeline cache miss for pipeline:" not in (
        captured.out + captured.err
    )


def test_pipeline_cache_dry_run_precompiles(
    sdk_tools, resources_helper, numpy_helper, capfd
):
    numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})

    cache_path = resources_helper.get_testenv_path("pipeline_cache_dry_run")
    cache_path.mkdir()

    sdk_tools.run_scenario(
        "test_pipeline_cache/enable_pipeline_cache.json",
        options=[
            "--pipeline-caching",
            "--cache-path",
            cache_path,
            "--dry-run",
        ],
    )

    captured = capfd.readouterr()
    assert captured.out.count("[Scenario-Runner] INFO: Pipeline Cache stored") == 2
    assert "[Scenario-Runner] INFO: Dispatch compute" not in captured.out
    assert not (resources_helper.get_testenv_path() / "outBufferAdd2.npy").exists()

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"
    assert files[0].stat().st_size > 0


def test_incorrect_pipeline_cache(sdk_tools, resources_helper, numpy_helper, capfd):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})

    cache_path = resources_helper.get_testenv_path("incorrect_cache")
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
    with open(files[0], "wb") as fout:
        fout.write(os.urandom(1024))

    # run the second time with junk cache
    sdk_tools.run_scenario(
        "test_pipeline_cache/enable_pipeline_cache.json",
        options=[
            "--pipeline-caching",
            "--cache-path",
            cache_path,
        ],
    )

    captured = capfd.readouterr()
    assert (
        "WARNING: Pipeline validation: Incorrect pipeline cache header size"
        in captured.out
    )
    assert "WARNING: Pipeline Cache skipped: failed to validate" in captured.out
