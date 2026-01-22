#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for shaders execution. """
import io
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
import vgfpy as vgf

DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000
VK_FORMAT_R8_SINT = 14
pretendVulkanHeaderVersion = 123

pytestmark = pytest.mark.repeated_runs


def test_chained_shaders_execution_count_times(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")

    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})
    sdk_tools.run_scenario("test_shader/chained_shaders.json", options=["--repeat=3"])

    result = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result, input1 + input2 + input2)


def test_performance_counters(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")
    dump_path = os.path.join(os.getcwd(), "perfCounterTest.json")
    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})
    sdk_tools.run_scenario(
        "test_shader/chained_shaders.json",
        options=["--repeat=2", "--perf-counters-dump-path", dump_path],
    )

    if os.path.exists(dump_path):
        os.remove(dump_path)
    result = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result, input1 + input2 + input2)


def test_profiling(sdk_tools, numpy_helper):
    input1 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferA.npy")
    input2 = numpy_helper.generate([10], dtype=np.float32, filename="inBufferB.npy")
    dump_path = os.path.join(os.getcwd(), "profilingTest.json")
    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})
    sdk_tools.run_scenario(
        "test_shader/chained_shaders.json",
        options=["--repeat=3", "--profiling-dump-path", dump_path],
    )

    if os.path.exists(dump_path):
        os.remove(dump_path)
    result = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result, input1 + input2 + input2)


def test_conv2d_vgf_count(sdk_tools, resources_helper, numpy_helper):

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
    conv2dCode = np.fromfile(conv2d_spv_path, dtype=np.uint32)

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Graph, "conv2d", "main", conv2dCode)

    conv2dInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 16, 16, 16], []
    )
    conv2dOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 8, 8, 16], []
    )

    conv2dInputBinding = encoder.AddBindingSlot(0, conv2dInput)
    conv2dOutputBinding = encoder.AddBindingSlot(1, conv2dOutput)

    conv2dDescSetInfo = encoder.AddDescriptorSetInfo(
        [conv2dInputBinding, conv2dOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [conv2dInputBinding],
        ["conv2dInput"],
        [conv2dOutputBinding],
        ["conv2dOutput"],
    )

    const0Shape = [16, 2, 2, 16]
    constantResource0 = encoder.AddConstantResource(VK_FORMAT_R8_SINT, const0Shape, [])
    constantData0 = np.full((16 * 2 * 2 * 16), 1, dtype=np.int8)

    constantRef0 = encoder.AddConstant(constantResource0, constantData0)

    segment0 = encoder.AddSegmentInfo(
        module0,
        "conv2d_graph_segment",
        [conv2dDescSetInfo],
        [conv2dInputBinding],
        [conv2dOutputBinding],
        [constantRef0],
    )

    encoder.Finish()

    vgfStream = io.FileIO(resources_helper.get_testenv_path("conv2d.vgf"), mode="wb")
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    vgfStream = io.FileIO(resources_helper.get_testenv_path("conv2d.vgf"), mode="rb")
    buffer = memoryview(vgfStream.read())

    assert buffer.nbytes >= vgf.HeaderSize()

    headerDecoder = vgf.CreateHeaderDecoder(buffer, buffer.nbytes)
    assert headerDecoder.IsValid()
    assert headerDecoder.CheckVersion()

    assert vgf.VerifyModuleTable(
        buffer[headerDecoder.GetModuleTableOffset() :],
        headerDecoder.GetModuleTableSize(),
    )
    moduleDecoder = vgf.CreateModuleTableDecoder(
        buffer[headerDecoder.GetModuleTableOffset() :],
        headerDecoder.GetModuleTableSize(),
    )

    assert moduleDecoder.getModuleCode(module0.reference) == memoryview(conv2dCode)

    vgfStream.close()

    input = numpy_helper.generate(
        [1, 16, 16, 16], dtype=np.int8, filename="conv2dInput.npy"
    )
    sdk_tools.run_scenario("test_vgf_graph/conv2d.json", options=["--repeat", "2"])


def test_enable_pipeline_cache_repeat_run(
    sdk_tools, resources_helper, numpy_helper, capfd
):
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
            "--repeat=2",
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
            "--repeat=2",
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

    result_third = numpy_helper.load("outBufferAdd2.npy", np.float32)
    assert np.array_equal(result_third, result_first)

    files = list(cache_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".cache"

    cache_data_third = files[0].read_bytes()
    assert len(cache_data_third) > 0
