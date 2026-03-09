#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import io
import json
import subprocess

import numpy as np
import pytest
import vgfpy as vgf

"""Tests for spv graph dispatch"""

DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000
VK_FORMAT_R8_SINT = 14
pretendVulkanHeaderVersion = 123


def test_maxpool(sdk_tools, resources_helper, numpy_helper):
    maxpool_spv_path = sdk_tools.assemble_spirv(
        "test_vgf_graph/maxpool.spvasm",
        {
            "INPUT_SET": "0",
            "INPUT_BINDING": "0",
            "OUTPUT_SET": "0",
            "OUTPUT_BINDING": "2",
        },
    )
    sdk_tools.validate_spirv(maxpool_spv_path)
    maxpoolCode = np.fromfile(maxpool_spv_path, dtype=np.uint32)

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)
    module0 = encoder.AddModule(vgf.ModuleType.Graph, "maxpool", "main", maxpoolCode)
    maxpoolInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 16, 16, 16], []
    )
    maxpoolOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 8, 8, 16], []
    )
    maxpoolInputBinding = encoder.AddBindingSlot(0, maxpoolInput)
    maxpoolOutputBinding = encoder.AddBindingSlot(2, maxpoolOutput)
    maxpoolDescSetInfo = encoder.AddDescriptorSetInfo(
        [maxpoolInputBinding, maxpoolOutputBinding]
    )
    encoder.AddModelSequenceInputsOutputs(
        [maxpoolInputBinding],
        ["maxpoolInput"],
        [maxpoolOutputBinding],
        ["maxpoolOutput"],
    )

    segment0 = encoder.AddSegmentInfo(
        module0,
        "maxpool_graph_segment",
        [maxpoolDescSetInfo],
        [maxpoolInputBinding],
        [maxpoolOutputBinding],
        [],
    )
    encoder.Finish()
    vgfStream = io.FileIO(resources_helper.get_testenv_path("maxpool.vgf"), mode="wb")
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    # Generate input tensor
    numpy_helper.generate([1, 16, 16, 16], dtype=np.int8, filename="maxpoolInput.npy")

    # Run VGF-backed scenario and capture result
    sdk_tools.run_scenario("test_spv_graph/maxpool.json")
    vgf_result = numpy_helper.load("maxpoolOutput.npy")

    # Run SPV-only scenario and capture result
    sdk_tools.run_scenario(
        "test_spv_graph/maxpool_spv.json",
        {
            "{SPV}": json.dumps(maxpool_spv_path.as_posix()),
            "{OUT}": json.dumps(
                resources_helper.get_testenv_path("maxpoolOutput_spv.npy").as_posix()
            ),
        },
    )
    spv_result = numpy_helper.load("maxpoolOutput_spv.npy")

    # Compare results
    assert np.array_equal(vgf_result, spv_result)


def test_conv2d_spv(sdk_tools, resources_helper, numpy_helper):

    # Input tensor: [1, 16, 16, 16] int8
    numpy_helper.generate(
        shape=[1, 16, 16, 16], dtype=np.int8, filename="conv2dInput.npy"
    )
    # Graph constant: [16, 2, 2, 16]
    const0_shape = [16, 2, 2, 16]
    numpy_helper.save(np.full(const0_shape, 1, dtype=np.int8), "graphConstant0.npy")

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

    # First set up VGF file
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

    constantResource0 = encoder.AddConstantResource(VK_FORMAT_R8_SINT, const0_shape, [])
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

    sdk_tools.run_scenario("test_spv_graph/conv2d.json")
    vgf_result = numpy_helper.load("conv2dOutput.npy")

    # Run same conv2d graph through spv only dispatch
    sdk_tools.run_scenario(
        "test_spv_graph/conv2d_spv.json",
        {
            "{SPV}": json.dumps(conv2d_spv_path.as_posix()),
            "{OUT}": json.dumps(
                resources_helper.get_testenv_path("conv2dOutput_spv.npy").as_posix()
            ),
            "{CONST}": json.dumps(
                resources_helper.get_testenv_path("graphConstant0.npy").as_posix()
            ),
            "{CONST_DIMS}": json.dumps(const0_shape),
        },
    )
    spv_result = numpy_helper.load("conv2dOutput_spv.npy")

    assert np.array_equal(vgf_result, spv_result)


def test_spv_graph_spv_path_unreadable(sdk_tools, resources_helper, numpy_helper):
    # Prepare inputs for maxpool
    numpy_helper.generate([1, 16, 16, 16], dtype=np.int8, filename="maxpoolInput.npy")

    # Point SPV path to a non-existent file
    missing_spv_path = resources_helper.get_testenv_path("does_not_exist.spv")
    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario(
            "test_spv_graph/maxpool_spv.json",
            {
                "{SPV}": json.dumps(missing_spv_path.as_posix()),
                "{OUT}": json.dumps(
                    resources_helper.get_testenv_path(
                        "maxpoolOutput_spv.npy"
                    ).as_posix()
                ),
            },
        )


def test_spv_graph_missing_graph_constant(sdk_tools, resources_helper, numpy_helper):
    # Prepare valid SPV and inputs
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

    numpy_helper.generate([1, 16, 16, 16], dtype=np.int8, filename="conv2dInput.npy")
    numpy_helper.save(np.full([16, 2, 2, 16], 1, dtype=np.int8), "graphConstant0.npy")

    # Rename the graph_constant uid so the graph_constants list cannot resolve it
    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario(
            "test_spv_graph/conv2d_spv.json",
            {
                "{SPV}": json.dumps(conv2d_spv_path.as_posix()),
                "{OUT}": json.dumps(
                    resources_helper.get_testenv_path("conv2dOutput_spv.npy").as_posix()
                ),
                "{CONST}": json.dumps(
                    resources_helper.get_testenv_path("graphConstant0.npy").as_posix()
                ),
                "{CONST_DIMS}": json.dumps([16, 2, 2, 16]),
                '"uid": "weights0"': '"uid": "weights_missing"',
            },
        )


def test_spv_graph_rejects_glsl_shader(sdk_tools, resources_helper, numpy_helper):
    # Prepare inputs for maxpool
    numpy_helper.generate([1, 16, 16, 16], dtype=np.int8, filename="maxpoolInput.npy")

    # Assemble a valid SPIR-V module for maxpool
    maxpool_spv_path = sdk_tools.assemble_spirv(
        "test_vgf_graph/maxpool.spvasm",
        {
            "INPUT_SET": "0",
            "INPUT_BINDING": "0",
            "OUTPUT_SET": "0",
            "OUTPUT_BINDING": "2",
        },
    )
    sdk_tools.validate_spirv(maxpool_spv_path)

    # Change shader type to GLSL in the scenario to trigger runtime rejection
    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario(
            "test_spv_graph/maxpool_spv.json",
            {
                "{SPV}": json.dumps(maxpool_spv_path.as_posix()),
                '"type": "SPIR-V"': '"type": "GLSL"',
                "{OUT}": json.dumps(
                    resources_helper.get_testenv_path(
                        "maxpoolOutput_spv.npy"
                    ).as_posix()
                ),
            },
        )
