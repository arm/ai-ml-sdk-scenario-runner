#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for scenarios combining shaders and VGF graphs."""
import io

import numpy as np
import pytest
import vgfpy as vgf

DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000
VK_FORMAT_R8_SINT = 14
pretendVulkanHeaderVersion = 123

pytestmark = pytest.mark.shader_and_vgf_graph


def test_compute_graph_compute_sandwich(sdk_tools, resources_helper, numpy_helper):

    sdk_tools.compile_shader("test_vgf_graph/add_shader.comp")
    sdk_tools.compile_shader("test_vgf_graph/sub_shader.comp")

    maxpool_spv_path = sdk_tools.assemble_spirv(
        "test_vgf_graph/maxpool.spvasm",
        {
            "INPUT_SET": "0",
            "INPUT_BINDING": "0",
            "OUTPUT_SET": "0",
            "OUTPUT_BINDING": "1",
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
    maxpoolOutputBinding = encoder.AddBindingSlot(1, maxpoolOutput)

    maxpoolDescSetInfo = encoder.AddDescriptorSetInfo(
        [maxpoolInputBinding, maxpoolOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [maxpoolInputBinding],
        ["maxpoolInput"],
        [maxpoolOutputBinding],
        ["maxpoolOutput"],
    )

    encoder.AddSegmentInfo(
        module0,
        "maxpool_graph_segment",
        [maxpoolDescSetInfo],
        [maxpoolInputBinding],
        [maxpoolOutputBinding],
        [],
    )

    encoder.Finish()

    graph_vgf_path = resources_helper.get_testenv_path("simple_graph.vgf")
    vgf_stream = io.FileIO(graph_vgf_path, mode="wb")
    assert encoder.WriteTo(vgf_stream)
    vgf_stream.close()

    vgf_stream = io.FileIO(graph_vgf_path, mode="rb")
    buffer = memoryview(vgf_stream.read())
    assert buffer.nbytes >= vgf.HeaderSize()

    header_decoder = vgf.CreateHeaderDecoder(buffer, buffer.nbytes)
    assert header_decoder is not None

    module_decoder = vgf.CreateModuleTableDecoder(
        buffer[header_decoder.GetModuleTableOffset() :],
        header_decoder.GetModuleTableSize(),
    )
    assert module_decoder is not None
    assert module_decoder.size() == 1
    assert module_decoder.getModuleType(module0.reference) == vgf.ModuleType.Graph
    assert module_decoder.getModuleCode(module0.reference) == memoryview(maxpoolCode)
    assert module_decoder.getModuleEntryPoint(module0.reference) == "main"

    vgf_stream.close()

    addInput0 = numpy_helper.generate(
        [1, 16, 16, 16], dtype=np.int8, filename="addInput0.npy", data=[5] * 4096
    )
    addInput1 = numpy_helper.generate(
        [1, 16, 16, 16], dtype=np.int8, filename="addInput1.npy", data=[4] * 4096
    )
    subInput = numpy_helper.generate(
        [1, 8, 8, 16], dtype=np.int8, filename="subInput.npy", data=[3] * 1024
    )

    sdk_tools.run_scenario("test_shader_and_vgf_graph/compute_graph_compute.json")

    addOutput = numpy_helper.load("addOutput.npy", np.int8)
    graphOutput = numpy_helper.load("graphOutput.npy", np.int8)
    finalOutput = numpy_helper.load("subOutput.npy", np.int8)

    expectedAdd = np.add(addInput0, addInput1)
    expectedGraph = np.full((1, 8, 8, 16), expectedAdd.max(), dtype=np.int8)
    expectedFinal = np.subtract(expectedGraph, subInput)

    assert np.array_equal(addOutput, expectedAdd)
    assert np.array_equal(graphOutput, expectedGraph)
    assert np.array_equal(finalOutput, expectedFinal)
