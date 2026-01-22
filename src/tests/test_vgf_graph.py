#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for VGF graph. """
import io
import itertools
import os
import subprocess

import numpy as np
import pytest
import vgfpy as vgf

DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6
DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000
VK_FORMAT_R8_UINT = 13
VK_FORMAT_R8_SINT = 14
pretendVulkanHeaderVersion = 123

pytestmark = pytest.mark.vgf_graph


def test_conv2d_vgf(sdk_tools, resources_helper, numpy_helper):

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

    sdk_tools.run_scenario("test_vgf_graph/conv2d.json")


@pytest.mark.parametrize(
    "resource_type, resource_data_type, resource_shape",
    [
        (DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [1, 16, 16, 16]),
        (DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_UINT, [1, 16, 16, 16]),
        (DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 8, 8, 16]),
    ],
)
def test_conv2d_vgf_mismatching_resource_type_or_resource_data_type_or_resource_shape(
    sdk_tools,
    resources_helper,
    numpy_helper,
    resource_type,
    resource_data_type,
    resource_shape,
):

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
        resource_type, resource_data_type, resource_shape, []
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

    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario("test_vgf_graph/conv2d.json")


def test_maxpool_conv2d_parallel_vgf(sdk_tools, resources_helper, numpy_helper):

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

    conv2d_spv_path = sdk_tools.assemble_spirv(
        "test_vgf_graph/conv2d.spvasm",
        {
            "INPUT_SET": "0",
            "INPUT_BINDING": "1",
            "OUTPUT_SET": "0",
            "OUTPUT_BINDING": "3",
        },
    )
    sdk_tools.validate_spirv(conv2d_spv_path)
    conv2dCode = np.fromfile(conv2d_spv_path, dtype=np.uint32)

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Graph, "maxpool", "main", maxpoolCode)
    module1 = encoder.AddModule(vgf.ModuleType.Graph, "conv2d", "main", conv2dCode)

    maxpoolInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 16, 16, 16], []
    )
    conv2dInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 16, 16, 16], []
    )
    maxpoolOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 8, 8, 16], []
    )
    conv2dOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R8_SINT, [1, 8, 8, 16], []
    )

    maxpoolInputBinding = encoder.AddBindingSlot(0, maxpoolInput)
    conv2dInputBinding = encoder.AddBindingSlot(1, conv2dInput)
    maxpoolOutputBinding = encoder.AddBindingSlot(2, maxpoolOutput)
    conv2dOutputBinding = encoder.AddBindingSlot(3, conv2dOutput)

    maxpoolDescSetInfo = encoder.AddDescriptorSetInfo(
        [maxpoolInputBinding, maxpoolOutputBinding]
    )

    conv2dDescSetInfo = encoder.AddDescriptorSetInfo(
        [conv2dInputBinding, conv2dOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [maxpoolInputBinding, conv2dInputBinding],
        ["maxpoolInput", "conv2dInput"],
        [maxpoolOutputBinding, conv2dOutputBinding],
        ["maxpoolOutput", "conv2dOutput"],
    )

    const0Shape = [16, 2, 2, 16]
    constantResource0 = encoder.AddConstantResource(VK_FORMAT_R8_SINT, const0Shape, [])
    constantData0 = np.full((16 * 2 * 2 * 16), 1, dtype=np.int8)

    constantRef0 = encoder.AddConstant(constantResource0, constantData0)

    segment0 = encoder.AddSegmentInfo(
        module0,
        "maxpool_graph_segment",
        [maxpoolDescSetInfo],
        [maxpoolInputBinding],
        [maxpoolOutputBinding],
        [],
    )

    segment1 = encoder.AddSegmentInfo(
        module1,
        "conv2d_graph_segment",
        [conv2dDescSetInfo],
        [conv2dInputBinding],
        [conv2dOutputBinding],
        [constantRef0],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("multiple_modules.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("multiple_modules.vgf"), mode="rb"
    )
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

    assert moduleDecoder.size() == 2
    assert moduleDecoder.getModuleType(module0.reference) == vgf.ModuleType.Graph
    assert moduleDecoder.getModuleType(module1.reference) == vgf.ModuleType.Graph
    assert moduleDecoder.hasSPIRV(module0.reference)
    assert moduleDecoder.hasSPIRV(module1.reference)
    assert moduleDecoder.getModuleEntryPoint(module0.reference) == "main"
    assert moduleDecoder.getModuleEntryPoint(module1.reference) == "main"
    assert moduleDecoder.getModuleCode(module0.reference) == memoryview(maxpoolCode)
    assert moduleDecoder.getModuleCode(module1.reference) == memoryview(conv2dCode)

    vgfStream.close()

    maxpoolInput = numpy_helper.generate(
        [1, 16, 16, 16], dtype=np.int8, filename="maxpoolInput.npy"
    )
    conv2dInput = numpy_helper.generate(
        [1, 16, 16, 16], dtype=np.int8, filename="conv2dInput.npy"
    )

    sdk_tools.run_scenario("test_vgf_graph/maxpool_conv2d.json")
