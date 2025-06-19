#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Tests for VGF shader. """
import io
import subprocess

import numpy as np
import pytest
import vgfpy as vgf

DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6
DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000
VK_FORMAT_R8_SINT = 14
pretendVulkanHeaderVersion = 123

pytestmark = pytest.mark.vgf_shader


def test_single_shader_module_in_vgf_with_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

    sdk_tools.compile_shader(
        "test_vgf_shader/add_shader.comp", output="single_shader.spv"
    )

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_one", "main")

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )
    shaderInput2 = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )
    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )

    shaderInputBinding = encoder.AddBindingSlot(0, shaderInput)
    shaderInput2Binding = encoder.AddBindingSlot(1, shaderInput2)
    shaderOutputBinding = encoder.AddBindingSlot(2, shaderOutput)

    shaderDescSetInfo = encoder.AddDescriptorSetInfo(
        [shaderInputBinding, shaderInput2Binding, shaderOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [shaderInputBinding, shaderInput2Binding],
        ["shaderInput", "shaderInput2"],
        [shaderOutputBinding],
        ["shaderOutput"],
    )

    segment0 = encoder.AddSegmentInfo(
        module0,
        "shader_segment",
        [shaderDescSetInfo],
        [shaderInputBinding, shaderInput2Binding],
        [shaderOutputBinding],
        [],
        [5, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("single_shader.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    input1 = numpy_helper.generate(
        [5], dtype=np.uint8, filename="input.npy", data=[42] * 5
    )
    input2 = numpy_helper.generate(
        [5], dtype=np.uint8, filename="input2.npy", data=[1] * 5
    )

    sdk_tools.run_scenario("test_vgf_shader/single_shader_module.json")

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, input1 + input2)

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("single_shader.vgf"), mode="rb"
    )
    buffer = memoryview(vgfStream.read())

    assert buffer.nbytes >= vgf.HeaderSize()

    headerDecoder = vgf.CreateHeaderDecoder(buffer)
    assert headerDecoder.IsValid()
    assert headerDecoder.CheckVersion()

    assert vgf.VerifyModuleTable(
        buffer[headerDecoder.GetModuleTableOffset() :],
        headerDecoder.GetModuleTableSize(),
    )
    moduleDecoder = vgf.CreateModuleTableDecoder(
        buffer[headerDecoder.GetModuleTableOffset() :]
    )

    assert moduleDecoder.size() == 1
    assert moduleDecoder.getModuleType(module0.reference) == vgf.ModuleType.Compute
    assert moduleDecoder.getModuleName(module0.reference) == "add_one"
    assert not moduleDecoder.hasSPIRV(module0.reference)
    assert moduleDecoder.getModuleEntryPoint(module0.reference) == "main"
    assert moduleDecoder.getModuleCode(module0.reference) is None


def test_single_shader_module_in_vgf_without_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

    spvFile = sdk_tools.compile_shader("test_vgf_shader/add_shader.comp")
    spv = np.fromfile(spvFile, dtype=np.uint32)

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_one", "main", spv)

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )
    shaderInput2 = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )
    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )

    shaderInputBinding = encoder.AddBindingSlot(0, shaderInput)
    shaderInput2Binding = encoder.AddBindingSlot(1, shaderInput2)
    shaderOutputBinding = encoder.AddBindingSlot(2, shaderOutput)

    shaderDescSetInfo = encoder.AddDescriptorSetInfo(
        [shaderInputBinding, shaderInput2Binding, shaderOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [shaderInputBinding, shaderInput2Binding],
        ["shaderInput", "shaderInput2"],
        [shaderOutputBinding],
        ["shaderOutput"],
    )

    segment0 = encoder.AddSegmentInfo(
        module0,
        "shader_segment",
        [shaderDescSetInfo],
        [shaderInputBinding, shaderInput2Binding],
        [shaderOutputBinding],
        [],
        [5, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("single_shader.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    input1 = numpy_helper.generate(
        [5], dtype=np.uint8, filename="input.npy", data=[42] * 5
    )
    input2 = numpy_helper.generate(
        [5], dtype=np.uint8, filename="input2.npy", data=[1] * 5
    )

    sdk_tools.run_scenario(
        "test_vgf_shader/single_shader_module_without_shader_substitution.json"
    )

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, input1 + input2)

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("single_shader.vgf"), mode="rb"
    )
    buffer = memoryview(vgfStream.read())

    assert buffer.nbytes >= vgf.HeaderSize()

    headerDecoder = vgf.CreateHeaderDecoder(buffer)
    assert headerDecoder.IsValid()
    assert headerDecoder.CheckVersion()

    assert vgf.VerifyModuleTable(
        buffer[headerDecoder.GetModuleTableOffset() :],
        headerDecoder.GetModuleTableSize(),
    )
    moduleDecoder = vgf.CreateModuleTableDecoder(
        buffer[headerDecoder.GetModuleTableOffset() :]
    )

    assert moduleDecoder.size() == 1
    assert moduleDecoder.getModuleType(module0.reference) == vgf.ModuleType.Compute
    assert moduleDecoder.getModuleName(module0.reference) == "add_one"
    assert moduleDecoder.hasSPIRV(module0.reference)
    assert moduleDecoder.getModuleEntryPoint(module0.reference) == "main"
    assert moduleDecoder.getModuleCode(module0.reference) == memoryview(spv)


def test_two_shader_module(sdk_tools, resources_helper, numpy_helper):

    sdk_tools.compile_shader(
        "test_vgf_shader/add_one_shader.comp", output="single_shader0.spv"
    )
    sdk_tools.compile_shader(
        "test_vgf_shader/add_two_shader.comp", output="single_shader1.spv"
    )

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_one", "main")
    module1 = encoder.AddModule(vgf.ModuleType.Compute, "add_two", "main")

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [1024], []
    )
    shaderIntermediate = encoder.AddIntermediateResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [1024], []
    )
    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [1024], []
    )

    shaderInputBinding = encoder.AddBindingSlot(0, shaderInput)
    shaderIntermediateBinding = encoder.AddBindingSlot(1, shaderIntermediate)
    shaderOutputBinding = encoder.AddBindingSlot(2, shaderOutput)

    shader0DescSetInfo = encoder.AddDescriptorSetInfo(
        [shaderInputBinding, shaderIntermediateBinding]
    )

    shader1DescSetInfo = encoder.AddDescriptorSetInfo(
        [shaderIntermediateBinding, shaderOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [shaderInputBinding],
        ["shaderInput"],
        [shaderOutputBinding],
        ["shaderOutput"],
    )

    segment0 = encoder.AddSegmentInfo(
        module0,
        "shader_segment0",
        [shader0DescSetInfo],
        [shaderInputBinding],
        [shaderIntermediateBinding],
        [],
        [1024, 1, 1],
    )

    segment1 = encoder.AddSegmentInfo(
        module1,
        "shader_segment1",
        [shader1DescSetInfo],
        [shaderIntermediateBinding],
        [shaderOutputBinding],
        [],
        [1024, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("two_shader_modules.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    input = numpy_helper.generate(
        [1024], dtype=np.uint8, filename="input.npy", data=[42] * 1024
    )

    sdk_tools.run_scenario("test_vgf_shader/two_shader_module.json")

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, np.add(input, 3))

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("two_shader_modules.vgf"), mode="rb"
    )
    buffer = memoryview(vgfStream.read())

    assert buffer.nbytes >= vgf.HeaderSize()

    headerDecoder = vgf.CreateHeaderDecoder(buffer)
    assert headerDecoder.IsValid()
    assert headerDecoder.CheckVersion()

    assert vgf.VerifyModuleTable(
        buffer[headerDecoder.GetModuleTableOffset() :],
        headerDecoder.GetModuleTableSize(),
    )
    moduleDecoder = vgf.CreateModuleTableDecoder(
        buffer[headerDecoder.GetModuleTableOffset() :]
    )

    assert moduleDecoder.size() == 2

    assert moduleDecoder.getModuleType(module0.reference) == vgf.ModuleType.Compute
    assert moduleDecoder.getModuleName(module0.reference) == "add_one"
    assert not moduleDecoder.hasSPIRV(module0.reference)
    assert moduleDecoder.getModuleEntryPoint(module0.reference) == "main"
    assert moduleDecoder.getModuleCode(module0.reference) is None

    assert moduleDecoder.getModuleType(module1.reference) == vgf.ModuleType.Compute
    assert moduleDecoder.getModuleName(module1.reference) == "add_two"
    assert not moduleDecoder.hasSPIRV(module1.reference)
    assert moduleDecoder.getModuleEntryPoint(module1.reference) == "main"
    assert moduleDecoder.getModuleCode(module1.reference) is None


@pytest.mark.parametrize(
    "resource_type, buffer_size",
    [
        (DESCRIPTOR_TYPE_TENSOR_ARM, [5]),
        (DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, [6]),
    ],
)
def test_single_shader_module_in_vgf_without_shader_substitution_mismatching_resource_type_or_buffer_size(
    sdk_tools, resources_helper, numpy_helper, resource_type, buffer_size
):

    spvFile = sdk_tools.compile_shader("test_vgf_shader/add_shader.comp")
    spv = np.fromfile(spvFile, dtype=np.uint32)

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_one", "main", spv)

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )
    shaderInput2 = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [5], []
    )
    shaderOutput = encoder.AddOutputResource(
        resource_type, VK_FORMAT_R8_SINT, buffer_size, []
    )

    shaderInputBinding = encoder.AddBindingSlot(0, shaderInput)
    shaderInput2Binding = encoder.AddBindingSlot(1, shaderInput2)
    shaderOutputBinding = encoder.AddBindingSlot(2, shaderOutput)

    shaderDescSetInfo = encoder.AddDescriptorSetInfo(
        [shaderInputBinding, shaderInput2Binding, shaderOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [shaderInputBinding, shaderInput2Binding],
        ["shaderInput", "shaderInput2"],
        [shaderOutputBinding],
        ["shaderOutput"],
    )

    segment0 = encoder.AddSegmentInfo(
        module0,
        "shader_segment",
        [shaderDescSetInfo],
        [shaderInputBinding, shaderInput2Binding],
        [shaderOutputBinding],
        [],
        [5, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("single_shader.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    input1 = numpy_helper.generate(
        [5], dtype=np.uint8, filename="input.npy", data=[42] * 5
    )
    input2 = numpy_helper.generate(
        [5], dtype=np.uint8, filename="input2.npy", data=[1] * 5
    )

    with pytest.raises(subprocess.CalledProcessError):
        sdk_tools.run_scenario(
            "test_vgf_shader/single_shader_module_without_shader_substitution.json"
        )
