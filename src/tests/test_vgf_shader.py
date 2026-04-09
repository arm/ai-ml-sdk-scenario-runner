#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import io
import subprocess

import numpy as np
import pytest
import vgfpy as vgf

"""Tests for VGF shader."""


DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6
DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000
VK_FORMAT_R8_SINT = 14
pretendVulkanHeaderVersion = 123

pytestmark = pytest.mark.vgf_shader


def _skip_if_hlsl_unsupported(sdk_tools) -> None:
    if sdk_tools.hlsl_compiler.path is None:
        pytest.skip("HLSL compiler not provided; skipping HLSL-dependent tests.")


def test_single_spirv_shader_module_in_vgf_with_shader_substitution(
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


def test_single_spirv_shader_module_in_vgf_without_shader_substitution(
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


def test_single_glsl_shader_module_in_vgf_with_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

    resources_helper.prepare_shader(
        "test_vgf_shader/add_shader.comp", output="single_shader.comp"
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

    sdk_tools.run_scenario("test_vgf_shader/single_shader_module_glsl.json")

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, input1 + input2)


def test_single_glsl_shader_module_in_vgf_without_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

    shaderCode = resources_helper.get_shader_path(
        "test_vgf_shader/add_shader.comp"
    ).read_text()

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(
        vgf.ModuleType.Compute, "add_one", "main", vgf.ShaderType.Glsl, shaderCode
    )

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
        "test_vgf_shader/single_shader_module_without_shader_substitution_glsl.json"
    )

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, input1 + input2)


def test_single_hlsl_shader_module_in_vgf_with_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):
    _skip_if_hlsl_unsupported(sdk_tools)

    resources_helper.prepare_shader(
        "test_vgf_shader/add_shader.hlsl", output="single_shader.hlsl"
    )

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_one", "main")

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [8], []
    )
    shaderInput2 = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [8], []
    )
    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [8], []
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

    encoder.AddSegmentInfo(
        module0,
        "shader_segment",
        [shaderDescSetInfo],
        [shaderInputBinding, shaderInput2Binding],
        [shaderOutputBinding],
        [],
        [2, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("single_shader.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    input1 = numpy_helper.generate(
        [8], dtype=np.uint8, filename="input.npy", data=[42] * 5 + [0] * 3
    )
    input2 = numpy_helper.generate(
        [8], dtype=np.uint8, filename="input2.npy", data=[1] * 5 + [0] * 3
    )

    sdk_tools.run_scenario("test_vgf_shader/single_shader_module_hlsl.json")

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, input1 + input2)


def test_single_hlsl_shader_module_in_vgf_without_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):
    _skip_if_hlsl_unsupported(sdk_tools)

    shaderCode = resources_helper.get_shader_path(
        "test_vgf_shader/add_shader.hlsl"
    ).read_text()

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(
        vgf.ModuleType.Compute, "add_one", "main", vgf.ShaderType.Hlsl, shaderCode
    )

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [8], []
    )
    shaderInput2 = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [8], []
    )
    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [8], []
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

    encoder.AddSegmentInfo(
        module0,
        "shader_segment",
        [shaderDescSetInfo],
        [shaderInputBinding, shaderInput2Binding],
        [shaderOutputBinding],
        [],
        [2, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("single_shader.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    input1 = numpy_helper.generate(
        [8], dtype=np.uint8, filename="input.npy", data=[42] * 5 + [0] * 3
    )
    input2 = numpy_helper.generate(
        [8], dtype=np.uint8, filename="input2.npy", data=[1] * 5 + [0] * 3
    )

    sdk_tools.run_scenario(
        "test_vgf_shader/single_shader_module_without_shader_substitution_hlsl.json"
    )

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, input1 + input2)


def test_two_spirv_shader_module_in_vgf_with_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

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


def test_two_spirv_shader_module_in_vgf_without_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

    spvFile0 = sdk_tools.compile_shader("test_vgf_shader/add_one_shader.comp")
    spv0 = np.fromfile(spvFile0, dtype=np.uint32)
    spvFile1 = sdk_tools.compile_shader("test_vgf_shader/add_two_shader.comp")
    spv1 = np.fromfile(spvFile1, dtype=np.uint32)

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_one", "main", spv0)
    module1 = encoder.AddModule(vgf.ModuleType.Compute, "add_two", "main", spv1)

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

    sdk_tools.run_scenario(
        "test_vgf_shader/two_shader_module_without_shader_substitution.json"
    )

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, np.add(input, 3))


def test_two_glsl_shader_module_in_vgf_with_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

    resources_helper.prepare_shader(
        "test_vgf_shader/add_one_shader.comp", output="single_shader0.comp"
    )
    resources_helper.prepare_shader(
        "test_vgf_shader/add_two_shader.comp", output="single_shader1.comp"
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

    sdk_tools.run_scenario("test_vgf_shader/two_shader_module_glsl.json")

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, np.add(input, 3))


def test_two_glsl_shader_module_in_vgf_without_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):

    shaderCode0 = resources_helper.get_shader_path(
        "test_vgf_shader/add_one_shader.comp"
    ).read_text()
    shaderCode1 = resources_helper.get_shader_path(
        "test_vgf_shader/add_two_shader.comp"
    ).read_text()

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(
        vgf.ModuleType.Compute, "add_one", "main", vgf.ShaderType.Glsl, shaderCode0
    )
    module1 = encoder.AddModule(
        vgf.ModuleType.Compute, "add_two", "main", vgf.ShaderType.Glsl, shaderCode1
    )

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

    sdk_tools.run_scenario(
        "test_vgf_shader/two_shader_module_without_shader_substitution.json"
    )

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, np.add(input, 3))


def test_two_hlsl_shader_module_in_vgf_with_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):
    _skip_if_hlsl_unsupported(sdk_tools)

    resources_helper.prepare_shader(
        "test_vgf_shader/add_one_shader.hlsl", output="single_shader0.hlsl"
    )
    resources_helper.prepare_shader(
        "test_vgf_shader/add_two_shader.hlsl", output="single_shader1.hlsl"
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

    encoder.AddSegmentInfo(
        module0,
        "shader_segment0",
        [shader0DescSetInfo],
        [shaderInputBinding],
        [shaderIntermediateBinding],
        [],
        [256, 1, 1],
    )

    encoder.AddSegmentInfo(
        module1,
        "shader_segment1",
        [shader1DescSetInfo],
        [shaderIntermediateBinding],
        [shaderOutputBinding],
        [],
        [256, 1, 1],
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

    sdk_tools.run_scenario("test_vgf_shader/two_shader_module_hlsl.json")

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, np.add(input, 3))


def test_two_hlsl_shader_module_in_vgf_without_shader_substitution(
    sdk_tools, resources_helper, numpy_helper
):
    _skip_if_hlsl_unsupported(sdk_tools)

    shaderCode0 = resources_helper.get_shader_path(
        "test_vgf_shader/add_one_shader.hlsl"
    ).read_text()
    shaderCode1 = resources_helper.get_shader_path(
        "test_vgf_shader/add_two_shader.hlsl"
    ).read_text()

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(
        vgf.ModuleType.Compute, "add_one", "main", vgf.ShaderType.Hlsl, shaderCode0
    )
    module1 = encoder.AddModule(
        vgf.ModuleType.Compute, "add_two", "main", vgf.ShaderType.Hlsl, shaderCode1
    )

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

    encoder.AddSegmentInfo(
        module0,
        "shader_segment0",
        [shader0DescSetInfo],
        [shaderInputBinding],
        [shaderIntermediateBinding],
        [],
        [256, 1, 1],
    )

    encoder.AddSegmentInfo(
        module1,
        "shader_segment1",
        [shader1DescSetInfo],
        [shaderIntermediateBinding],
        [shaderOutputBinding],
        [],
        [256, 1, 1],
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

    sdk_tools.run_scenario(
        "test_vgf_shader/two_shader_module_without_shader_substitution_hlsl.json"
    )

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, np.add(input, 3))


@pytest.mark.parametrize(
    "resource_type, buffer_size",
    [
        (DESCRIPTOR_TYPE_TENSOR_ARM, [5]),
        (DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, [6]),
    ],
)
def test_single_spirv_shader_module_in_vgf_without_shader_substitution_mismatching_resource_type_or_buffer_size(
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
