#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import io
import json
import subprocess

import numpy as np
import pytest
import vgfpy as vgf

"""Tests for VGF shader."""


DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6
DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000
VK_FORMAT_R8_SINT = 14
VK_FORMAT_R16_UINT = 74
VK_FORMAT_R32_UINT = 98
VK_FORMAT_R32_SFLOAT = 100
pretendVulkanHeaderVersion = 123

pytestmark = pytest.mark.vgf_shader

INPUT0_BINDING = 4
INPUT1_BINDING = 9
OUTPUT_BINDING = 17
INPUT_SET = 0
OUTPUT_SET = 1


def _skip_if_hlsl_unsupported(sdk_tools) -> None:
    if sdk_tools.hlsl_compiler.path is None:
        pytest.skip("HLSL compiler not provided; skipping HLSL-dependent tests.")


def _write_two_descriptor_sets_vgf(
    resources_helper,
    vgf_filename: str,
    *,
    explicit_set_indices: bool,
    swap_descriptor_order: bool,
) -> None:
    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_one", "main")

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [10], []
    )
    shaderInput2 = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [10], []
    )
    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [10], []
    )

    shaderInputBinding = encoder.AddBindingSlot(INPUT0_BINDING, shaderInput)
    shaderInput2Binding = encoder.AddBindingSlot(INPUT1_BINDING, shaderInput2)
    shaderOutputBinding = encoder.AddBindingSlot(OUTPUT_BINDING, shaderOutput)

    input_bindings = [shaderInputBinding, shaderInput2Binding]
    output_bindings = [shaderOutputBinding]
    if explicit_set_indices:
        # These set indices must match shader layout(set=...) declarations.
        inputDescSetInfo = encoder.AddDescriptorSetInfo(input_bindings, INPUT_SET)
        outputDescSetInfo = encoder.AddDescriptorSetInfo(output_bindings, OUTPUT_SET)
    else:
        inputDescSetInfo = encoder.AddDescriptorSetInfo(input_bindings)
        outputDescSetInfo = encoder.AddDescriptorSetInfo(output_bindings)

    descriptor_sets = (
        [outputDescSetInfo, inputDescSetInfo]
        if swap_descriptor_order
        else [inputDescSetInfo, outputDescSetInfo]
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
        descriptor_sets,
        [shaderInputBinding, shaderInput2Binding],
        [shaderOutputBinding],
        [],
        [10, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(resources_helper.get_testenv_path(vgf_filename), mode="wb")
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()


def _write_graph_push_constants_vgf(resources_helper, vgf_filename: str) -> None:
    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "add_push", "main")

    shaderInput = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R32_SFLOAT, [10], []
    )
    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R32_SFLOAT, [10], []
    )

    shaderInputBinding = encoder.AddBindingSlot(0, shaderInput)
    shaderOutputBinding = encoder.AddBindingSlot(1, shaderOutput)
    shaderDescSetInfo = encoder.AddDescriptorSetInfo(
        [shaderInputBinding, shaderOutputBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [shaderInputBinding],
        ["shaderInput"],
        [shaderOutputBinding],
        ["shaderOutput"],
    )

    encoder.AddSegmentInfo(
        module0,
        "shader_segment",
        [shaderDescSetInfo],
        [shaderInputBinding],
        [shaderOutputBinding],
        [],
        [10, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(resources_helper.get_testenv_path(vgf_filename), mode="wb")
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()


def _write_graph_specialization_constants_vgf(
    resources_helper, vgf_filename: str, spv: np.ndarray
) -> None:
    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "write_spec_const", "main", spv)

    shaderOutput = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R32_UINT, [1], []
    )

    shaderOutputBinding = encoder.AddBindingSlot(0, shaderOutput)
    shaderDescSetInfo = encoder.AddDescriptorSetInfo([shaderOutputBinding])

    encoder.AddModelSequenceInputsOutputs(
        [],
        [],
        [shaderOutputBinding],
        ["shaderOutput"],
    )

    encoder.AddSegmentInfo(
        module0,
        "shader_segment",
        [shaderDescSetInfo],
        [],
        [shaderOutputBinding],
        [],
        [1, 1, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(resources_helper.get_testenv_path(vgf_filename), mode="wb")
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()


def test_graph_shader_push_constants(sdk_tools, resources_helper, numpy_helper):
    sdk_tools.compile_shader(
        "test_shader/add_shader_with_push_constants.comp",
        output="graph_push_constants.spv",
    )
    _write_graph_push_constants_vgf(resources_helper, "graph_push_constants.vgf")

    input_data = numpy_helper.generate(
        [10],
        dtype=np.float32,
        filename="input.npy",
        data=[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
    )
    push_constants = numpy_helper.generate(
        [10],
        dtype=np.float32,
        filename="data.npy",
        data=[0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 4.0, 3.0, 2.0, 1.0],
    )

    sdk_tools.run_scenario("test_vgf_shader/graph_shader_push_constants.json")

    result = numpy_helper.load("output.npy", np.float32)
    assert np.array_equal(result, input_data + push_constants)


def test_embedded_graph_shader_specialization_constants(
    sdk_tools, resources_helper, numpy_helper
):
    spv_file = sdk_tools.compile_shader("test_spec_const/uint_shader.comp")
    spv = np.fromfile(spv_file, dtype=np.uint32)
    _write_graph_specialization_constants_vgf(
        resources_helper, "graph_specialization_constants.vgf", spv
    )

    sdk_tools.run_scenario("test_vgf_shader/graph_shader_specialization_constants.json")

    result = numpy_helper.load("output.npy", np.uint32)
    assert np.array_equal(result, np.array([42], dtype=np.uint32))


def test_vgf_explicit_descriptor_set_index_is_used(
    sdk_tools, resources_helper, numpy_helper
):
    sdk_tools.compile_shader(
        "test_vgf_shader/add_shader_non_sequential_bindings.comp",
        {"TestType": "int8_t"},
        output="add_shader.spv",
    )
    _write_two_descriptor_sets_vgf(
        resources_helper,
        "two_descriptor_sets_explicit.vgf",
        explicit_set_indices=True,
        swap_descriptor_order=True,
    )

    input1 = numpy_helper.generate(
        [10], dtype=np.uint8, filename="input.npy", data=[42] * 10
    )
    input2 = numpy_helper.generate(
        [10], dtype=np.uint8, filename="input2.npy", data=[1] * 10
    )

    sdk_tools.run_scenario(
        "test_vgf_shader/two_descriptor_sets.json",
        {"{VGF_FILE}": "two_descriptor_sets_explicit.vgf"},
    )

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, input1 + input2)


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

    dump_path = resources_helper.get_testenv_path("two_shader_module_profiling.json")
    sdk_tools.run_scenario(
        "test_vgf_shader/two_shader_module.json",
        options=["--profiling-dump-path", dump_path.as_posix()],
    )

    with open(dump_path, encoding="utf-8") as dump_file:
        profiling_data = json.load(dump_file)

    timestamps = profiling_data["Timestamps"]
    assert [timestamp["Command type"] for timestamp in timestamps] == [
        "ComputeDispatch",
        "ComputeDispatch",
    ]
    assert [timestamp["Command name"] for timestamp in timestamps] == [
        "vgfGraph/shader_segment0",
        "vgfGraph/shader_segment1",
    ]

    result = numpy_helper.load("output.npy", np.uint8)
    assert np.array_equal(result, np.add(input, 3))


def test_vgf_alias_group_for_internal_intermediates(
    sdk_tools, resources_helper, numpy_helper
):
    sdk_tools.compile_shader(
        "test_memory_aliasing/write_alias_buffer_u16.comp",
        output="write_alias_buffer_u16.spv",
    )
    sdk_tools.compile_shader(
        "test_memory_aliasing/copy_tensor_shader.comp",
        output="copy_tensor_shader.spv",
    )

    encoder = vgf.CreateEncoder(pretendVulkanHeaderVersion)

    module0 = encoder.AddModule(vgf.ModuleType.Compute, "write_alias_buffer", "main")
    module1 = encoder.AddModule(vgf.ModuleType.Compute, "copy_alias_tensor", "main")

    aliasBuffer = encoder.AddIntermediateResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R8_SINT, [512], [], 7
    )
    aliasTensor = encoder.AddIntermediateResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R16_UINT, [1, 16, 16, 1], [], 7
    )
    outputTensor = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_TENSOR_ARM, VK_FORMAT_R16_UINT, [1, 16, 16, 1], []
    )

    aliasBufferBinding = encoder.AddBindingSlot(0, aliasBuffer)
    aliasTensorBinding = encoder.AddBindingSlot(0, aliasTensor)
    outputTensorBinding = encoder.AddBindingSlot(1, outputTensor)

    writerDescSetInfo = encoder.AddDescriptorSetInfo([aliasBufferBinding])
    readerDescSetInfo = encoder.AddDescriptorSetInfo(
        [aliasTensorBinding, outputTensorBinding]
    )

    encoder.AddModelSequenceInputsOutputs(
        [],
        [],
        [outputTensorBinding],
        ["outputTensor"],
    )

    encoder.AddSegmentInfo(
        module0,
        "write_alias_buffer_segment",
        [writerDescSetInfo],
        [],
        [aliasBufferBinding],
        [],
        [16, 16, 1],
    )
    encoder.AddSegmentInfo(
        module1,
        "copy_alias_tensor_segment",
        [readerDescSetInfo],
        [aliasTensorBinding],
        [outputTensorBinding],
        [],
        [16, 16, 1],
    )

    encoder.Finish()

    vgfStream = io.FileIO(
        resources_helper.get_testenv_path("alias_intermediates.vgf"), mode="wb"
    )
    assert encoder.WriteTo(vgfStream)
    vgfStream.close()

    sdk_tools.run_scenario(
        "test_vgf_shader/vgf_alias_group_intermediate_buffer_tensor.json"
    )

    result = numpy_helper.load("output.npy", np.uint16)
    expected = (np.arange(256, dtype=np.uint16) + np.uint16(5)).reshape(1, 16, 16, 1)

    assert np.array_equal(result, expected)


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
