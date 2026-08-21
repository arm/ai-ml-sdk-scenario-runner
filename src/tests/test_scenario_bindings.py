#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import json

import numpy as np
import pytest


def test_python_scenario_and_builder_types(sr):
    builder = sr.ScenarioBuilder()
    options = sr.ScenarioOptions()

    buffer_info = sr.BufferInfo()
    buffer_info.debug_name = "buffer"
    buffer_info.size = 4
    buffer_id = builder.add_buffer(buffer_info)

    assert isinstance(builder, sr.ScenarioBuilder)
    assert isinstance(buffer_id, sr.BufferId)
    assert buffer_id.value == 0

    scenario = builder.build(options=options)
    assert isinstance(scenario, sr.Scenario)


def test_numpy_image_upload_supports_single_mip(sr):
    builder = sr.ScenarioBuilder()
    image_id = builder.add_image(
        [1, 2, 2, 1],
        sr.Format.R8Uint,
        is_input=True,
        is_sampled=True,
        tiling=sr.Tiling.Linear,
    )
    scenario = builder.build()
    image = np.array([1, 2, 3, 4], dtype=np.uint8).reshape(1, 2, 2, 1)

    scenario.upload(image_id, image)
    np.testing.assert_array_equal(scenario.download(image_id), image)

    with pytest.raises(TypeError):
        scenario.upload(image_id, image, mip_levels=2)


def test_scenario_supports_repeated_numpy_transfers(sr, tmp_path):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "commands": [],
                "resources": [
                    {
                        "buffer": {
                            "uid": "buffer",
                            "size": 4,
                            "shader_access": "readwrite",
                        }
                    },
                    {
                        "tensor": {
                            "uid": "tensor",
                            "dims": [1, 2, 2, 1],
                            "format": "VK_FORMAT_R8_SINT",
                            "shader_access": "readwrite",
                        }
                    },
                    {
                        "image": {
                            "uid": "image",
                            "dims": [1, 2, 2, 1],
                            "format": "VK_FORMAT_R8_UINT",
                            "shader_access": "readwrite",
                            "mips": 1,
                        }
                    },
                ],
            }
        )
    )

    scenario = sr.ScenarioJsonFactory.make(scenario_path)
    buffer_id = scenario.get_buffer_id("buffer")
    tensor_id = scenario.get_tensor_id("tensor")
    image_id = scenario.get_image_id("image")

    first = np.array([1, 2, 3, 4], dtype=np.uint8)
    tensor = np.array([1, 2, 3, 4], dtype=np.int8).reshape(1, 2, 2, 1)
    image = np.array([5, 6, 7, 8], dtype=np.uint8).reshape(1, 2, 2, 1)
    scenario.upload(buffer_id, first)
    scenario.upload(tensor_id, tensor)
    scenario.upload(image_id, image)
    scenario.run()
    np.testing.assert_array_equal(scenario.download(buffer_id), first)
    np.testing.assert_array_equal(scenario.download(tensor_id), tensor)
    np.testing.assert_array_equal(scenario.download(image_id), image)

    second = np.array([5, 6, 7, 8], dtype=np.uint8)
    scenario.upload(buffer_id, second)
    scenario.run()
    np.testing.assert_array_equal(scenario.download(buffer_id), second)

    non_contiguous = np.arange(8, dtype=np.uint8)[::2]
    with np.testing.assert_raises_regex(ValueError, "C-contiguous"):
        scenario.upload(buffer_id, non_contiguous)


def test_scenario_json_factory_builds_interface(sr, tmp_path):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "commands": [],
                "resources": [
                    {
                        "buffer": {
                            "uid": "buffer",
                            "size": 4,
                            "shader_access": "readwrite",
                        }
                    }
                ],
            }
        )
    )
    options = sr.ScenarioOptions()

    scenario = sr.ScenarioJsonFactory.make(scenario_path, options=options)

    assert isinstance(scenario, sr.Scenario)


def test_in_memory_scenario_builder_executes_compute(sr, tmp_path, glsl_compiler):
    shader_path = tmp_path / "increment.comp"
    shader_path.write_text("""
        #version 450
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(set = 0, binding = 0) readonly buffer Input { uint values[]; } input_buffer;
        layout(set = 0, binding = 1) writeonly buffer Output { uint values[]; } output_buffer;
        void main() {
            uint index = gl_GlobalInvocationID.x;
            output_buffer.values[index] = input_buffer.values[index] + 1;
        }
        """)
    compiled_shader = tmp_path / "increment.spv"
    glsl_compiler.run("--input", shader_path, "--output", compiled_shader)

    builder = sr.ScenarioBuilder()

    shader_info = sr.ShaderInfo()
    shader_info.debug_name = "increment"
    shader_info.entry = "main"
    shader_info.shader_type = sr.ShaderType.SpirV
    shader_info.stage = sr.ShaderStage.Compute
    shader_info.load_source(str(compiled_shader))
    shader_id = builder.add_shader(shader_info)

    input_info = sr.BufferInfo()
    input_info.debug_name = "input"
    input_info.size = 16
    input_id = builder.add_buffer(input_info)

    output_info = sr.BufferInfo()
    output_info.debug_name = "output"
    output_info.size = 16
    output_id = builder.add_buffer(output_info)

    command = sr.DispatchComputeData(shader_id)
    command.debug_name = "increment"
    command.bindings = [
        sr.TypedBinding(0, 0, input_id, sr.DescriptorType.StorageBuffer),
        sr.TypedBinding(0, 1, output_id, sr.DescriptorType.StorageBuffer),
    ]
    dispatch = sr.ComputeDispatch()
    dispatch.group_count_x = 4
    dispatch.profile_name = command.debug_name
    command.compute_dispatch = dispatch
    builder.add_dispatch_compute(command)

    assert isinstance(builder, sr.ScenarioBuilder)
    scenario = builder.build()
    assert isinstance(scenario, sr.Scenario)

    first_input = np.array([1, 2, 3, 4], dtype=np.uint32)
    scenario.upload(input_id, first_input)
    scenario.run()
    np.testing.assert_array_equal(
        scenario.download(output_id).view(np.uint32), first_input + 1
    )

    second_input = np.array([10, 20, 30, 40], dtype=np.uint32)
    scenario.upload(input_id, second_input)
    scenario.run()
    np.testing.assert_array_equal(
        scenario.download(output_id).view(np.uint32), second_input + 1
    )


def test_in_memory_scenario_builder_aliases_image_and_tensor(sr):
    # Programmatic equivalent of image_from_npy_alias_to_tensor.json.
    width, height, channels = 80, 48, 2
    input_data = np.arange(width * height * channels, dtype=np.float16).reshape(
        1, height, width, channels
    )

    builder = sr.ScenarioBuilder()

    image_info = sr.ImageInfo()
    image_info.debug_name = "input image"
    image_info.shape = [1, width, height, 1]
    image_info.format = sr.Format.R16G16Sfloat
    image_info.is_input = True
    image_info.is_sampled = True
    image_info.mips = 1
    image_id = builder.add_image(image_info)

    tensor_info = sr.TensorInfo()
    tensor_info.debug_name = "output tensor"
    tensor_info.shape = [1, height, width, channels]
    tensor_info.format = sr.Format.R16Uint
    tensor_id = builder.add_tensor(tensor_info)

    group_id = builder.create_memory_group()
    builder.add_resource_to_memory_group(group_id, image_id)
    builder.add_resource_to_memory_group(group_id, tensor_id)

    scenario = builder.build()

    # ImageInfo uses [N, W, H, depth]; pack two float16 channels into one image element.
    scenario.upload(image_id, input_data.view("V4").reshape(1, width, height, 1))
    scenario.run()

    np.testing.assert_array_equal(
        scenario.download(tensor_id).view(np.uint16), input_data.view(np.uint16)
    )


def test_in_memory_scenario_builder_executes_explicit_buffer_barrier(sr, sdk_tools):
    # Programmatic equivalent of buffer_barrier.json with explicit stages.
    compiled_shader = sdk_tools.compile_shader(
        "test_barrier/add_one.comp", output="addOne.spv"
    )

    builder = sr.ScenarioBuilder()

    shader_info = sr.ShaderInfo()
    shader_info.debug_name = "addOne"
    shader_info.entry = "main"
    shader_info.shader_type = sr.ShaderType.SpirV
    shader_info.stage = sr.ShaderStage.Compute
    shader_info.load_source(str(compiled_shader))
    shader_id = builder.add_shader(shader_info)

    input_info = sr.BufferInfo()
    input_info.debug_name = "input"
    input_info.size = 256
    input_id = builder.add_buffer(input_info)

    intermediate_info = sr.BufferInfo()
    intermediate_info.debug_name = "intermediate"
    intermediate_info.size = 256
    intermediate_id = builder.add_buffer(intermediate_info)

    output_info = sr.BufferInfo()
    output_info.debug_name = "output"
    output_info.size = 256
    output_id = builder.add_buffer(output_info)

    barrier_info = sr.BufferBarrierInfo()
    barrier_info.debug_name = "intermediate write-to-read"
    barrier_info.buffer = intermediate_id
    barrier_info.size = 256
    barrier_info.src_access = sr.MemoryAccess.ComputeShaderWrite
    barrier_info.dst_access = sr.MemoryAccess.ComputeShaderRead
    barrier_info.src_stages = [sr.PipelineStage.Compute]
    barrier_info.dst_stages = [sr.PipelineStage.Compute]
    barrier_id = builder.add_buffer_barrier(barrier_info)

    def add_one_dispatch(input_buffer, output_buffer):
        command = sr.DispatchComputeData(shader_id)
        command.bindings = [
            sr.TypedBinding(0, 0, input_buffer, sr.DescriptorType.StorageBuffer),
            sr.TypedBinding(0, 1, output_buffer, sr.DescriptorType.StorageBuffer),
        ]
        command.implicit_barrier = False
        dispatch = sr.ComputeDispatch()
        dispatch.group_count_x = 256
        command.compute_dispatch = dispatch
        builder.add_dispatch_compute(command)

    add_one_dispatch(input_id, intermediate_id)
    barriers = sr.PipelineBarrierData()
    barriers.buffer_barriers = [barrier_id]
    builder.add_pipeline_barrier(barriers)
    add_one_dispatch(intermediate_id, output_id)

    scenario = builder.build()
    input_data = np.full(256, 42, dtype=np.uint8)
    scenario.upload(input_id, input_data)
    scenario.run()

    np.testing.assert_array_equal(scenario.download(output_id), input_data + 2)
