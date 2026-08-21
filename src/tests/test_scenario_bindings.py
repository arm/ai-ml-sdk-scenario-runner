#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

sr = pytest.importorskip("scenario_runner_py")


def test_python_interfaces_and_builder_types():
    builder = sr.ScenarioBuilder()
    options = sr.ScenarioOptions()

    buffer_info = sr.BufferInfo()
    buffer_info.debug_name = "buffer"
    buffer_info.size = 4
    buffer_id = builder.add_buffer(buffer_info)

    assert isinstance(builder, sr.IScenarioBuilder)
    assert isinstance(buffer_id, sr.BufferId)
    assert buffer_id.value == 0

    scenario = builder.build(options=options)
    assert isinstance(scenario, sr.IScenario)


def test_numpy_image_upload_supports_single_mip():
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


def test_in_memory_scenario_builder_executes_compute(tmp_path, glsl_compiler):
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
    shader_info.src = str(compiled_shader)
    shader_info.shader_type = sr.ShaderType.SpirV
    shader_info.stage = sr.ShaderStage.Compute
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

    assert isinstance(builder, sr.IScenarioBuilder)
    scenario = builder.build()
    assert isinstance(scenario, sr.IScenario)

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
