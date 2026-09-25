#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import io
import json
from types import ModuleType

import numpy as np
import pytest
import vgfpy as vgf

sr: ModuleType


@pytest.fixture(scope="module", autouse=True)
def load_scenario_runner(request: pytest.FixtureRequest) -> None:
    global sr
    sr = request.getfixturevalue("sr")


DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6
VK_FORMAT_R32_UINT = 98
VULKAN_HEADER_VERSION = 123


def _compile_increment_shader(tmp_path, glsl_compiler):
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
    return compiled_shader


def _add_increment_shader(builder, tmp_path, glsl_compiler):
    shader_info = sr.ShaderInfo()
    shader_info.debug_name = "increment"
    shader_info.entry = "main"
    shader_info.shader_type = sr.ShaderType.SpirV
    shader_info.stage = sr.ShaderStage.Compute
    shader_info.load_source(str(_compile_increment_shader(tmp_path, glsl_compiler)))
    return builder.add_shader(shader_info)


def _add_increment_dispatch(builder, shader_id, input_id, output_id, debug_name):
    command = sr.DispatchComputeData(shader_id)
    command.debug_name = debug_name
    command.bindings = [
        sr.TypedBinding(0, 0, input_id, sr.DescriptorType.StorageBuffer),
        sr.TypedBinding(0, 1, output_id, sr.DescriptorType.StorageBuffer),
    ]
    dispatch = sr.ComputeDispatch()
    dispatch.group_count_x = 4
    dispatch.profile_name = debug_name
    command.compute_dispatch = dispatch
    builder.add_dispatch_compute(command)


def _write_increment_vgf(path):
    encoder = vgf.CreateEncoder(VULKAN_HEADER_VERSION)
    module = encoder.AddModule(vgf.ModuleType.Compute, "increment", "main")
    input_resource = encoder.AddInputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R32_UINT, [4], []
    )
    output_resource = encoder.AddOutputResource(
        DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT, VK_FORMAT_R32_UINT, [4], []
    )
    input_binding = encoder.AddBindingSlot(0, input_resource)
    output_binding = encoder.AddBindingSlot(1, output_resource)
    descriptor_set = encoder.AddDescriptorSetInfo([input_binding, output_binding])
    encoder.AddModelSequenceInputsOutputs(
        [input_binding], ["input"], [output_binding], ["output"]
    )
    encoder.AddSegmentInfo(
        module,
        "increment_segment",
        [descriptor_set],
        [input_binding],
        [output_binding],
        [],
        [4, 1, 1],
    )
    encoder.Finish()

    with io.FileIO(path, mode="wb") as stream:
        assert encoder.WriteTo(stream)


def _add_vgf_dispatch(builder, vgf_id, shader_id, input_id, output_id, name):
    command = sr.DispatchVgfData(vgf_id)
    command.debug_name = name
    command.bindings = [
        sr.TypedBinding(0, 0, input_id, sr.DescriptorType.StorageBuffer),
        sr.TypedBinding(0, 1, output_id, sr.DescriptorType.StorageBuffer),
    ]
    command.shader_substitutions = [sr.ShaderSubstitution(shader_id, "increment")]
    builder.add_dispatch_vgf(command)


class TestDeviceResidentChaining:
    """Dispatches exchange and retain resources without Python transfers."""

    # This test verifies that:
    # - A second dispatch consumes the first dispatch's device-resident output.
    # - The final dispatch writes back to the state buffer used by the next run.
    # - State remains on the device across repeated runs after one NumPy upload.
    def test_builder_reuses_device_resident_state_across_runs(
        self, tmp_path, glsl_compiler
    ):
        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        state_id = builder.add_buffer(16, debug_name="state")
        intermediate_id = builder.add_buffer(16, debug_name="intermediate")
        _add_increment_dispatch(
            builder, shader_id, state_id, intermediate_id, "increment_first"
        )
        _add_increment_dispatch(
            builder, shader_id, intermediate_id, state_id, "increment_second"
        )
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
        scenario.upload(state_id, input_values)
        scenario.run(repeat_count=3)

        np.testing.assert_array_equal(
            scenario.download(intermediate_id).view(np.uint32), input_values + 5
        )
        np.testing.assert_array_equal(
            scenario.download(state_id).view(np.uint32), input_values + 6
        )

    # This test verifies that:
    # - A Python-built scenario can contain more than two dependent dispatches.
    # - Each dispatch passes its device-resident output to the next dispatch.
    def test_builder_executes_multi_stage_in_memory_pipeline(
        self, tmp_path, glsl_compiler
    ):
        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        input_id = builder.add_buffer(16, debug_name="input")
        first_output_id = builder.add_buffer(16, debug_name="first_output")
        second_output_id = builder.add_buffer(16, debug_name="second_output")
        final_output_id = builder.add_buffer(16, debug_name="final_output")
        _add_increment_dispatch(
            builder, shader_id, input_id, first_output_id, "increment_first"
        )
        _add_increment_dispatch(
            builder, shader_id, first_output_id, second_output_id, "increment_second"
        )
        _add_increment_dispatch(
            builder, shader_id, second_output_id, final_output_id, "increment_third"
        )
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
        scenario.upload(input_id, input_values)
        scenario.run()

        for increment, resource_id in enumerate(
            (first_output_id, second_output_id, final_output_id), start=1
        ):
            np.testing.assert_array_equal(
                scenario.download(resource_id).view(np.uint32), input_values + increment
            )


class TestPythonMemoryOwnership:
    """Uploads and downloads have explicit ownership across the Python boundary."""

    # This test verifies that:
    # - One Python-built scenario's output can be passed to another scenario without a resource file.
    # - The handoff uses host NumPy memory; each scenario owns its device resources.
    def test_builder_passes_in_memory_output_between_scenarios(
        self, tmp_path, glsl_compiler
    ):
        producer_builder = sr.ScenarioBuilder()
        producer_shader_id = _add_increment_shader(
            producer_builder, tmp_path, glsl_compiler
        )
        producer_input_id = producer_builder.add_buffer(16, debug_name="producer_input")
        producer_intermediate_id = producer_builder.add_buffer(
            16, debug_name="producer_intermediate"
        )
        producer_output_id = producer_builder.add_buffer(
            16, debug_name="producer_output"
        )
        _add_increment_dispatch(
            producer_builder,
            producer_shader_id,
            producer_input_id,
            producer_intermediate_id,
            "producer_increment_first",
        )
        _add_increment_dispatch(
            producer_builder,
            producer_shader_id,
            producer_intermediate_id,
            producer_output_id,
            "producer_increment_second",
        )
        producer = producer_builder.build()

        consumer_builder = sr.ScenarioBuilder()
        consumer_shader_id = _add_increment_shader(
            consumer_builder, tmp_path, glsl_compiler
        )
        consumer_input_id = consumer_builder.add_buffer(16, debug_name="consumer_input")
        consumer_intermediate_id = consumer_builder.add_buffer(
            16, debug_name="consumer_intermediate"
        )
        consumer_output_id = consumer_builder.add_buffer(
            16, debug_name="consumer_output"
        )
        _add_increment_dispatch(
            consumer_builder,
            consumer_shader_id,
            consumer_input_id,
            consumer_intermediate_id,
            "consumer_increment_first",
        )
        _add_increment_dispatch(
            consumer_builder,
            consumer_shader_id,
            consumer_intermediate_id,
            consumer_output_id,
            "consumer_increment_second",
        )
        consumer = consumer_builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)

        producer.upload(producer_input_id, input_values)
        producer.run()
        consumer.upload(
            consumer_input_id, producer.download(producer_output_id).view(np.uint32)
        )
        consumer.run()

        np.testing.assert_array_equal(
            consumer.download(consumer_output_id).view(np.uint32), input_values + 4
        )

    # This test verifies that:
    # - Upload takes ownership of the values rather than borrowing the NumPy allocation.
    # - Python can modify or release its source array before scenario execution.
    def test_upload_copies_numpy_input_before_execution(self, tmp_path, glsl_compiler):
        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        input_id = builder.add_buffer(16, debug_name="input")
        output_id = builder.add_buffer(16, debug_name="output")
        _add_increment_dispatch(builder, shader_id, input_id, output_id, "increment")
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
        expected_output = input_values + 1
        scenario.upload(input_id, input_values)

        input_values.fill(100)
        scenario.run()

        np.testing.assert_array_equal(
            scenario.download(output_id).view(np.uint32), expected_output
        )

    # This test verifies that:
    # - Download returns Python-owned memory rather than a view of device-resident state.
    # - Modifying a downloaded array cannot change subsequent scenario downloads.
    def test_download_returns_independent_numpy_memory(self, tmp_path, glsl_compiler):
        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        input_id = builder.add_buffer(16, debug_name="input")
        output_id = builder.add_buffer(16, debug_name="output")
        _add_increment_dispatch(builder, shader_id, input_id, output_id, "increment")
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
        scenario.upload(input_id, input_values)
        scenario.run()

        first_download = scenario.download(output_id).view(np.uint32)
        first_download.fill(100)

        np.testing.assert_array_equal(
            scenario.download(output_id).view(np.uint32), input_values + 1
        )

    # This test verifies that:
    # - Tensor data can be supplied directly from Python memory.
    # - Upload and download do not borrow or expose the scenario's tensor storage.
    def test_tensor_transfers_use_independent_python_memory(self):
        builder = sr.ScenarioBuilder()
        tensor_id = builder.add_tensor([4], sr.Format.R32Uint, debug_name="state")
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
        expected_values = input_values.copy()
        scenario.upload(tensor_id, input_values)
        input_values.fill(100)
        scenario.run()

        first_download = scenario.download(tensor_id)
        np.testing.assert_array_equal(first_download, expected_values)
        first_download.fill(200)
        np.testing.assert_array_equal(scenario.download(tensor_id), expected_values)

    # This test verifies that:
    # - Image data is copied during upload rather than borrowed from NumPy.
    # - Each download owns its memory and cannot modify the scenario's image.
    def test_image_transfers_use_independent_python_memory(self):
        builder = sr.ScenarioBuilder()
        image_id = builder.add_image(
            [1, 2, 2, 1],
            sr.Format.R8Uint,
            debug_name="image",
            is_input=True,
            is_storage=True,
        )
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint8).reshape(1, 2, 2, 1)
        expected_values = input_values.copy()
        scenario.upload(image_id, input_values)
        input_values.fill(100)
        scenario.run()

        first_download = scenario.download(image_id)
        np.testing.assert_array_equal(first_download, expected_values)
        first_download.fill(200)
        np.testing.assert_array_equal(scenario.download(image_id), expected_values)


class TestTensorDispatch:
    """Tensor resources participate directly in Python-built dispatches."""

    # This test verifies that:
    # - Python-created tensor IDs can be bound to a compute dispatch.
    # - Tensor input and output remain in memory throughout execution.
    def test_builder_executes_compute_with_tensor_bindings(
        self, tmp_path, glsl_compiler
    ):
        shader_path = tmp_path / "increment_tensor.comp"
        shader_path.write_text("""
            #version 450
            #extension GL_ARM_tensors : require
            #extension GL_EXT_shader_explicit_arithmetic_types : require

            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            layout(set = 0, binding = 0) readonly uniform tensorARM<int8_t, 4> input_tensor;
            layout(set = 0, binding = 1) writeonly uniform tensorARM<int8_t, 4> output_tensor;

            void main() {
                uint coordinates[4] = uint[](0, gl_GlobalInvocationID.x, 0, 0);
                int8_t value;
                tensorReadARM(input_tensor, coordinates, value);
                tensorWriteARM(output_tensor, coordinates, value + int8_t(1));
            }
            """)
        compiled_shader = tmp_path / "increment_tensor.spv"
        glsl_compiler.run("--input", shader_path, "--output", compiled_shader)

        builder = sr.ScenarioBuilder()
        shader_info = sr.ShaderInfo()
        shader_info.debug_name = "increment tensor"
        shader_info.entry = "main"
        shader_info.shader_type = sr.ShaderType.SpirV
        shader_info.stage = sr.ShaderStage.Compute
        shader_info.load_source(str(compiled_shader))
        shader_id = builder.add_shader(shader_info)
        input_id = builder.add_tensor(
            [1, 4, 1, 1], sr.Format.R8Sint, debug_name="input tensor"
        )
        output_id = builder.add_tensor(
            [1, 4, 1, 1], sr.Format.R8Sint, debug_name="output tensor"
        )

        command = sr.DispatchComputeData(shader_id)
        command.debug_name = "increment tensor"
        command.bindings = [
            sr.TypedBinding(0, 0, input_id, sr.DescriptorType.TensorArm),
            sr.TypedBinding(0, 1, output_id, sr.DescriptorType.TensorArm),
        ]
        dispatch = sr.ComputeDispatch()
        dispatch.group_count_x = 4
        command.compute_dispatch = dispatch
        builder.add_dispatch_compute(command)
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.int8).reshape(1, 4, 1, 1)
        scenario.upload(input_id, input_values)
        scenario.run()

        np.testing.assert_array_equal(scenario.download(output_id), input_values + 1)


class TestVgfDeviceResidentChaining:
    """VGF external bindings preserve resources between graph dispatches."""

    # This test verifies that:
    # - VGF external bindings can refer to resources created through the Python builder.
    # - The first VGF dispatch's output remains on the device for the second dispatch.
    # - Python transfers only the pipeline input and final output.
    @pytest.mark.vgf_graph
    def test_builder_chains_vgf_dispatches_in_device_memory(
        self, tmp_path, glsl_compiler
    ):
        vgf_path = tmp_path / "increment.vgf"
        _write_increment_vgf(vgf_path)

        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        vgf_info = sr.VgfInfo()
        vgf_info.debug_name = "increment_graph"
        vgf_info.load_source(str(vgf_path))
        vgf_id = builder.add_vgf(vgf_info)
        input_id = builder.add_buffer(16, debug_name="input")
        intermediate_id = builder.add_buffer(16, debug_name="intermediate")
        output_id = builder.add_buffer(16, debug_name="output")
        _add_vgf_dispatch(
            builder,
            vgf_id,
            shader_id,
            input_id,
            intermediate_id,
            "increment_first",
        )
        _add_vgf_dispatch(
            builder,
            vgf_id,
            shader_id,
            intermediate_id,
            output_id,
            "increment_second",
        )
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
        scenario.upload(input_id, input_values)
        scenario.run()

        np.testing.assert_array_equal(
            scenario.download(output_id).view(np.uint32), input_values + 2
        )

    # This test verifies that:
    # - VGF output can become the next run's input without a Python transfer.
    # - Repeated runs update the same device-resident state after one upload.
    @pytest.mark.vgf_graph
    def test_vgf_reuses_device_resident_state_across_runs(
        self, tmp_path, glsl_compiler
    ):
        vgf_path = tmp_path / "increment.vgf"
        _write_increment_vgf(vgf_path)

        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        vgf_info = sr.VgfInfo()
        vgf_info.debug_name = "increment_graph"
        vgf_info.load_source(str(vgf_path))
        vgf_id = builder.add_vgf(vgf_info)
        state_id = builder.add_buffer(16, debug_name="state")
        intermediate_id = builder.add_buffer(16, debug_name="intermediate")
        _add_vgf_dispatch(
            builder,
            vgf_id,
            shader_id,
            state_id,
            intermediate_id,
            "increment_first",
        )
        _add_vgf_dispatch(
            builder,
            vgf_id,
            shader_id,
            intermediate_id,
            state_id,
            "increment_second",
        )
        scenario = builder.build()

        input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
        scenario.upload(state_id, input_values)
        scenario.run(repeat_count=3)

        np.testing.assert_array_equal(
            scenario.download(state_id).view(np.uint32), input_values + 6
        )


class TestInMemoryProfiling:
    """Python-built scenarios emit profiling for their in-memory dispatches."""

    # This test verifies that:
    # - A Python-built in-memory scenario writes a profiling data file.
    # - The dump contains timestamp records for each Python-defined dispatch.
    def test_builder_dumps_profiling_for_in_memory_dispatches(
        self, tmp_path, glsl_compiler
    ):
        profiling_path = tmp_path / "profiling.json"
        options = sr.ScenarioOptions()
        options.profiling_path = profiling_path

        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        input_id = builder.add_buffer(16, debug_name="input")
        intermediate_id = builder.add_buffer(16, debug_name="intermediate")
        output_id = builder.add_buffer(16, debug_name="output")
        _add_increment_dispatch(
            builder, shader_id, input_id, intermediate_id, "increment_first"
        )
        _add_increment_dispatch(
            builder, shader_id, intermediate_id, output_id, "increment_second"
        )
        scenario = builder.build(options=options)

        # Builder-created scenarios must emit the same profiling data as JSON scenarios.
        scenario.upload(input_id, np.array([1, 2, 3, 4], dtype=np.uint32))
        scenario.run(repeat_count=3)

        profiling_data = json.loads(profiling_path.read_text())
        timestamps = profiling_data["Timestamps"]
        assert [timestamp["Command type"] for timestamp in timestamps] == [
            "ComputeDispatch"
        ] * 6
        assert [timestamp["Command name"] for timestamp in timestamps] == [
            "increment_first",
            "increment_second",
        ] * 3
        assert [timestamp["Iteration"] for timestamp in timestamps] == [
            1,
            1,
            2,
            2,
            3,
            3,
        ]

    # This test verifies that:
    # - Profiling includes dispatches created through VGF external bindings.
    # - VGF profiling is available without a JSON scenario or resource data files.
    @pytest.mark.vgf_graph
    def test_builder_dumps_profiling_for_vgf_dispatch(self, tmp_path, glsl_compiler):
        profiling_path = tmp_path / "vgf_profiling.json"
        options = sr.ScenarioOptions()
        options.profiling_path = profiling_path
        vgf_path = tmp_path / "increment.vgf"
        _write_increment_vgf(vgf_path)

        builder = sr.ScenarioBuilder()
        shader_id = _add_increment_shader(builder, tmp_path, glsl_compiler)
        vgf_info = sr.VgfInfo()
        vgf_info.debug_name = "increment_graph"
        vgf_info.load_source(str(vgf_path))
        vgf_id = builder.add_vgf(vgf_info)
        input_id = builder.add_buffer(16, debug_name="input")
        output_id = builder.add_buffer(16, debug_name="output")
        _add_vgf_dispatch(
            builder, vgf_id, shader_id, input_id, output_id, "vgf_increment"
        )
        scenario = builder.build(options=options)

        scenario.upload(input_id, np.array([1, 2, 3, 4], dtype=np.uint32))
        scenario.run()

        timestamps = json.loads(profiling_path.read_text())["Timestamps"]
        assert len(timestamps) == 1
        assert timestamps[0]["Command type"] == "ComputeDispatch"
        assert timestamps[0]["Command name"] == "vgf_increment/increment_segment"
        assert timestamps[0]["Iteration"] == 1
