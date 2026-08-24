#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

# Validate JSON scenarios through the same in-memory API used by Python-built
# scenarios. These tests reuse scenarios from the standalone test suite.
# Together with the programmatic builder tests in test_scenario_bindings.py,
# they verify that both construction paths preserve the same runtime behavior
# for chained dispatches, aliased resources, explicit barriers, and repeated
# in-memory runs.


def build_json_scenario(
    sr, resources_helper, monkeypatch, scenario_name, replacements=None
):
    scenario_path = resources_helper.prepare_scenario(scenario_name, replacements)
    monkeypatch.chdir(resources_helper.get_testenv_path())
    return sr.ScenarioJsonFactory.make(scenario_path)


def test_json_scenario_executes_chained_dispatches_in_memory(
    sr, sdk_tools, resources_helper, numpy_helper, monkeypatch
):
    first_input = numpy_helper.generate(
        [10], dtype=np.float32, filename="inBufferA.npy"
    )
    second_input = numpy_helper.generate(
        [10], dtype=np.float32, filename="inBufferB.npy"
    )
    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": "float"})
    sdk_tools.run_scenario("test_shader/chained_shaders.json")
    standalone_output = numpy_helper.load("outBufferAdd2.npy").view(np.uint8).copy()

    scenario = build_json_scenario(
        sr, resources_helper, monkeypatch, "test_shader/chained_shaders.json"
    )
    input_a_id = scenario.get_buffer_id("inBufferA")
    input_b_id = scenario.get_buffer_id("inBufferB")
    output_id = scenario.get_buffer_id("outBufferAdd2")

    scenario.upload(input_a_id, first_input)
    scenario.upload(input_b_id, second_input)
    scenario.run()
    np.testing.assert_array_equal(
        scenario.download(output_id),
        standalone_output,
    )

    updated_input = np.full(10, 4.0, dtype=np.float32)
    scenario.upload(input_a_id, updated_input)
    scenario.run()
    np.testing.assert_array_equal(
        scenario.download(output_id).view(np.float32),
        updated_input + second_input + second_input,
    )


@pytest.mark.parametrize(
    "numpy_type, shader_type",
    [
        (np.float32, "float"),
        (np.int8, "int8_t"),
        (np.int32, "uint"),
    ],
)
def test_json_scenario_matches_standalone_for_shader_data_types(
    sr,
    sdk_tools,
    resources_helper,
    numpy_helper,
    monkeypatch,
    numpy_type,
    shader_type,
):
    first_input = numpy_helper.generate(
        [10], dtype=numpy_type, filename="inBufferA.npy"
    )
    second_input = numpy_helper.generate(
        [10], dtype=numpy_type, filename="inBufferB.npy"
    )
    replacements = {"{DATA_SIZE}": str(first_input.nbytes)}
    sdk_tools.compile_shader("test_shader/add_shader.comp", {"TestType": shader_type})
    sdk_tools.run_scenario("test_shader/add_shader.json", replacements)
    standalone_output = numpy_helper.load("outBufferAdd.npy", numpy_type).copy()

    scenario = build_json_scenario(
        sr,
        resources_helper,
        monkeypatch,
        "test_shader/add_shader.json",
        replacements,
    )
    scenario.upload(scenario.get_buffer_id("inBufferA"), first_input)
    scenario.upload(scenario.get_buffer_id("inBufferB"), second_input)
    scenario.run()

    python_output = scenario.download(scenario.get_buffer_id("outBufferAdd"))
    np.testing.assert_array_equal(python_output.view(numpy_type), standalone_output)


def test_json_scenario_preserves_image_tensor_aliasing_in_memory(
    sr, sdk_tools, resources_helper, numpy_helper, monkeypatch
):
    width, height, channels = 80, 48, 2
    input_data = numpy_helper.generate(
        [1, height, width, channels],
        dtype=np.float16,
        filename="input.npy",
        data=list(range(width * height * channels)),
    )
    sdk_tools.run_scenario("test_image_from_npy/image_from_npy_alias_to_tensor.json")
    standalone_output = numpy_helper.load("output.npy").copy()

    scenario = build_json_scenario(
        sr,
        resources_helper,
        monkeypatch,
        "test_image_from_npy/image_from_npy_alias_to_tensor.json",
    )
    input_id = scenario.get_image_id("inputImage")
    output_id = scenario.get_tensor_id("outputTensor")

    # ImageInfo uses [N, W, H, depth]; pack the two float16 channels into one image element.
    image_data = input_data.view("V4").reshape(1, width, height, 1)
    scenario.upload(input_id, image_data)
    scenario.run()

    output_data = scenario.download(output_id)
    np.testing.assert_array_equal(output_data, standalone_output)

    updated_input = np.full_like(input_data, 3.0)
    updated_image = updated_input.view("V4").reshape(1, width, height, 1)
    scenario.upload(input_id, updated_image)
    scenario.run()

    updated_output = scenario.download(output_id)
    np.testing.assert_array_equal(
        updated_output.view(np.uint16), updated_input.view(np.uint16)
    )


def test_json_scenario_executes_explicit_buffer_barrier_in_memory(
    sr, sdk_tools, resources_helper, numpy_helper, monkeypatch
):
    input_data = numpy_helper.generate(
        [256], dtype=np.uint8, filename="input.npy", data=[42] * 256
    )
    sdk_tools.compile_shader("test_barrier/add_one.comp", output="addOne.spv")
    barrier_dispatch = """
{
    "dispatch_barrier": {
        "image_barrier_refs": [],
        "memory_barrier_refs": [],
        "buffer_barrier_refs": ["bufferBarrier"]
    }
},
""".strip()
    stages = """
    "src_stage": ["compute"],
    "dst_stage": ["compute"],
""".strip()
    replacements = {
        "{IMPL_BARRIER}": '"implicit_barrier": false,',
        "{BARRIER_DISPATCH}": barrier_dispatch,
        "{STAGES}": stages,
    }
    sdk_tools.run_scenario("test_barrier/buffer_barrier.json", replacements)
    standalone_output = numpy_helper.load("output.npy").copy()

    scenario = build_json_scenario(
        sr,
        resources_helper,
        monkeypatch,
        "test_barrier/buffer_barrier.json",
        replacements,
    )
    input_id = scenario.get_buffer_id("inputBuffer")
    output_id = scenario.get_buffer_id("outputBuffer")

    scenario.upload(input_id, input_data)
    scenario.run()
    np.testing.assert_array_equal(scenario.download(output_id), standalone_output)

    updated_input = np.full(256, 7, dtype=np.uint8)
    scenario.upload(input_id, updated_input)
    scenario.run()
    np.testing.assert_array_equal(scenario.download(output_id), updated_input + 2)
