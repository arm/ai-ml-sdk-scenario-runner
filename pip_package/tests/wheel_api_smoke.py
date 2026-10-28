#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Exercise the installed Python API without creating a Vulkan device.

Run with: python -I pip_package/tests/wheel_api_smoke.py
"""

import scenario_runner as sr


def check_resources():
    builder = sr.ScenarioBuilder()
    info = sr.BufferInfo()
    info.debug_name = "wheel-smoke-buffer"
    info.size = 16
    buffer_id = builder.add_buffer(info)

    assert isinstance(buffer_id, sr.BufferId)
    assert buffer_id.value == 0
    assert info.debug_name == "wheel-smoke-buffer"
    assert info.size == 16

    second_buffer_id = builder.add_buffer(32, debug_name="second-buffer")
    assert isinstance(second_buffer_id, sr.BufferId)
    assert second_buffer_id.value == buffer_id.value + 1

    image_info = sr.ImageInfo()
    image_info.debug_name = "image"
    image_info.shape = [1, 2, 2, 1]
    image_info.format = sr.Format.R8Uint
    image_info.is_input = True
    image_id = builder.add_image(image_info)
    assert isinstance(image_id, sr.ImageId)
    assert image_info.shape == [1, 2, 2, 1]
    assert image_info.format == sr.Format.R8Uint
    assert image_info.is_input

    tensor_id = builder.add_tensor([1, 2, 2, 1], sr.Format.R8Uint)
    assert isinstance(tensor_id, sr.TensorId)
    group_id = builder.create_memory_group()
    builder.add_resource_to_memory_group(group_id, image_id)
    builder.add_resource_to_memory_group(group_id, tensor_id)


def check_binary_dispatch():
    builder = sr.ScenarioBuilder()
    info = sr.RawDataInfo()
    info.debug_name = "binary"
    # Registration must not need a real file or a Vulkan device.
    info.src = "wheel-smoke.dgb"
    binary_id = builder.add_raw_data(info)
    assert isinstance(binary_id, sr.RawDataId)
    assert binary_id.value == 0

    dispatch = sr.DispatchDataGraphBinaryData(binary_id)
    assert isinstance(dispatch, sr.DispatchDataGraphBinaryData)
    builder.add_dispatch_data_graph_binary(dispatch)


def check_invalid_resource_type():
    builder = sr.ScenarioBuilder()
    try:
        builder.add_buffer("invalid-buffer-size")
    except TypeError:
        pass
    else:
        raise AssertionError("add_buffer accepted a string as a buffer size")


def main():
    check_resources()
    check_binary_dispatch()
    check_invalid_resource_type()


if __name__ == "__main__":
    main()
