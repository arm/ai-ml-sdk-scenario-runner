#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np


def test_image_can_load_from_npy(sdk_tools, numpy_helper):
    width, height = 80, 48
    channels = 2

    input_data = numpy_helper.generate(
        [1, height, width, channels],
        dtype=np.float16,
        filename="input.npy",
        data=list(range(width * height * channels)),
    )

    sdk_tools.run_scenario("test_image_from_npy/image_from_npy.json")

    output_dds = sdk_tools.resources_helper.get_testenv_path("output.dds")
    output_npy_path = sdk_tools.convert_dds_to_npy(output_dds, "output.dds.npy", 2)
    output_data = numpy_helper.load(output_npy_path, np.uint16)

    assert output_data.shape == input_data.shape
    assert output_data.tobytes() == input_data.view(np.uint16).tobytes()


def test_image_npy_can_alias_tensor_output(sdk_tools, numpy_helper):
    width, height = 80, 48
    channels = 2

    input_data = numpy_helper.generate(
        [1, height, width, channels],
        dtype=np.float16,
        filename="input.npy",
        data=list(range(width * height * channels)),
    )

    sdk_tools.run_scenario("test_image_from_npy/image_from_npy_alias_to_tensor.json")

    output_data = numpy_helper.load("output.npy", np.uint16)

    assert output_data.shape == input_data.shape
    assert output_data.tobytes() == input_data.view(np.uint16).tobytes()
