# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import scenario_runner_py as sr

sample_dir = Path(__file__).resolve().parent.parent
scenario = sr.load_scenario(sample_dir / "increment.json", work_dir=sample_dir)
input_id = scenario.get_buffer_id("input")
output_id = scenario.get_buffer_id("output")

input_values = np.array([10, 20, 30, 40], dtype=np.uint32)
scenario.upload(input_id, input_values)
scenario.run()

# Buffer downloads contain bytes; view them using the data type used by the shader.
output_values = scenario.download(output_id).view(np.uint32)
np.testing.assert_array_equal(output_values, input_values + 1)
