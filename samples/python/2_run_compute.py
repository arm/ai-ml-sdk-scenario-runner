# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import scenario_runner_py as sr

sample_dir = Path(__file__).resolve().parent.parent
builder = sr.ScenarioBuilder()

shader = sr.ShaderInfo()
shader.debug_name = "increment"
shader.entry = "main"
shader.shader_type = sr.ShaderType.Glsl
shader.stage = sr.ShaderStage.Compute
shader.load_source(str(sample_dir / "increment.comp"))
shader_id = builder.add_shader(shader)

input_id = builder.add_buffer(16, debug_name="input")
output_id = builder.add_buffer(16, debug_name="output")

dispatch = sr.DispatchComputeData(shader_id)
dispatch.debug_name = "increment"
dispatch.bindings = [
    sr.TypedBinding(0, 0, input_id, sr.DescriptorType.StorageBuffer),
    sr.TypedBinding(0, 1, output_id, sr.DescriptorType.StorageBuffer),
]
dispatch.compute_dispatch.group_count_x = 4
dispatch.compute_dispatch.profile_name = dispatch.debug_name
builder.add_dispatch_compute(dispatch)

scenario = builder.build()
input_values = np.array([1, 2, 3, 4], dtype=np.uint32)
scenario.upload(input_id, input_values)
scenario.run()

output_values = scenario.download(output_id).view(np.uint32)
np.testing.assert_array_equal(output_values, input_values + 1)
