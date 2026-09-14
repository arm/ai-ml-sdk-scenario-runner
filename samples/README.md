<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# ML SDK Scenario Runner API samples

These examples are small, complete programs for applications that use the
ML SDK Scenario Runner C++ API rather than the command-line executable.

1. `1_in_memory_buffer.cpp` creates a scenario containing one buffer and
   demonstrates direct upload and download.
2. `2_run_compute.cpp` defines a compute scenario, compiles a GLSL shader at
   runtime, and validates its output.
3. `3_run_vgf_inference.cpp` builds a small VGF data graph, supplies its
   shader at runtime, and runs it with in-memory input, output, and profiling.
4. `4_run_json.cpp` loads the equivalent workload from JSON, resolves resource
   IDs from JSON UIDs, and transfers data through the resulting scenario.

Build all samples with the regular ML SDK Scenario Runner build:

```sh
cmake --build build
```

Run the samples executable:

```sh
./build/samples/scenario_runner_samples
```

The samples are built by default but are not registered as CTest tests. They
need a Vulkan® device or software Vulkan® driver at runtime.
