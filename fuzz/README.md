# Scenario Runner Library fuzzers

Configure with Clang and `-DSCENARIO_RUNNER_ENABLE_FUZZER=ON`. The
`ScenarioRunnerFuzzerLib` and fuzzer harnesses are instrumented with ASan and
UBSan; the fuzzer executables additionally link libFuzzer.

- `scenario_fuzzer` seeds one scenario with valid compute, graphics, tensor,
  generated VGF, and native SPIR-V™ data-graph resources. It then decodes the
  remaining input as an arbitrary sequence of builder API calls before
  attempting the complete build, upload, run, and download path.

The runtime targets need a Vulkan® implementation with the required features;
configure the SDK emulation layers as appropriate.
Before processing the corpus, the fuzzer runs its seeded scenario once and
terminates with a diagnostic if the runtime environment cannot complete it.
For example:

```sh
mkdir -p corpus
build-fuzzer/fuzz/scenario_fuzzer -max_len=512 -max_total_time=60 corpus
```

Builder calls use fixed-size 16-byte records, up to 32 records per input. The
operation switch covers resource creation, memory groups, shaders, raw data,
VGF, graph constants, all barrier resources, all dispatch types, pipeline
barriers, and frame boundaries. Invalid-ID operations and malformed transfer
checks remain available as low-probability negative tests. Those checks require
the expected exception type and diagnostic. Scenario-construction failures are
tolerated, but exceptions after a scenario has built, failed rejection checks,
and sanitizer failures terminate the fuzzer. Generated files are stored in an
exclusively created temporary directory per process and removed on normal exit.

Buffer, tensor, and image creation can either generate a new description or
duplicate an existing compatible description. Generated VGFs use storage
buffers, storage images, or tensors selected by the fuzz input. Compute shaders
and VGFs retain the resource contract used to create them. Their dispatch
operations select from the original resources and any compatible duplicates,
so mutations can vary resource identity without accidentally changing required
sizes, formats, shapes, or usages. Optical-flow dispatches likewise require
three distinct storage images with matching descriptions.

Native data-graph shaders are assembled from a constant-output SPIR-V™ template.
The input and output shapes are independently selected from rank-4 tensors, and
dispatches can substitute compatible fuzz-generated resources. This keeps the
graph valid without imposing operation-specific shape relationships.

The fuzzer prints `SCENARIO_FUZZER_STATS` every 1024 completed inputs and once
at normal process exit. Reports include operation records, rejected operations,
scenario build results, the number of buffer-, image-, and tensor-backed VGFs
and native data graphs, completed run calls and iterations, and fully completed
end-to-end inputs.
