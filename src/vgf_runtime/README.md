# VGF Runtime

`vgf_runtime` provides a public C++ API for decoding VGF content and executing VGF graph segments.

## Public API

The installed public header is:

```cpp
#include <vgf_runtime/runtime.hpp>
```

That header exposes both `mlsdk::vgf_runtime::VGF` and `mlsdk::vgf_runtime::Session`.

## Build-tree usage

Enable the runtime when configuring Scenario Runner:

```sh
cmake -S . -B build \
  -DSCENARIO_RUNNER_EXPERIMENTAL_VGF_RUNTIME=ON
```

Link one of the build-tree targets:

```cmake
target_link_libraries(my_target PRIVATE vgf_runtime)
```

or, if you prefer a namespaced target in the same build:

```cmake
target_link_libraries(my_target PRIVATE ScenarioRunner::vgf_runtime)
```

## Installed package usage

When Scenario Runner is installed, the package exports `vgf_runtime` through the
`ScenarioRunner` package. Downstream consumers should use
`ScenarioRunner::vgf_runtime`.

```cmake
find_package(ScenarioRunner CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE ScenarioRunner::vgf_runtime)
```

The package target expects the `VGF` and `VulkanHeaders` packages to be discoverable at configure time.

## Consumption model

- `vgf_runtime` is install-tree consumable today via `find_package(ScenarioRunner)`.
- `vgf_runtime` depends on external packages (`VGF`, `VulkanHeaders`), and those
  dependencies are resolved via the ScenarioRunner package config.

## Current constraints

- `vgf_runtime` is only built when `SCENARIO_RUNNER_EXPERIMENTAL_VGF_RUNTIME=ON`.
- The target is currently added only on Linux, matching the existing gate in `src/CMakeLists.txt`.
