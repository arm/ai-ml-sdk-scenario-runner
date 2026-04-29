# Scenario Runner — Release Notes

---

## Version 0.9.0 – *Datatype, Dispatch & DXC Support*

### Highlights

- Added BF16 and FP8 (`fp8e4m3`, `fp8e5m2`) datatype support.
- Added SPIR-V™ only data graph dispatch support for shader-only workloads.
- Added DXC API integration and HLSL shader support. Note: This feature is
  experimental there is no tensorARM support in HLSL shaders and
  DirectXShaderCompiler is linked dynamically as it does not support static
  linking.
- Updated profiling/runtime behavior with data graph memory tracking,
  pipeline-cache miss enforcement, and fixes for `--repeat` memory leaks.
- Added `implicit_barrier` control for data graph dispatch and updated default
  log-level behavior.

### Platform & Compliance

- Expanded DXGI coverage, including core format placement and additional format
  support for Windows® parity.
- Added PNG load/save support and broader image/texture format coverage.

### Bug Fixes

- Fixed memory leaks in the `--repeat` option.
- Fixed frame-boundary extension detection and default log-level behavior.
- Fixed Linux out-of-bounds failures.

## Version 0.8.0 – *Format Coverage & Tooling Upgrades*

### Execution & Scenario Control

- Enabled `storagePushConstant16` when the device advertises it so shader
  workloads can consume 16-bit push constants without custom tweaks.
- Added `--pause-on-exit` to the runner CLI for easier debugging on interactive
  sessions.
- Experimental:
  - Expanded Windows® parity by supporting nearly the entire DXGI format catalog,
    ensuring shaders and tensor IO behave consistently across platforms.

### Build, Packaging & Developer Experience

- Modernized the pip package: switched to `pyproject.toml`, added the missing
  metadata, and fixed package naming/installation ordering issues that affected
  `--install`.
- Defaulted the build system to Ninja, refined the CMake packaging flow.
- Introduced `clang-tidy` configuration and streamlined cppcheck
  invocation/CLI integration (including build-script driven execution).

### Platform & Compliance

- Added Darwin targets for AArch64 to the pip packaging matrix.
- Refreshed SBOM data and adopted usage of `REUSE.toml`.

### Bug Fixes

- Fixed missing package version initialization, bad package names, and
  SDK-root `--install` failures in packaging flows.
- Fixed MSVC AddressSanitizer support in the build and test flow.

### Supported Platforms

The following platform combinations are supported:

- Linux - AArch64 and x86-64
- Windows® - x86-64
- Darwin - AArch64 via MoltenVK (experimental)
- Android™ - AArch64 (experimental)

---

## Version 0.7.0 – *Initial Public Release*

## Purpose

Executes ML workloads for **functional validation** and **performance
exploration**.

## Features

### Execution

- **Workload Execution**: Runs shader and neural network graph workloads through
  both Vulkan® core compute and Vulkan® ML extensions for comprehensive ML
  workload testing
- **Complete Workflow**: Loads input stimuli, executes computational graphs, and
  writes output artifacts - providing end-to-end scenario execution
- **Validation & Benchmarking**: Serves as a validation vehicle for drivers and
  devices, plus performance exploration and benchmarking capabilities

### Test description

- **Declarative Scenarios**: Test cases and scenarios defined in JSON format for
  easy configuration, repeatability, and programmatic test generation
- **Several types of resources**: Supports tensors, images, buffers, etc. so
  scenarios can exercise varied input/output.

## Platform Support

The following platform combinations are supported:

- Linux - X86-64
- Windows® - X86-64

---
