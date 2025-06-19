# ML SDK Scenario Runner

The Scenario Runner is an application that executes shader and neural network
graph workloads through the Vulkan® or Arm® Vulkan® ML extensions. The
Scenario Runner acts as a validation and performance exploration vehicle. The
Scenario Runner also acts as a mechanism to define test-cases called scenarios
in a declarative way via a JSON description. The Scenario Runner can parse the
JSON, load the input stimulus that is described in the JSON, execute the
scenario and produce output artefacts.

## Building Scenario Runner from source

The build system must have:

- CMake 3.25 or later.
- C/C++ 17 compiler: GCC, or optionally Clang on Linux and MSVC on Windows®.
- Python 3.10 or later. Required python libraries for building are listed in
  `tooling-requirements.txt`.
- Flatbuffers flatc compiler.

The following dependencies are also needed:

- [Argument Parser for Modern C++](https://github.com/p-ranav/argparse).
- [glslang](https://github.com/KhronosGroup/glslang).
- [JSON for Modern C++](https://github.com/nlohmann/json).
- [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers).
- [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools).
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers).
- [GoogleTest](https://github.com/google/googletest). Optional, for testing.

For the preferred dependency versions see the manifest file.

### Providing Flatc

There are 3 options for providing the flatc binary.

1.  Installing flatc to the system. To install flatc to the system and make it
    available on the searchable `PATH`, see the
    [flatbuffers documentation](https://flatbuffers.dev/).

2.  Building flatc from repo source. When the repository is initialized using
    the repo manifest, you can find a flatbuffers source checkout in
    `<repo-root>/dependencies/flatbuffers/`. Navigate to this location and run
    the following commands:

```bash
cmake -B build && cmake --build build --parallel $(nproc)
```

```{note}
If you build into the `build` folder, the VGF-lib cmake scripts
automatically find the flatc binary.
```

3.  Providing a custom flatc path. If flatc cannot be searched using `PATH` or
    found in the default `<repo-root>/dependencies/flatbuffers/build` path, you
    can provide a custom binary file path to the build script using the
    `--flatc-path <path_to_flatc>` option, see
    [Building with the script](#building-with-the-script).

<a name="building-with-the-script"></a>

### Building with the script

To build the script on your current platform, for example, Linux or Windows®,
run the following command:

```bash
python3 $SDK_PATH/sw/scenario-runner/scripts/build.py -j $(nproc) \
    --flatbuffers-path ${PATH_TO_FLATBUFFERS_CHECKOUT} \
    --argparse-path ${PATH_TO_ARGPARSE_CHECKOUT} \
    --json-path ${PATH_TO_JSON_CHECKOUT} \
    --vulkan-headers-path ${PATH_TO_VULKAN_HEADERS_CHECKOUT} \
    --glslang-path ${PATH_TO_GLSLANG_CHECKOUT} \
    --spirv-headers-path ${PATH_TO_SPIRV_HEADERS_CHECKOUT} \
    --spirv-tools-path ${PATH_TO_SPIRV_TOOLS_CHECKOUT} \
    --vgf-lib-path ${PATH_TO_VGF_LIB_CHECKOUT} \
    --gtest-path ${PATH_TO_GOOGLETEST_CHECKOUT}
```

To cross compile for AArch64 architecture, you can add the following option:

```bash
python3 $SDK_PATH/sw/scenario-runner/scripts/build.py -j $(nproc) \
    --flatbuffers-path ${PATH_TO_FLATBUFFERS_CHECKOUT} \
    --argparse-path ${PATH_TO_ARGPARSE_CHECKOUT} \
    --json-path ${PATH_TO_JSON_CHECKOUT} \
    --vulkan-headers-path ${PATH_TO_VULKAN_HEADERS_CHECKOUT} \
    --glslang-path ${PATH_TO_GLSLANG_CHECKOUT} \
    --spirv-headers-path ${PATH_TO_SPIRV_HEADERS_CHECKOUT} \
    --spirv-tools-path ${PATH_TO_SPIRV_TOOLS_CHECKOUT} \
    --vgf-lib-path ${PATH_TO_VGF_LIB_CHECKOUT} \
    --gtest-path ${PATH_TO_GOOGLETEST_CHECKOUT} \
    --target-platform aarch64
```

To enable and run tests, use the `--test` flag. To lint the tests, use the
`--lint` flag. To build the documentation, use the `--doc` flag. To build the
documentation, you must have `sphinx` and `doxygen` installed on your machine.

You can install the build artifacts for this project into a specified location.
To install the build artifacts, pass the `--install` option with the required
path.

To create an archive with the build artifacts, you must add `--package`. The
archive is stored in the provided location.

For more command line options, consult the program help:

```bash
python3 $SDK_PATH/sw/scenario-runner/scripts/build.py --help
```

### Usage

To run a scenario file, use the following command:

```bash
./build/scenario-runner --scenario ${SCENARIO_JSON_FILE}
```

For more details, see the help output:

```bash
./build/scenario-runner --help
```

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)

## Security

If you have found a security issue, see [Security Section](SECURITY.md)

## Trademark notice

Arm® is a registered trademarks of Arm Limited (or its subsidiaries) in the US
and/or elsewhere.

Khronos®, Vulkan® and SPIR-V™ are registered trademarks of the
[Khronos® Group](https://www.khronos.org/legal/trademarks).
