#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


logger = logging.getLogger(__name__)


class NumpyHelper:
    """Helper class to work with NumPy files."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def generate(
        self,
        shape: list[int],
        dtype: np.dtype | None = None,
        filename: str | None = None,
        data: list | None = None,
    ) -> np.ndarray:
        """Generate a random ndarray and save it into a file if needed."""
        if data:
            arr = np.array(data).astype(dtype)
            arr.shape = shape
        elif np.issubdtype(dtype, np.floating):
            arr = np.random.uniform(-100, 100, shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
            arr = np.random.randint(min_val, max_val, size=shape, dtype=dtype)
        else:
            raise ValueError(f"Unsupported data type {dtype}")

        if filename:
            self.save(arr, filename)

        return arr

    def load(
        self, filename: str | None = None, dtype: np.dtype | None = None
    ) -> np.ndarray:
        """Load ndarray from a file."""
        arr = np.load(self.path / filename)

        return arr.view(dtype) if dtype else arr

    def save(self, arr: np.ndarray, filename: str):
        """Save ndarray to a file."""
        np.save(self.path / filename, arr)


class ResourcesHelper:
    """Helper class to work with test resources."""

    def __init__(self, resources_path: Path, test_runtime_path: Path) -> None:
        self.resources_path = resources_path
        self.test_runtime_path = test_runtime_path

    def get_testenv_path(self, name: str | None = None) -> Path:
        """Get path to the test runtime directory."""
        if not name:
            return self.test_runtime_path

        return self.test_runtime_path / name

    def get_shader_path(self, name: str) -> Path:
        """Returns path to the shader resource file."""
        return self.resources_path / "shaders" / name

    def get_spvasm_path(self, name: str) -> Path:
        """Returns path to the spvasm resource file."""
        return self.resources_path / "spvasm" / name

    def get_scenario_path(self, name: str) -> Path:
        """Returns path to the scenario resource file."""
        return self.resources_path / "scenarios" / name

    def prepare_scenario(
        self, name: str, replacements: dict[str, str] | None = None
    ) -> Path:
        """Find the scenario and prepare it for execution."""
        scenario_res_path = self.get_scenario_path(name)

        if not scenario_res_path.is_file():
            raise ValueError(f"Cannot find scenario {name}")

        scenario_text = self._replace(scenario_res_path.read_text(), replacements)

        scenario_path = self.test_runtime_path / scenario_res_path.name
        scenario_path.write_text(scenario_text)

        return scenario_path

    def prepare_shader(
        self,
        name: str,
        replacements: dict[str, str] | None = None,
        output: str | None = None,
    ) -> Path:
        """Find the shader and prepare it for execution."""
        shader_res_path = self.get_shader_path(name)

        if not shader_res_path.is_file():
            raise ValueError(f"Cannot find shader {name}")

        shader_text = self._replace(shader_res_path.read_text(), replacements)

        shader_path = self.test_runtime_path / (output or shader_res_path.name)
        shader_path.write_text(shader_text)

        return shader_path

    def prepare_spvasm(
        self,
        name: str,
        replacements: dict[str, str] | None = None,
        output: str | None = None,
    ) -> Path:
        """Find the spvasm and prepare it for execution."""
        spvasm_res_path = self.get_spvasm_path(name)

        if not spvasm_res_path.is_file():
            raise ValueError(f"Cannot find spvasm {name}")

        spvasm_text = self._replace(spvasm_res_path.read_text(), replacements)

        spvasm_path = self.test_runtime_path / (output or spvasm_res_path.name)
        spvasm_path.write_text(spvasm_text)

        return spvasm_path

    def _replace(self, text: str, replacements: dict[str, str] | None = None) -> str:
        if replacements:
            for item, replacement in replacements.items():
                text = text.replace(item, replacement)

        return text


class SDKTool:
    """Wrapper around the tool executable."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def run(self, *params: str) -> None:
        """Run the tool's executable."""
        as_strings = (str(param) for param in params)
        cmd = [self.path.as_posix(), *as_strings]

        logger.debug("Executing command: %s", cmd)
        subprocess.check_call(cmd)


class SDKTools:
    """Class with collection utility methods to compile and run code via ScenarioRunner."""

    def __init__(
        self,
        scenario_runner,
        glsl_compiler,
        dds_utils,
        resources_helper,
        spirv_as,
        spirv_val,
        emulation_layer,
    ) -> None:
        self.scenario_runner = scenario_runner
        self.glsl_compiler = glsl_compiler
        self.dds_utils = dds_utils
        self.resources_helper = resources_helper
        self.spirv_as = spirv_as
        self.spirv_val = spirv_val
        self.emulation_layer = emulation_layer

    def generate_dds_file(
        self,
        height: int,
        width: int,
        element_dtype: str,
        element_size: int,
        dxgi_format: str,
        filename: str,
        data: bytes | None = None,
    ) -> Path:
        """Generate DDS file."""
        dds_file_path = self.resources_helper.get_testenv_path(filename)

        self.dds_utils.run(
            "--action",
            "generate",
            "--height",
            height,
            "--width",
            width,
            "--element-dtype",
            element_dtype,
            "--element-size",
            element_size,
            "--format",
            dxgi_format,
            "--output",
            dds_file_path,
            *(["--header-only"] if data else []),
        )

        if data:
            content = dds_file_path.read_bytes()
            dds_file_path.write_bytes(content + data)

        return dds_file_path

    def convert_dds_to_npy(
        self, dds_file_path: Path, filename: str, elementSize: int
    ) -> Path:
        """Extract data from DDS file and save as npy file."""
        npy_file_path = self.resources_helper.get_testenv_path(filename)

        self.dds_utils.run(
            "--action",
            "to_npy",
            "--input",
            dds_file_path,
            "--output",
            npy_file_path,
            "--element-size",
            f"{elementSize}",
        )

        return npy_file_path

    def compare_dds(
        self, dds_input_path: Path, dds_output_path: Path, element_dtype: str
    ) -> Path:
        """Compare two DDS files."""
        try:
            self.dds_utils.run(
                "--action",
                "compare",
                "--input",
                dds_input_path,
                "--output",
                dds_output_path,
                "--element-dtype",
                element_dtype,
            )
        except subprocess.CalledProcessError:
            return False

        return True

    def assemble_spirv(
        self,
        spvasm: str,
        replacements: dict[str, str] | None = None,
    ) -> Path:
        """Compile the spvasm and return path to the SPIR-V module."""
        spvasm = self.resources_helper.prepare_spvasm(spvasm, replacements)
        logger.debug("spvasm code:\n%s", spvasm.read_text())

        spv_path = self.resources_helper.get_testenv_path(spvasm.stem + ".spv")
        self.spirv_as.run("-o", spv_path, "--target-env", "spv1.3", spvasm)
        return spv_path

    def validate_spirv(
        self,
        spv_path: Path,
    ) -> None:
        """Validates a SPIR-V module."""
        self.spirv_val.run("--target-env", "spv1.3", spv_path)

    def compile_shader(
        self,
        shader: str,
        replacements: dict[str, str] | None = None,
        compile_opts: str | None = None,
        output: str | None = None,
    ) -> Path:
        """Compile the shader and return path to the SPIR-V module."""
        shader = self.resources_helper.prepare_shader(shader, replacements, output)
        logger.debug("Shader code:\n%s", shader.read_text())

        compiled_shader_path = self.resources_helper.get_testenv_path(
            shader.stem + ".spv"
        )

        extra_opts = []
        if compile_opts:
            # app fails when parameter is passed with -D
            compile_opts = compile_opts.replace("-D", "+D")
            extra_opts.extend(["--build-opts", compile_opts])

        self.glsl_compiler.run(
            "--input", shader, "--output", compiled_shader_path, *extra_opts
        )

        return compiled_shader_path

    def run_scenario(
        self,
        scenario_template: str,
        replacements: dict[str, str] | None = None,
        options: list[str] | None = None,
    ) -> None:
        """Prepare the scenario and run it via ScenarioRunner."""
        scenario = self.resources_helper.prepare_scenario(
            scenario_template, replacements
        )
        logger.debug("Scenario to run:\n%s", scenario.read_text())
        import sys

        print("Scenario to run:\n%s", scenario.read_text(), file=sys.stderr)

        self.scenario_runner.run("--scenario", scenario, *(options or []))


def valid_path(value) -> Path:
    """Check if provided value is a valid path to a file."""
    path = Path(value)

    if not path.is_file():
        raise pytest.UsageError(f"Path {path} does not exist or not a file")

    return path.resolve()


def valid_dir(value):
    """Check if provided value is a valid path to a directory."""
    path = Path(value)

    if not path.is_dir():
        raise pytest.UsageError(f"Path {path} does not exist or not a directory")

    return path.resolve().as_posix()


def pytest_addoption(parser) -> None:
    """Add command line options to pytest."""
    parser.addoption(
        "--scenario-runner",
        type=valid_path,
        required=True,
        help="Path to the scenario runner",
    )
    parser.addoption(
        "--glsl-compiler",
        type=valid_path,
        required=True,
        help="Path to the GLSL compiler",
    )
    parser.addoption(
        "--dds-utils",
        type=valid_path,
        required=True,
        help="Path to the DDS utils",
    )
    parser.addoption(
        "--spirv-as",
        type=valid_path,
        required=True,
        help="Path to SPIR-V assembler",
    )
    parser.addoption(
        "--spirv-val",
        type=valid_path,
        required=True,
        help="Path to SPIR-V validator",
    )
    parser.addoption(
        "--vgf-pylib-dir",
        action="store",
        type=valid_dir,
        required=True,
        help="Directory of VGF Python Lib file",
    )
    parser.addoption(
        "--emulation-layer",
        action="store_true",
        default=False,
        required=False,
        help="Specifies if emulation layer is enabled",
    )


@pytest.fixture
def scenario_runner(request) -> SDKTool:
    scenario_runner_path = request.config.getoption("--scenario-runner")
    return SDKTool(scenario_runner_path)


@pytest.fixture
def glsl_compiler(request) -> SDKTool:
    glsl_compiler_path = request.config.getoption("--glsl-compiler")
    return SDKTool(glsl_compiler_path)


@pytest.fixture
def dds_utils(request) -> SDKTool:
    dds_utils_path = request.config.getoption("--dds-utils")
    return SDKTool(dds_utils_path)


@pytest.fixture
def spirv_as(request) -> SDKTool:
    spirv_as_path = request.config.getoption("--spirv-as")
    return SDKTool(spirv_as_path)


@pytest.fixture
def spirv_val(request) -> SDKTool:
    spirv_val_path = request.config.getoption("--spirv-val")
    return SDKTool(spirv_val_path)


@pytest.fixture
def emulation_layer(request) -> SDKTool:
    emulation_layer = request.config.getoption("--emulation-layer")
    return SDKTool(emulation_layer)


@pytest.fixture
def numpy_helper(resources_helper) -> NumpyHelper:
    return NumpyHelper(resources_helper.get_testenv_path())


@pytest.fixture
def resources_helper(tmp_path: Path) -> ResourcesHelper:
    return ResourcesHelper(Path(__file__).parent / "resources", tmp_path)


@pytest.fixture
def sdk_tools(
    scenario_runner,
    glsl_compiler,
    dds_utils,
    resources_helper,
    spirv_as,
    spirv_val,
    emulation_layer,
) -> SDKTools:
    return SDKTools(
        scenario_runner,
        glsl_compiler,
        dds_utils,
        resources_helper,
        spirv_as,
        spirv_val,
        emulation_layer,
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--emulation-layer"):
        skip_emulation_layer_incompatible = pytest.mark.skip(
            reason="incompatible with --emulation-layer"
        )
        for item in items:
            if "emulation_layer_incompatible" in item.keywords:
                item.add_marker(skip_emulation_layer_incompatible)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "emulation_layer_incompatible: mark test as incompatible with emulation-layer",
    )
    sys.path.append(config.option.vgf_pylib_dir)
