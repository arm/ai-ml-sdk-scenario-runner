#
# SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import re
import sys
from importlib import import_module

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.statemachine import StringList

sys.path.insert(0, os.path.abspath("."))

# Scenario Runner project config
SR_project = "Scenario Runner"
copyright = "2022-2026, Arm Limited and/or its affiliates <open-source-office@arm.com>"
author = "Arm Limited"
git_repo_tool_url = "https://gerrit.googlesource.com/git-repo"

# Set home project name
project = SR_project

rst_epilog = """
.. |SR_project| replace:: %s
.. |git_repo_tool_url| replace:: %s
""" % (
    SR_project,
    git_repo_tool_url,
)

# Enabled extensions
extensions = [
    "breathe",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

# Disable superfluous warnings
suppress_warnings = [
    "sphinx.ext.autosectionlabel.*",
    "myst.xref_missing",
    "myst.header",
]

# Breathe Configuration
breathe_projects = {"ScenarioRunner": "../generated/xml"}
breathe_default_project = "ScenarioRunner"
breathe_domain_by_extension = {"h": "c"}

# Enable RTD theme
html_theme = "sphinx_rtd_theme"

tags.add("WITH_BASE_MD")


class PyEnumTable(Directive):
    """Generate a compact table of enum members from the Python module."""

    has_content = False

    def run(self):
        module = import_module("scenario_runner_py")
        enums = sorted(
            (name, value)
            for name, value in vars(module).items()
            if isinstance(value, type)
            and hasattr(value, "__members__")
            and name != "Format"
        )
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 30 70",
            "",
            "   * - Enumeration",
            "     - Members",
        ]
        for name, enum in enums:
            members = ", ".join(f"``{member}``" for member in enum.__members__)
            lines.extend([f"   * - ``{name}``", f"     - {members}"])

        container = nodes.container()
        self.state.nested_parse(StringList(lines), self.content_offset, container)
        return container.children


class PybindOverloads(Directive):
    """Render selected pybind11 overloads as normal Python method entries."""

    has_content = False
    required_arguments = 1
    option_spec = {"methods": directives.unchanged_required}

    def run(self):
        module = import_module("scenario_runner_py")
        class_name = self.arguments[0]
        methods = [method.strip() for method in self.options["methods"].split(",")]
        lines = []

        for method in methods:
            docstring = getattr(getattr(module, class_name), method).__doc__ or ""
            _, _, overloads = docstring.partition("Overloaded function.\n\n")
            for index, overload in enumerate(
                re.split(r"\n(?=\d+\. )", overloads.strip())
            ):
                signature, _, description = overload.partition("\n\n")
                signature = re.sub(r"^\d+\. ", "", signature)
                signature = signature.replace("scenario_runner_py.", "")
                signature = re.sub(
                    rf"^{method}\(self: {class_name}, ?",
                    f"{class_name}.{method}(",
                    signature,
                )
                lines.extend([f".. py:method:: {signature}"])
                if index:
                    lines.append("   :no-index:")
                lines.append("")
                lines.extend(
                    f"   {line}" if line else "" for line in description.splitlines()
                )
                lines.append("")

        container = nodes.container()
        self.state.nested_parse(StringList(lines), self.content_offset, container)
        return container.children


def normalize_pybind11_docstrings(app, what, name, obj, options, lines):
    """Render pybind11's overload signatures as literals in autodoc output."""
    if what not in {"function", "method"}:
        return

    for index, line in enumerate(lines):
        if re.match(r"^\d+\. [A-Za-z_][A-Za-z0-9_]*\(", line):
            lines[index] = f"``{line}``"


def hide_pybind11_wrapper_signature(
    app, what, name, obj, options, signature, return_annotation
):
    """Prefer pybind11's detailed overload signatures to its *args wrapper."""
    if signature == "(*args, **kwargs)":
        return "", return_annotation
    return signature, return_annotation


def document_pybind11_attributes(app, what, name, obj, options, lines):
    """Render typed pybind11 properties declared in a class annotation map."""
    if what != "class":
        return

    for attribute, annotation in getattr(obj, "__annotations__", {}).items():
        lines.extend(
            [
                "",
                f".. py:attribute:: {attribute}",
                f"   :type: {annotation}",
            ]
        )


def setup(app):
    app.add_directive("py-enum-table", PyEnumTable)
    app.add_directive("pybind-overloads", PybindOverloads)
    app.connect("autodoc-process-docstring", normalize_pybind11_docstrings)
    app.connect("autodoc-process-signature", hide_pybind11_wrapper_signature)
    app.connect("autodoc-process-docstring", document_pybind11_attributes)
