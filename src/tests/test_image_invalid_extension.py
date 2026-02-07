#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Tests handling of unsupported image extensions."""
import pytest


def test_png_unsupported_extension_fails(sdk_tools, resources_helper):
    input = resources_helper.get_testenv_path("input.bmp")
    input.write_bytes(b"not a real image")
    scenario_path = resources_helper.prepare_scenario(
        "test_image_invalid_extension/unsupported.json"
    )
    with pytest.raises(Exception):
        sdk_tools.run_scenario(scenario_path.name)
