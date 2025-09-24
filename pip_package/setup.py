#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
from datetime import datetime

from setuptools import find_packages
from setuptools import setup

setup(
    name="ai_ml_sdk_scenario_runner",
    version=datetime.today().strftime("%m.%d"),
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "scenario_runner": ["binaries/*"],
    },
    entry_points={
        "console_scripts": [
            "scenario_runner=scenario_runner.cli:main",
        ],
    },
    zip_safe=False,
)
