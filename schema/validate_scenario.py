#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
""" Validate the scenario against the json schema """
import argparse
import json
import os

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ErrorTree
from referencing import Registry
from referencing import Resource


def validate_scenario(schema_filename, json_filename):
    with open(json_filename) as scenario_file:
        scenario = json.loads(scenario_file.read())

    schema_path = os.path.dirname(schema_filename)

    registry = Registry()

    for fname in os.listdir(schema_path):
        fpath = os.path.join(schema_path, fname)
        if fpath.endswith(".schema"):
            with open(fpath, "r") as schema_file:
                schema = json.loads(schema_file.read())
                resource = Resource.from_contents(schema)
                registry = resource @ registry

    with open(schema_filename) as schema_file:
        schema = json.loads(schema_file.read())

    validator = Draft202012Validator(schema, registry=registry)
    validator.check_schema(schema)

    errors = sorted(validator.iter_errors(scenario), key=str)

    idx = 0
    for error in errors:
        print(f"Error {idx}")
        print(error.instance)
        print(error.message)
        print(list(error.path))
        for suberror in sorted(error.context, key=lambda e: e.schema_path):
            print(list(suberror.schema_path), suberror.message, sep=", ")
        print("")

        idx += 1

    if len(errors) == 0:
        return errors

    def next_indent(num):
        print("")
        print(" " * (num - 1), end=" ")

    tree = ErrorTree(validator.iter_errors(scenario))
    print("=== TREE ===")

    def recurse(tree_in, depth=0):
        if len(tree_in.errors):
            print(f"{tree_in.errors}", end="")
            next_indent(depth)

        for error in tree_in.__iter__():
            print(f"{error}", end=" --> ")
            recurse(tree_in[error], depth + 5 + len(str(error)))

    recurse(tree)
    print("\n=== END ===")

    return errors


def validate(args=None):
    schema_base = os.path.dirname(__file__)
    schema_file = "scenario.schema"
    schema = os.path.join(schema_base, schema_file)

    errors = validate_scenario(schema, args.scenario)
    if len(errors) == 0:
        print(f"{args.scenario} OK!\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        "-s",
        help="Scenario to validate",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    validate(parse_arguments())
