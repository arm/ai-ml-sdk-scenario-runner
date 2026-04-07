#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pathlib
import sys

import pytest
from jsonschema.exceptions import best_match

"""Validate the scenario against the json schema"""

RUNNER_ROOT_DIR = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / ".." / ".."
SCHEMA_FILE = RUNNER_ROOT_DIR / "schema" / "scenario.schema"
TEST_BASE = RUNNER_ROOT_DIR / "src" / "tests" / "json"

# Import validation script
PLUGIN_PATH = os.path.dirname(SCHEMA_FILE)
sys.path.append(PLUGIN_PATH)
from validate_scenario import validate_scenario


def RequiredProperty(prop_name):
    return f"'{prop_name}' is a required property"


def UnexpectedProperty(prop_name):
    return f"Additional properties are not allowed ('{prop_name}' was unexpected)"


def RequiredMin(value):
    return f"is less than the minimum of {value}"


def IsEmpty():
    return "should be non-empty"


def TooShort():
    return "is too short"


def TooLong():
    return "is too long"


def InvalidStringValue(value):
    return f"'{value}' is not one of ["


def InvalidConstValue(value):
    return f"'{value}' was expected"


def InvalidMutuallyExclusive():
    return "is valid under each of "


def InvalidEnum():
    return "is not one of"


def Ok():
    return ""


def InvalidType(type_name):
    return f"is not of type '{type_name}'"


def InvalidAnyOf():
    return "is not valid under any of the given schemas"


def ShouldBeNonEmpty():
    return "should be non-empty"


def check_scenario(schema_filename, expected_msg, json_filename):
    json_filename = TEST_BASE / json_filename

    errors = validate_scenario(schema_filename, json_filename)

    if expected_msg == Ok():
        assert len(errors) == 0
    else:
        print(errors)
        assert expected_msg in best_match(errors).message


# define paths
commands_path = pathlib.Path("commands")
dispatch_compute_path = commands_path / "dispatch_compute"
dispatch_graph_path = commands_path / "dispatch_graph"
dispatch_spirv_graph_path = commands_path / "dispatch_spirv_graph"
dispatch_barrier_path = commands_path / "dispatch_barrier"
dispatch_fragment_path = commands_path / "dispatch_fragment"
mark_boundary_path = commands_path / "mark_boundary"
resources_path = pathlib.Path("resources")
shader_path = resources_path / "shader"
graph_path = resources_path / "graph"
buffer_path = resources_path / "buffer"
tensor_path = resources_path / "tensor"
image_path = resources_path / "image"
memory_group_path = resources_path / "memory_group"
raw_data_path = resources_path / "raw_data"
buffer_barrier_path = resources_path / "buffer_barrier"
image_barrier_path = resources_path / "image_barrier"
tensor_barrier_path = resources_path / "tensor_barrier"
memory_barrier_path = resources_path / "memory_barrier"
graph_constant_path = resources_path / "graph_constant"


@pytest.mark.parametrize(
    "expected_msg, json_filename",
    [
        # fmt: off
        # List of expected_message:json_filename pair

        ############
        # COMMANDS
        (RequiredProperty("commands"), "no_commands.json"),
        (UnexpectedProperty("not_a_real_command"), commands_path/"invalid_command.json"),
        (InvalidMutuallyExclusive(), commands_path/"mutually_exclusive_commands.json"),
        # dispatch_compute
        (Ok(), dispatch_compute_path/"reference.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), dispatch_compute_path/"invalid_property.json"),
        (RequiredProperty("bindings"), dispatch_compute_path/"missing_bindings.json"),
        (Ok(), dispatch_compute_path/"missing_push_data_ref.json"),
        (RequiredProperty("rangeND"), dispatch_compute_path/"missing_rangeND.json"),
        (RequiredProperty("shader_ref"), dispatch_compute_path/"missing_shader_ref.json"),
        (RequiredMin(1), dispatch_compute_path/"rangeND_invalid_dim.json"),
        (TooShort(), dispatch_compute_path/"rangeND_invalid_length.json"),
        (TooLong(), dispatch_compute_path/"rangeND_invalid_length2.json"),
        # dispatch_compute->bindings
        (RequiredProperty("resource_ref"), dispatch_compute_path/"binding_missing_resource_ref.json"),
        (RequiredMin(0), dispatch_compute_path/"binding_negative_id.json"),
        (RequiredMin(0), dispatch_compute_path/"binding_negative_set.json"),
        # dispatch_fragment
        (Ok(), dispatch_fragment_path/"reference.json"),
        (Ok(), dispatch_fragment_path/"reference_render_extent.json"),
        (RequiredProperty("vertex_shader_ref"), dispatch_fragment_path/"missing_vertex_shader_ref.json"),
        (RequiredProperty("fragment_shader_ref"), dispatch_fragment_path/"missing_fragment_shader_ref.json"),
        (RequiredProperty("bindings"), dispatch_fragment_path/"missing_bindings.json"),
        (InvalidAnyOf(), dispatch_fragment_path/"missing_render_extent_and_color.json"),
        (ShouldBeNonEmpty(), dispatch_fragment_path/"invalid_color_attachment_refs.json"),
        (TooShort(), dispatch_fragment_path/"invalid_render_extent.json"),
        (RequiredMin(1), dispatch_fragment_path/"invalid_render_extent_value.json"),
        # dispatch_graph
        (Ok(), dispatch_graph_path/"reference.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), dispatch_graph_path/"invalid_property.json"),
        (RequiredProperty("bindings"), dispatch_graph_path/"missing_bindings.json"),
        (RequiredProperty("graph_ref"), dispatch_graph_path/"missing_graph_ref.json",),

        # dispatch_spirv_graph
        (Ok(), dispatch_spirv_graph_path/"reference.json"),
        (Ok(), dispatch_spirv_graph_path/"graph_constants_ok.json"),
        (InvalidType("array"), dispatch_spirv_graph_path/"graph_constants_invalid_type.json"),
        (InvalidType("string"), dispatch_spirv_graph_path/"graph_constants_items_invalid_type.json"),
        # dispatch_graph->bindings
        (RequiredProperty("resource_ref"),dispatch_graph_path/"binding_missing_resource_ref.json"),
        (RequiredMin(0), dispatch_graph_path/"binding_negative_id.json"),
        (RequiredMin(0), dispatch_graph_path/"binding_negative_set.json"),
        # dispatch_barrier
        (Ok(), dispatch_barrier_path/"reference.json"),
        (RequiredProperty("image_barrier_refs"), dispatch_barrier_path/"missing_image_barrier_refs.json"),
        (RequiredProperty("tensor_barrier_refs"), dispatch_barrier_path/"missing_tensor_barrier_refs.json"),
        (RequiredProperty("buffer_barrier_refs"), dispatch_barrier_path/"missing_buffer_barrier_refs.json"),
        (RequiredProperty("memory_barrier_refs"), dispatch_barrier_path/"missing_memory_barrier_refs.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), dispatch_barrier_path/"invalid_property.json"),

        # mark_boundary
        (Ok(), mark_boundary_path/"reference.json"),
        (Ok(), mark_boundary_path/"minimal.json"),
        (RequiredProperty("resources"), mark_boundary_path/"no_resource.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), mark_boundary_path/"invalid_property.json"),

        ############
        # RESOURCES
        (RequiredProperty("resources"), "no_resources.json"),
        (UnexpectedProperty("not_a_real_resource"), resources_path/"invalid_resource.json"),
        (InvalidMutuallyExclusive(), resources_path/"mutually_exclusive_resources.json"),
        # shader
        (Ok(), shader_path/"reference.json"),
        (Ok(), shader_path/"minimal.json"),
        (Ok(), shader_path/"valid_types.json"),
        (InvalidStringValue("PASCAL"), shader_path/"invalid_types.json"),
        (RequiredProperty("uid"), shader_path/"missing_uid.json"),
        (RequiredProperty("src"), shader_path/"missing_src.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), shader_path/"invalid_property.json"),
        (InvalidConstValue("main"), shader_path/"glsl_invalid_entry.json"),
        # buffer
        (Ok(), buffer_path/"reference.json"),
        (Ok(), buffer_path/"minimal.json"),
        (RequiredProperty("uid"), buffer_path/"missing_uid.json"),
        (RequiredProperty("size"), buffer_path/"missing_size.json"),
        (RequiredProperty("shader_access"), buffer_path/"missing_shader_access.json"),
        (InvalidStringValue("writeread"), buffer_path/"invalid_shader_access.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), buffer_path/"invalid_property.json"),
        (InvalidMutuallyExclusive(), buffer_path/"mutually_exclusive_src_dst.json"),
        # graph
        (Ok(), graph_path/"reference.json"),
        (Ok(), graph_path/"minimal.json"),
        (RequiredProperty("uid"), graph_path/"missing_uid.json"),
        (RequiredProperty("src"), graph_path/"missing_src.json"),
        (RequiredProperty("shader_ref"), graph_path/"missing_shader_sub_shader_ref.json"),
        (RequiredProperty("target"), graph_path/"missing_shader_sub_target.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), graph_path/"invalid_property.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), graph_path/"invalid_spec_consts_map_property.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), graph_path/"invalid_spec_const_property.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), graph_path/"invalid_shader_sub_property.json"),
        # raw_data
        (Ok(), raw_data_path/"reference.json"),
        (RequiredProperty("uid"), raw_data_path/"missing_uid.json"),
        (RequiredProperty("src"), raw_data_path/"missing_src.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), raw_data_path/"invalid_property.json"),
        # tensor
        (Ok(), tensor_path/"reference.json"),
        (Ok(), tensor_path/"minimal.json"),
        (RequiredProperty("uid"), tensor_path/"missing_uid.json"),
        (RequiredProperty("dims"), tensor_path/"missing_dims.json"),
        (RequiredProperty("format"), tensor_path/"missing_format.json"),
        (RequiredProperty("shader_access"), tensor_path/"missing_shader_access.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), tensor_path/"invalid_property.json"),
        (TooLong(), tensor_path/"invalid_dims_length.json"),
        (RequiredMin(1), tensor_path/"invalid_dim_value.json"),
        (InvalidMutuallyExclusive(), tensor_path/"mutually_exclusive_src_dst.json"),
        # image
        (Ok(), image_path/"reference.json"),
        (Ok(), image_path/"minimal.json"),
        (RequiredProperty("uid"), image_path/"missing_uid.json"),
        (RequiredProperty("format"), image_path/"missing_format.json"),
        (RequiredProperty("dims"), image_path/"missing_dims.json"),
        (TooLong(), image_path/"invalid_dims_length.json"),
        (RequiredMin(0), image_path/"invalid_dim_value.json"),
        (RequiredProperty("shader_access"), image_path/"missing_shader_access.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), image_path/"invalid_property.json"),
        (InvalidEnum(), image_path/"invalid_shader_access.json"),
        (InvalidEnum(), image_path/"invalid_min_filter.json"),
        (InvalidEnum(), image_path/"invalid_mag_filter.json"),
        (InvalidEnum(), image_path/"invalid_mip_filter.json"),
        (RequiredMin(1), image_path/"invalid_mips.json"),
        (InvalidMutuallyExclusive(), image_path/"mutually_exclusive_src_dst.json"),
        (InvalidEnum(), image_path/"invalid_border_address_mode.json"),
        (InvalidEnum(), image_path/"invalid_border_color.json"),
        (TooLong(), image_path/"invalid_custom_border_color_length.json"),
        # memory group
        (Ok(), memory_group_path/"minimal.json"),
        (RequiredProperty("id"), memory_group_path/"no_id.json"),
        # memory barrier
        (Ok(), memory_barrier_path/"reference.json"),
        (InvalidEnum(), memory_barrier_path/"invalid_dst_access.json"),
        (InvalidEnum(), memory_barrier_path/"invalid_src_access.json"),
        (RequiredProperty("uid"), memory_barrier_path/"missing_uid.json"),
        (RequiredProperty("dst_access"), memory_barrier_path/"missing_dst_access.json"),
        (RequiredProperty("src_access"), memory_barrier_path/"missing_src_access.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), memory_barrier_path/"invalid_property.json"),
        # tensor barrier
        (Ok(), tensor_barrier_path/"reference.json"),
        (InvalidEnum(), tensor_barrier_path/"invalid_dst_access.json"),
        (InvalidEnum(), tensor_barrier_path/"invalid_src_access.json"),
        (RequiredProperty("uid"), tensor_barrier_path/"missing_uid.json"),
        (RequiredProperty("dst_access"), tensor_barrier_path/"missing_dst_access.json"),
        (RequiredProperty("src_access"), tensor_barrier_path/"missing_src_access.json"),
        (RequiredProperty("tensor_resource"), tensor_barrier_path/"missing_tensor_resource.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), tensor_barrier_path/"invalid_property.json"),
        # buffer barrier
        (Ok(), buffer_barrier_path/"reference.json"),
        (InvalidEnum(), buffer_barrier_path/"invalid_dst_access.json"),
        (InvalidEnum(), buffer_barrier_path/"invalid_src_access.json"),
        (RequiredProperty("uid"), buffer_barrier_path/"missing_uid.json"),
        (RequiredProperty("size"), buffer_barrier_path/"missing_size.json"),
        (RequiredProperty("dst_access"), buffer_barrier_path/"missing_dst_access.json"),
        (RequiredProperty("src_access"), buffer_barrier_path/"missing_src_access.json"),
        (RequiredProperty("buffer_resource"), buffer_barrier_path/"missing_buffer_resource.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), buffer_barrier_path/"invalid_property.json"),
        (RequiredMin(0), buffer_barrier_path/"invalid_offset.json"),
        (Ok(), buffer_barrier_path/"graphics_stage.json"),
        # image barrier
        (Ok(), image_barrier_path/"reference.json"),
        (InvalidEnum(), image_barrier_path/"invalid_dst_access.json"),
        (InvalidEnum(), image_barrier_path/"invalid_src_access.json"),
        (RequiredProperty("uid"), image_barrier_path/"missing_uid.json"),
        (RequiredProperty("dst_access"), image_barrier_path/"missing_dst_access.json"),
        (RequiredProperty("src_access"), image_barrier_path/"missing_src_access.json"),
        (RequiredProperty("image_resource"), image_barrier_path/"missing_image_resource.json"),
        (RequiredProperty("old_layout"), image_barrier_path/"missing_old_layout.json"),
        (RequiredProperty("new_layout"), image_barrier_path/"missing_new_layout.json"),
        (RequiredProperty("base_array_layer"), image_barrier_path/"missing_subresource_range_base_array_layer.json"),
        (RequiredProperty("base_mip_level"), image_barrier_path/"missing_subresource_range_base_mip_level.json"),
        (RequiredProperty("layer_count"), image_barrier_path/"missing_subresource_range_layer_count.json"),
        (RequiredProperty("level_count"), image_barrier_path/"missing_subresource_range_level_count.json"),
        (RequiredMin(0), image_barrier_path/"invalid_subresource_range_base_mip_level.json"),
        (RequiredMin(0), image_barrier_path/"invalid_subresource_range_base_array_layer.json"),
        (RequiredMin(1), image_barrier_path/"invalid_subresource_range_mip_level_count.json"),
        (RequiredMin(1), image_barrier_path/"invalid_subresource_range_array_layer_count.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), image_barrier_path/"invalid_property.json"),
        # graph constant
        (Ok(), graph_constant_path/"reference.json"),
        (RequiredProperty("uid"), graph_constant_path/"missing_uid.json"),
        (RequiredProperty("dims"), graph_constant_path/"missing_dims.json"),
        (RequiredProperty("format"), graph_constant_path/"missing_format.json"),
        (RequiredProperty("src"), graph_constant_path/"missing_src.json"),
        (RequiredMin(1), graph_constant_path/"invalid_dim_value.json"),
        (TooLong(), graph_constant_path/"invalid_dims_length.json"),
        (IsEmpty(), graph_constant_path/"invalid_dims_empty.json"),
        (UnexpectedProperty("this_is_an_invalid_property"), graph_constant_path/"invalid_property.json"),
        # TODO
        # 1. src and dst properties MUST be mutually exclusive
        # 2. double check completeness of negative test coverage
        # fmt: on
    ],
)
def test_scenario(expected_msg, json_filename):
    check_scenario(SCHEMA_FILE, expected_msg, json_filename)
