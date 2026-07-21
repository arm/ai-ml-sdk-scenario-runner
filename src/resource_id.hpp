/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <functional>
#include <variant>

namespace mlsdk::scenariorunner {

template <typename Tag> class ResourceId {
  public:
    using ValueType = size_t;

    explicit constexpr ResourceId(ValueType value) : _value{value} {}

    constexpr ValueType value() const { return _value; }

    friend constexpr bool operator==(ResourceId lhs, ResourceId rhs) { return lhs._value == rhs._value; }
    friend constexpr bool operator!=(ResourceId lhs, ResourceId rhs) { return !(lhs == rhs); }

  private:
    ValueType _value;
};

struct BufferIdTag;
struct ImageIdTag;
struct TensorIdTag;
struct ShaderIdTag;
struct RawDataIdTag;
struct DataGraphIdTag;
struct GraphConstantResourceIdTag;

using BufferId = ResourceId<BufferIdTag>;
using ImageId = ResourceId<ImageIdTag>;
using TensorId = ResourceId<TensorIdTag>;
using ShaderId = ResourceId<ShaderIdTag>;
using RawDataId = ResourceId<RawDataIdTag>;
using DataGraphId = ResourceId<DataGraphIdTag>;
using GraphConstantResourceId = ResourceId<GraphConstantResourceIdTag>;

using TypedResourceId =
    std::variant<BufferId, ImageId, TensorId, ShaderId, RawDataId, DataGraphId, GraphConstantResourceId>;

} // namespace mlsdk::scenariorunner

namespace std {
template <typename Tag> struct hash<mlsdk::scenariorunner::ResourceId<Tag>> {
    size_t operator()(mlsdk::scenariorunner::ResourceId<Tag> id) const noexcept { return hash<size_t>{}(id.value()); }
};
} // namespace std
