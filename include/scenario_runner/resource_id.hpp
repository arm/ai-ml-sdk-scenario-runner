/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <functional>
#include <variant>

namespace mlsdk::scenariorunner {

/// @brief Strongly typed index identifying a registered scenario resource.
///
/// IDs are created by ScenarioBuilder and are valid only for the scenario built
/// by that builder. Different resource types cannot be mixed accidentally.
template <typename Tag> class ResourceId {
  public:
    /// @brief Underlying index type.
    using ValueType = size_t;

    /// @brief Construct an ID from an index.
    /// @param value Resource index within its type-specific collection.
    explicit constexpr ResourceId(ValueType value) : _value{value} {}

    /// @brief Return the underlying type-specific resource index.
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
struct MemoryGroupIdTag;
struct ImageBarrierIdTag;
struct BufferBarrierIdTag;
struct TensorBarrierIdTag;
struct MemoryBarrierIdTag;

using BufferId = ResourceId<BufferIdTag>;                               ///< Buffer resource ID.
using ImageId = ResourceId<ImageIdTag>;                                 ///< Image resource ID.
using TensorId = ResourceId<TensorIdTag>;                               ///< Tensor resource ID.
using ShaderId = ResourceId<ShaderIdTag>;                               ///< Shader resource ID.
using RawDataId = ResourceId<RawDataIdTag>;                             ///< Raw-data resource ID.
using DataGraphId = ResourceId<DataGraphIdTag>;                         ///< Data-graph resource ID.
using GraphConstantResourceId = ResourceId<GraphConstantResourceIdTag>; ///< Graph-constant resource ID.
using MemoryGroupId = ResourceId<MemoryGroupIdTag>;                     ///< Memory aliasing group ID.
using ImageBarrierId = ResourceId<ImageBarrierIdTag>;                   ///< Image barrier ID.
using BufferBarrierId = ResourceId<BufferBarrierIdTag>;                 ///< Buffer barrier ID.
using TensorBarrierId = ResourceId<TensorBarrierIdTag>;                 ///< Tensor barrier ID.
using MemoryBarrierId = ResourceId<MemoryBarrierIdTag>;                 ///< Global memory barrier ID.

/// @brief ID of a buffer, image, or tensor that can belong to a memory group.
using MemoryResourceId = std::variant<BufferId, ImageId, TensorId>;

} // namespace mlsdk::scenariorunner

namespace std {
template <typename Tag> struct hash<mlsdk::scenariorunner::ResourceId<Tag>> {
    size_t operator()(mlsdk::scenariorunner::ResourceId<Tag> id) const noexcept { return hash<size_t>{}(id.value()); }
};
} // namespace std
