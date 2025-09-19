/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "guid.hpp"

#include "vulkan/vulkan_raii.hpp"

#include <array>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlsdk::scenariorunner {

enum class ModuleType { SHADER, GRAPH };
enum class FilterMode { Linear, Nearest, Unknown };
enum class AddressMode { ClampBorder, ClampEdge, Repeat, MirroredRepeat, Unknown };
enum class BorderColor {
    FloatTransparentBlack,
    FloatOpaqueBlack,
    FloatOpaqueWhite,
    IntTransparentBlack,
    IntOpaqueBlack,
    IntOpaqueWhite,
    FloatCustomEXT,
    IntCustomEXT,
    Unknown
};
enum class Tiling { Optimal, Linear, Unknown };
enum class MemoryAccess {
    ComputeShaderWrite,
    MemoryWrite,
    MemoryRead,
    GraphWrite,
    GraphRead,
    ComputeShaderRead,
    Unknown
};
enum class PipelineStage { Graph, Compute, All, Unknown };
enum class ImageLayout { General, TensorAliasing, Undefined, Unknown };
struct SubresourceRange {
    uint32_t baseMipLevel{0};
    uint32_t levelCount{1};
    uint32_t baseArrayLayer{0};
    uint32_t layerCount{1};
};

/// \brief A variant of the potential custom border colors */
using CustomColorValue = std::variant<std::array<float, 4>, std::array<int32_t, 4>>;

/// \brief Structure that describes 1-dimensional buffer data
///
/// \note We don't account for any specialized meta-data like
/// for example padding or stride information. We assume that
/// the provided data are packed in a linear manner
struct BufferInfo {
    std::string debugName;
    uint32_t size;
    uint64_t memoryOffset{};
};

/// \brief Structure that describes N-dimensional data
///
/// \note We don't account for any specialized meta-data like
/// for example padding or stride information. We assume that
/// the provided data are packed in a linear manner
struct TensorInfo {
    std::string debugName;
    std::vector<int64_t> shape;
    vk::Format format;
    int64_t sparsityDimension{-1};
    bool isAliasedWithImage{false};
    Tiling tiling{Tiling::Linear};
    uint64_t memoryOffset{};
};

/// @brief Structure that describes the sampler of an image
struct SamplerSettings {
    FilterMode minFilter = FilterMode::Nearest;
    FilterMode magFilter = FilterMode::Nearest;
    FilterMode mipFilter = FilterMode::Nearest;
    AddressMode borderAddressMode = AddressMode::ClampEdge;
    BorderColor borderColor = BorderColor::FloatTransparentBlack;
    CustomColorValue customBorderColor;
};

/// \brief Structure that describes image
///
/// \note We don't account for any specialized meta-data like
/// for example padding or stride information. We assume that
/// the provided data are packed in a linear manner
struct ImageInfo {
    std::string debugName;
    std::vector<int64_t> shape;
    vk::Format format;
    vk::Format targetFormat;
    bool isInput;
    SamplerSettings samplerSettings;
    bool isAliased{false};
    uint32_t mips;
    bool isSampled{false};
    bool isStorage{false};
    std::optional<Tiling> tiling;
    uint64_t memoryOffset{};
};

/// \brief Constant structure
///
/// Used for specialization constants
union Constant {
    int32_t i;   ///< Signed 32-bit integer value
    uint32_t ui; ///< Unsigned 32-bit integer value
    float f;     ///< 32-bit floating point value
};

/// \brief Typed binding
struct TypedBinding {
    uint32_t set{};
    uint32_t id{};
    Guid resourceRef;
    std::optional<uint32_t> lod;
    vk::DescriptorType vkDescriptorType{};
};

/// \brief Group count for x, y and z
struct ComputeDispatch {
    uint32_t gwcx{1};
    uint32_t gwcy{1};
    uint32_t gwcz{1};
};

} // namespace mlsdk::scenariorunner
