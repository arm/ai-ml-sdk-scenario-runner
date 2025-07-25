/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "commands.hpp"
#include "nlohmann/json.hpp"

#include "vulkan/vulkan_raii.hpp"

#include <variant>

using json = nlohmann::json;

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
typedef std::variant<std::array<float, 4>, std::array<int32_t, 4>> CustomColorValue;

/// \brief Structure that describes DDS pixel data layout
struct DDSPixelFormat {
    uint32_t size{32};
    uint32_t flags{};
    uint32_t fourCC{};
    uint32_t rgbBitCount{};
    uint32_t rBitMask{};
    uint32_t gBitMask{};
    uint32_t bBitMask{};
    uint32_t aBitMask{};
};

/// \brief Structure that describes main part of header from DDS file
///
/// \note Order of members and presence of "Reserved" data allow memcpy to work on file data
struct DDSHeader {
    uint32_t size{124};
    uint32_t flags{};
    uint32_t height{};
    uint32_t width{};
    uint32_t pitchOrLinearSize{};
    uint32_t depth{};
    uint32_t mipMapCount{};
    uint32_t reserved[11]{};
    DDSPixelFormat pixelFormat{};
    uint32_t caps{};
    uint32_t caps2{};
    uint32_t caps3{};
    uint32_t caps4{};
    uint32_t reserved2{};
};

/// \brief Structure that describes optional extension to DDS header format
struct DDSHeaderDX10 {
    uint32_t dxgiFormat{};
    uint32_t resourceDimension{};
    uint32_t miscFlag{};
    uint32_t arraySize{};
    uint32_t miscFlags2{};
};

/// \brief Structure that stores all non-pixel data from DDS file
struct DDSHeaderInfo {
    uint32_t magicWord{};
    DDSHeader header{};
    DDSHeaderDX10 header10{};
    bool isDx10{};
};

/// \brief Structure that describes 1-dimensional buffer data
///
/// \note We don't account for any specialized meta-data like
/// for example padding or stride information. We assume that
/// the provided data are packed in a linear manner
struct BufferInfo {
    std::string debugName;
    uint32_t size;
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
    bool isAliased{false};
    Tiling tiling{Tiling::Linear};
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
};

/// @brief Structure that describes the image barrier
struct ImageBarrierData {
    std::string debugName;
    MemoryAccess srcAccess;
    MemoryAccess dstAccess;
    std::vector<PipelineStage> srcStages;
    std::vector<PipelineStage> dstStages;
    ImageLayout oldLayout;
    ImageLayout newLayout;
    vk::Image image;
    SubresourceRange imageRange;
};

/// @brief Structure that describes the tensor barrier
struct TensorBarrierData {
    std::string debugName;
    MemoryAccess srcAccess;
    MemoryAccess dstAccess;
    std::vector<PipelineStage> srcStages;
    std::vector<PipelineStage> dstStages;
    vk::TensorARM tensor;
};

/// @brief Structure that describes the memory barrier
struct MemoryBarrierData {
    std::string debugName;
    MemoryAccess srcAccess;
    MemoryAccess dstAccess;
    std::vector<PipelineStage> srcStages;
    std::vector<PipelineStage> dstStages;
};

/// @brief Structure that describes the image barrier
struct BufferBarrierData {
    std::string debugName;
    MemoryAccess srcAccess;
    MemoryAccess dstAccess;
    std::vector<PipelineStage> srcStages;
    std::vector<PipelineStage> dstStages;
    uint64_t offset;
    uint64_t size;
    vk::Buffer buffer;
};

struct ConstantTensorData {
    uint32_t id;
    TensorInfo info;
    std::vector<char> data;
};

/// \brief Constant structure
///
/// Used for specialization constants
union Constant {
    int32_t i;   ///< Signed 32-bit integer value
    uint32_t ui; ///< Unsigned 32-bit integer value
    float f;     ///< 32-bit floating point value
};

} // namespace mlsdk::scenariorunner
