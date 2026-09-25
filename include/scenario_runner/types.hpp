/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "resource_id.hpp"
#include "shader_stage.hpp"

#include "vulkan/vulkan_raii.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlsdk::scenariorunner {

class VgfView;

/// \brief Filtering operation used when sampling an image
enum class FilterMode {
    Linear,  ///< Linearly interpolate neighboring samples
    Nearest, ///< Select the sample nearest to the texture coordinates
    Unknown  ///< Filter mode has not been specified
};

/// \brief Behavior for sampling outside an image's normalized coordinate range
enum class AddressMode {
    ClampBorder,    ///< Use the configured border color
    ClampEdge,      ///< Extend the nearest edge texel
    Repeat,         ///< Repeat the image periodically
    MirroredRepeat, ///< Repeat and mirror the image periodically
    Unknown         ///< Address mode has not been specified
};

/// \brief Color returned when sampling beyond an image border
enum class BorderColor {
    FloatTransparentBlack, ///< Floating-point transparent black
    FloatOpaqueBlack,      ///< Floating-point opaque black
    FloatOpaqueWhite,      ///< Floating-point opaque white
    IntTransparentBlack,   ///< Integer transparent black
    IntOpaqueBlack,        ///< Integer opaque black
    IntOpaqueWhite,        ///< Integer opaque white
    FloatCustomEXT,        ///< Application-provided floating-point border color
    IntCustomEXT,          ///< Application-provided integer border color
    Unknown                ///< Border color has not been specified
};

/// \brief Memory organization used for an image or tensor
enum class Tiling {
    Optimal, ///< Implementation-dependent organization optimized for device access
    Linear,  ///< Texels or tensor elements use a linear organization
    Unknown  ///< Tiling has not been specified
};

/// \brief Type of memory access synchronized by a barrier
enum class MemoryAccess {
    ComputeShaderWrite, ///< Write performed by a compute shader
    MemoryWrite,        ///< Any memory write
    MemoryRead,         ///< Any memory read
    GraphWrite,         ///< Write performed by a data graph
    GraphRead,          ///< Read performed by a data graph
    ComputeShaderRead,  ///< Read performed by a compute shader
    Unknown             ///< Memory access has not been specified
};
/// \brief Pipeline stage synchronized by a barrier
enum class PipelineStage {
    Graph,    ///< Data graph pipeline stage
    Compute,  ///< Compute shader pipeline stage
    Graphics, ///< Graphics pipeline stages
    All,      ///< All supported pipeline stages
    Unknown   ///< Pipeline stage has not been specified
};

/// \brief Image layout used before or after an image barrier
enum class ImageLayout {
    General,        ///< General-purpose image layout
    TensorAliasing, ///< Layout used when image memory aliases tensor memory
    Undefined,      ///< Previous image contents need not be preserved
    Unknown         ///< Image layout has not been specified
};
/// \brief Mip levels and array layers affected by an image operation
struct SubresourceRange {
    /// First mip level in the range.
    uint32_t baseMipLevel{0};
    /// Number of mip levels in the range.
    uint32_t levelCount{1};
    /// First array layer in the range.
    uint32_t baseArrayLayer{0};
    /// Number of array layers in the range.
    uint32_t layerCount{1};
};

/// \brief A variant of the supported custom border color representations
using CustomColorValue = std::variant<std::array<float, 4>, std::array<int32_t, 4>>;

/// \brief Structure that describes 1-dimensional buffer data
///
/// \note We don't account for any specialized meta-data like
/// for example padding or stride information. We assume that
/// the provided data are packed in a linear manner
struct BufferInfo {
    /// Human-readable name used in diagnostics and profiling output.
    std::string debugName;
    /// Buffer size in bytes.
    uint32_t size;
    /// Byte offset within an aliased memory allocation.
    uint64_t memoryOffset{};
};

/// \brief Information needed to load raw data from a file
struct RawDataInfo {
    /// Human-readable name used in diagnostics.
    std::string debugName;
    /// Path to the file containing the raw bytes.
    std::string src;
};

/// \brief Structure that describes N-dimensional data
///
/// \note We don't account for any specialized meta-data like
/// for example padding or stride information. We assume that
/// the provided data are packed in a linear manner
struct TensorInfo {
    /// Human-readable name used in diagnostics and profiling output.
    std::string debugName;
    /// Logical tensor dimensions.
    std::vector<int64_t> shape;
    /// Vulkan® format describing each tensor element.
    vk::Format format;
    /// Sparse dimension index, or -1 for a dense tensor.
    int64_t sparsityDimension{-1};
    /// Enable descriptor-buffer capture and replay for this tensor.
    bool descriptorBufferCaptureReplay{false};
    /// Memory organization used by the tensor.
    Tiling tiling{Tiling::Linear};
    /// Byte offset within an aliased memory allocation.
    uint64_t memoryOffset{};
};

/// @brief Structure that describes the sampler of an image
struct SamplerSettings {
    /// Minification filter.
    FilterMode minFilter = FilterMode::Nearest;
    /// Magnification filter.
    FilterMode magFilter = FilterMode::Nearest;
    /// Filter used between mip levels.
    FilterMode mipFilter = FilterMode::Nearest;
    /// Addressing mode for the U coordinate.
    AddressMode addressModeU = AddressMode::ClampEdge;
    /// Addressing mode for the V coordinate.
    AddressMode addressModeV = AddressMode::ClampEdge;
    /// Addressing mode for the W coordinate.
    AddressMode addressModeW = AddressMode::ClampEdge;
    /// Border color used by clamp-to-border addressing.
    BorderColor borderColor = BorderColor::FloatTransparentBlack;
    /// Value used when borderColor selects a custom color.
    CustomColorValue customBorderColor;
};

/// \brief Structure that describes image
///
/// \note We don't account for any specialized meta-data like
/// for example padding or stride information. We assume that
/// the provided data are packed in a linear manner
struct ImageInfo {
    /// Human-readable name used in diagnostics and profiling output.
    std::string debugName;
    /// Logical image dimensions.
    std::vector<int64_t> shape;
    /// Format of data supplied to the image.
    vk::Format format;
    /// Format used by the Vulkan® image.
    vk::Format targetFormat;
    /// Whether the image is populated as an input during initialization.
    bool isInput;
    /// Sampling configuration for the image.
    SamplerSettings samplerSettings;
    /// Number of mip levels.
    uint32_t mips;
    /// Allow the image to be used as a sampled image.
    bool isSampled{false};
    /// Allow the image to be used as a storage image.
    bool isStorage{false};
    /// Allow the image to be used as a color attachment.
    bool isColorAttachment{false};
    /// Requested memory tiling, or no value to let Scenario Runner choose.
    std::optional<Tiling> tiling;
    /// Byte offset within an aliased memory allocation.
    uint64_t memoryOffset{};
};

/// \brief Value of a shader specialization constant
///
/// Used for specialization constants
union Constant {
    int32_t i;
    uint32_t ui;
    float f;
};

/// \brief Data and tensor metadata for a graph constant
struct GraphConstantInfo {
    GraphConstantInfo() = default;
    GraphConstantInfo(std::string debugName, vk::Format format, std::vector<int64_t> dims)
        : format(format), dims(std::move(dims)), debugName(std::move(debugName)) {}

    /// Vulkan® format describing each constant element.
    vk::Format format{vk::Format::eUndefined};
    /// Logical dimensions of the constant tensor.
    std::vector<int64_t> dims;
    /// Constant bytes in tightly packed tensor order.
    std::vector<uint8_t> data;
    /// Human-readable name used in diagnostics.
    std::string debugName;
};

/// \brief Associates a descriptor binding location with a memory resource
struct TypedBinding {
    /// Descriptor set index.
    uint32_t set{};
    /// Binding index within the descriptor set.
    uint32_t id{};
    /// Buffer, image, or tensor bound to the descriptor.
    MemoryResourceId resource;
    /// Image mip level to bind, or no value for the base level.
    std::optional<uint32_t> lod;
    /// Descriptor type used for this binding.
    vk::DescriptorType vkDescriptorType{};
};

/// \brief Representation of shader source or binary data
enum class ShaderType {
    Unknown, ///< Shader representation has not been specified
    SPIR_V,  ///< SPIR-V™ binary
    GLSL,    ///< GLSL source code
    HLSL,    ///< HLSL source code
};

/// \brief Specialization constant identifier and value
struct SpecializationConstant {
    /// Specialization constant ID declared by the shader.
    int id{};
    /// Value supplied for the specialization constant.
    Constant value;
};

/// \brief Maps specialization constants to a shader within a data graph
struct SpecializationConstantMap {
    /// Constants supplied to the target shader.
    std::vector<SpecializationConstant> specializationConstants;
    /// Name of the shader within the data graph.
    std::string shaderTarget;
};

/// \brief Information needed to load and configure a VGF
struct VgfInfo {
    /// Human-readable name used in diagnostics and profiling output.
    std::string debugName;
    /// Immutable in-memory VGF view.
    std::shared_ptr<const VgfView> src;
    /// Number of push-constant bytes required by the graph.
    uint32_t pushConstantsSize{};
    /// Per-shader specialization constants applied to the graph.
    std::vector<SpecializationConstantMap> specializationConstantMaps;
};

/// Load a VGF file into an immutable view.
///
/// The returned view can be assigned directly to VgfInfo::src.
std::shared_ptr<const VgfView> loadVgfView(const std::string &sourcePath);

/// \brief Shader source, compilation settings, and execution metadata
struct ShaderInfo {
    /// Human-readable name used in diagnostics and profiling output.
    std::string debugName;
    /// Shader entry-point name.
    std::string entry;
    /// Number of push-constant bytes required by the shader.
    uint32_t pushConstantsSize{};
    /// Specialization constants applied when creating the pipeline.
    std::vector<SpecializationConstant> specializationConstants;
    /// Immutable in-memory SPIR-V module.
    std::shared_ptr<const std::vector<uint32_t>> src;
    /// Representation from which src was produced.
    ShaderType shaderType{ShaderType::Unknown};
    /// Pipeline stage that executes the shader.
    ShaderStage stage{ShaderStage::Unknown};
    /// Additional options passed to the shader compiler.
    std::string buildOpts;
    /// Directories searched for source include files.
    std::vector<std::string> includeDirs;
};

/// Load or compile a shader file into an immutable SPIR-V module.
///
/// The shader type and compilation settings are taken from shaderInfo. The
/// returned module can be assigned directly to ShaderInfo::src.
std::shared_ptr<const std::vector<uint32_t>> readShaderCode(const std::string &sourcePath,
                                                            const ShaderInfo &shaderInfo);

/// \brief Access and pipeline-stage dependencies shared by barrier resources
struct BaseBarrierInfo {
    /// Human-readable name used in diagnostics.
    std::string debugName;
    /// Access operations that must complete before the barrier.
    MemoryAccess srcAccess{MemoryAccess::Unknown};
    /// Access operations that wait for the barrier.
    MemoryAccess dstAccess{MemoryAccess::Unknown};
    /// Pipeline stages that must complete before the barrier.
    std::vector<PipelineStage> srcStages{PipelineStage::All};
    /// Pipeline stages that wait for the barrier.
    std::vector<PipelineStage> dstStages{PipelineStage::All};
};

/// \brief Barrier dependencies and layout transition for an image
struct ImageBarrierInfo : BaseBarrierInfo {
    /// Image synchronized by the barrier.
    ImageId image{0};
    /// Expected layout before the barrier.
    ImageLayout oldLayout{ImageLayout::Undefined};
    /// Required layout after the barrier.
    ImageLayout newLayout{ImageLayout::Undefined};
    /// Mip levels and array layers synchronized by the barrier.
    SubresourceRange range;
};

/// \brief Barrier dependencies for a byte range within a buffer
struct BufferBarrierInfo : BaseBarrierInfo {
    /// Buffer synchronized by the barrier.
    BufferId buffer{0};
    /// First byte synchronized by the barrier.
    uint64_t offset{};
    /// Number of bytes synchronized by the barrier.
    uint64_t size{};
};

/// \brief Barrier dependencies for a tensor
struct TensorBarrierInfo : BaseBarrierInfo {
    /// Tensor synchronized by the barrier.
    TensorId tensor{0};
};

/// \brief Global memory barrier dependencies
struct MemoryBarrierInfo : BaseBarrierInfo {};

} // namespace mlsdk::scenariorunner
