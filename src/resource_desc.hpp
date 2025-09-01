/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "barrier.hpp"
#include "commands.hpp"
#include "guid.hpp"
#include "types.hpp"

namespace mlsdk::scenariorunner {

enum class ResourceType {
    Unknown,
    Buffer,
    DataGraph,
    Shader,
    RawData,
    Tensor,
    Image,
    ImageBarrier,
    MemoryBarrier,
    TensorBarrier,
    BufferBarrier
};

struct MemoryGroup {
    Guid memoryUid;
    uint64_t offset{};
};

/**
 * @brief ResourceDesc describes a Resource.
 *
 */
struct ResourceDesc {
    ResourceDesc() = default;
    ResourceDesc(ResourceType resourceType, Guid guid, const std::string &guidStr);
    const std::optional<std::string> &getSource() const { return src; };
    void setSrc(std::string s) { src = std::move(s); }
    const std::optional<std::string> &getDestination() const { return dst; };
    virtual ~ResourceDesc() = default;

    ResourceType resourceType{ResourceType::Unknown};
    Guid guid;
    std::string guidStr;
    std::optional<std::string> src;
    std::optional<std::string> dst;
};

enum class ShaderAccessType { Unknown, ReadOnly, WriteOnly, ReadWrite, ImageRead };

/**
 * @brief BufferDesc describes a Buffer.
 *
 */
struct BufferDesc : ResourceDesc {
    BufferDesc();
    BufferDesc(Guid guid, const std::string &guidStr, uint32_t size, ShaderAccessType shaderAccess);

    uint32_t size{};
    ShaderAccessType shaderAccess{ShaderAccessType::Unknown};
    std::optional<MemoryGroup> memoryGroup;
};

/**
 * @brief Specialization constants used in shaders
 *
 */
struct SpecializationConstant {
    SpecializationConstant() = default;
    SpecializationConstant(int id, Constant value);
    int id{};
    Constant value;
};

/**
 * @brief SpecializationConstantMap is used to map specialization constants to multiple shaders within a graph
 *
 */
struct SpecializationConstantMap {
    SpecializationConstantMap() = default;
    SpecializationConstantMap(std::vector<SpecializationConstant> specializationConstants, Guid shaderTarget);

    std::vector<SpecializationConstant> specializationConstants;
    Guid shaderTarget;
};

/**
 * @brief DataGraphDesc describes a DataGraph file.
 *
 */
struct DataGraphDesc : ResourceDesc {
    DataGraphDesc();
    DataGraphDesc(Guid guid, const std::string &guidStr, std::string src);

    std::vector<ShaderSubstitutionDesc> shaderSubstitutions;
    std::vector<SpecializationConstantMap> specializationConstantMaps;
    uint32_t pushConstantsSize{};
};

enum class ShaderType { Unknown, SPIR_V, GLSL };

/**
 * @brief ShaderDesc describes a Shader.
 *
 */
struct ShaderDesc : ResourceDesc {
    ShaderDesc();
    ShaderDesc(Guid guid, const std::string &guidStr, std::string src, std::string entry, ShaderType type);

    std::string entry;
    ShaderType shaderType{ShaderType::Unknown};
    uint32_t pushConstantsSize{};
    std::vector<SpecializationConstant> specializationConstants;
    std::string buildOpts;
    std::vector<std::string> includeDirs;
};

/**
 * @brief RawDataDesc describes a raw data resource.
 *
 */
struct RawDataDesc : ResourceDesc {
    RawDataDesc();
    RawDataDesc(Guid guid, const std::string &guidStr, std::string src);
};

/**
 * @brief TensorDesc describes a Tensor.
 *
 */
struct TensorDesc : ResourceDesc {
    TensorDesc();
    TensorDesc(Guid guid, const std::string &guidStr, std::vector<int64_t> dims, ShaderAccessType shaderAccess);

    std::vector<int64_t> dims;
    ShaderAccessType shaderAccess{ShaderAccessType::Unknown};
    std::string format;
    std::optional<Tiling> tiling;
    std::optional<MemoryGroup> memoryGroup;
};

/**
 * @brief ImageDesc describes an Image.
 *
 */
struct ImageDesc : ResourceDesc {
    ImageDesc();
    ImageDesc(Guid guid, const std::string &guidStr, std::vector<uint32_t> dims, uint32_t mips,
              ShaderAccessType shaderAccess);

    std::vector<uint32_t> dims;
    uint32_t mips{1};
    std::string format;
    ShaderAccessType shaderAccess = ShaderAccessType::Unknown;

    std::optional<FilterMode> minFilter;
    std::optional<FilterMode> magFilter;
    std::optional<FilterMode> mipFilter;
    std::optional<AddressMode> borderAddressMode;
    std::optional<BorderColor> borderColor;
    std::optional<CustomColorValue> customBorderColor;
    std::optional<Tiling> tiling;
    std::optional<MemoryGroup> memoryGroup;
};

/**
 * @brief BaseBarrierDesc
 *
 */
struct BaseBarrierDesc : ResourceDesc {
    BaseBarrierDesc(ResourceType resourceType, const std::string &guidStr, MemoryAccess srcAccess,
                    MemoryAccess dstAccess);
    using ResourceDesc::ResourceDesc;

    MemoryAccess srcAccess{MemoryAccess::Unknown};
    MemoryAccess dstAccess{MemoryAccess::Unknown};
    std::vector<PipelineStage> srcStages = {PipelineStage::All};
    std::vector<PipelineStage> dstStages = {PipelineStage::All};
};

/**
 * @brief ImageBarrierDesc describes a Barrier.
 *
 */
struct ImageBarrierDesc : BaseBarrierDesc {
    ImageBarrierDesc();
    ImageBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess, ImageLayout oldLayout,
                     ImageLayout newLayout, std::string imageResource, SubresourceRange imageRange);

    ImageLayout oldLayout{ImageLayout::Undefined};
    ImageLayout newLayout{ImageLayout::Undefined};
    std::string imageResource;
    SubresourceRange imageRange{0, 1, 0, 1};
};

/**
 * @brief MemoryBarrierDesc describes a memory barrier.
 *
 */
struct MemoryBarrierDesc : BaseBarrierDesc {
    MemoryBarrierDesc();
    MemoryBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess);
};

/**
 * @brief TensorBarrierDesc describes a Tensor memory barrier.
 *
 */
struct TensorBarrierDesc : BaseBarrierDesc {
    TensorBarrierDesc();
    TensorBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess, std::string resource);

    std::string tensorResource;
};

/**
 * @brief ImageBarrierDesc describes an Image memory barrier.
 *
 */
struct BufferBarrierDesc : BaseBarrierDesc {
    BufferBarrierDesc();
    BufferBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess,
                      std::string bufferResource, uint64_t offset, uint64_t size);

    std::string bufferResource;
    uint64_t offset{};
    uint64_t size{};
};

} // namespace mlsdk::scenariorunner
