/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "barrier.hpp"
#include "guid.hpp"
#include "types.hpp"
#include <optional>
#include <string>
#include <typeinfo>

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

/**
 * @brief ResourceDesc describes a Resource.
 *
 */
struct ResourceDesc {
    ResourceDesc() = default;
    ResourceDesc(ResourceType resourceType, Guid guid, const std::string &guidStr);
    virtual const std::optional<std::string> &getSource() const { return src; };
    virtual const std::optional<std::string> &getDestination() const { return dst; };
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

    const std::optional<std::string> &getSource() const override { return src; };
    const std::optional<std::string> &getDestination() const override { return dst; };

    std::optional<std::string> src;
    std::optional<std::string> dst;
    uint32_t size{};
    ShaderAccessType shaderAccess{ShaderAccessType::Unknown};
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
    DataGraphDesc(Guid guid, const std::string &guidStr, const std::string &src);

    std::string src;
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
    ShaderDesc(Guid guid, const std::string &guidStr, const std::string &src, const std::string &entry,
               ShaderType type);

    std::string src;
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
    RawDataDesc(Guid guid, const std::string &guidStr, const std::string &src);

    std::string src{""};
};

struct AliasTarget {
    Guid resourceRef;
    int mipLevel{};
    int arrayLayer{};
};

/**
 * @brief TensorDesc describes a Tensor.
 *
 */
struct TensorDesc : ResourceDesc {
    TensorDesc();
    TensorDesc(Guid guid, const std::string &guidStr, const std::vector<int64_t> &dims, ShaderAccessType shaderAccess);

    const std::optional<std::string> &getSource() const override { return src; };
    const std::optional<std::string> &getDestination() const override { return dst; };

    std::vector<int64_t> dims;
    ShaderAccessType shaderAccess{ShaderAccessType::Unknown};
    std::optional<std::string> src;
    std::optional<std::string> dst;
    std::string format;
    AliasTarget aliasTarget{};
    std::optional<Tiling> tiling;
};

/**
 * @brief ImageDesc describes an Image.
 *
 */
struct ImageDesc : ResourceDesc {
    ImageDesc();
    ImageDesc(Guid guid, const std::string &guidStr, const std::vector<uint32_t> &dims, uint32_t mips,
              ShaderAccessType shaderAccess);

    const std::optional<std::string> &getSource() const override { return src; };
    const std::optional<std::string> &getDestination() const override { return dst; };

    std::vector<uint32_t> dims;
    uint32_t mips{1};
    std::string format;
    ShaderAccessType shaderAccess = ShaderAccessType::Unknown;
    std::optional<std::string> src;
    std::optional<std::string> dst;

    std::optional<FilterMode> minFilter;
    std::optional<FilterMode> magFilter;
    std::optional<FilterMode> mipFilter;
    std::optional<AddressMode> borderAddressMode;
    std::optional<BorderColor> borderColor;
    std::optional<CustomColorValue> customBorderColor;
    std::optional<Tiling> tiling;
};

/**
 * @brief ImageBarrierDesc describes a Barrier.
 *
 */
struct ImageBarrierDesc : ResourceDesc {
    ImageBarrierDesc();
    ImageBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess, ImageLayout oldLayout,
                     ImageLayout newLayout, const std::string &imageResource, SubresourceRange imageRange);

    MemoryAccess srcAccess{MemoryAccess::Unknown};
    MemoryAccess dstAccess{MemoryAccess::Unknown};
    std::vector<PipelineStage> srcStages = {PipelineStage::All};
    std::vector<PipelineStage> dstStages = {PipelineStage::All};
    ImageLayout oldLayout{ImageLayout::Undefined};
    ImageLayout newLayout{ImageLayout::Undefined};
    std::string imageResource;
    SubresourceRange imageRange{0, 1, 0, 1};
};

/**
 * @brief MemoryBarrierDesc describes a memory barrier.
 *
 */
struct MemoryBarrierDesc : ResourceDesc {
    MemoryBarrierDesc();
    MemoryBarrierDesc(const std::string &guidStr, const MemoryAccess srcAccess, MemoryAccess dstAccess);

    MemoryAccess srcAccess{MemoryAccess::Unknown};
    MemoryAccess dstAccess{MemoryAccess::Unknown};
    std::vector<PipelineStage> srcStages = {PipelineStage::All};
    std::vector<PipelineStage> dstStages = {PipelineStage::All};
};

/**
 * @brief TensorBarrierDesc describes a Tensor memory barrier.
 *
 */
struct TensorBarrierDesc : ResourceDesc {
    TensorBarrierDesc();
    TensorBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess, std::string resource);

    MemoryAccess srcAccess{MemoryAccess::Unknown};
    MemoryAccess dstAccess{MemoryAccess::Unknown};
    std::vector<PipelineStage> srcStages = {PipelineStage::All};
    std::vector<PipelineStage> dstStages = {PipelineStage::All};
    std::string tensorResource;
};

/**
 * @brief ImageBarrierDesc describes an Image memory barrier.
 *
 */
struct BufferBarrierDesc : ResourceDesc {
    BufferBarrierDesc();
    BufferBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess,
                      const std::string &bufferResource, uint64_t offset, uint64_t size);

    MemoryAccess srcAccess{MemoryAccess::Unknown};
    MemoryAccess dstAccess{MemoryAccess::Unknown};
    std::vector<PipelineStage> srcStages = {PipelineStage::All};
    std::vector<PipelineStage> dstStages = {PipelineStage::All};
    std::string bufferResource;
    uint64_t offset{};
    uint64_t size{};
};

} // namespace mlsdk::scenariorunner
