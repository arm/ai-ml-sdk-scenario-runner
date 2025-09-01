/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resource_desc.hpp"
#include "utils.hpp"

namespace mlsdk::scenariorunner {

/**
 * @brief Construct a new ResourceDesc object
 *
 * @param resourceType
 * @param guid
 * @param guidStr
 */
ResourceDesc::ResourceDesc(ResourceType resourceType, Guid guid, const std::string &guidStr)
    : resourceType(resourceType), guid(guid), guidStr(guidStr) {}

/**
 * @brief Construct a new BufferDesc object
 *
 * @param guid
 * @param guidStr
 * @param size
 * @param shaderAccess
 */
BufferDesc::BufferDesc(Guid guid, const std::string &guidStr, uint32_t size, ShaderAccessType shaderAccess)
    : ResourceDesc(ResourceType::Buffer, guid, guidStr), size(size), shaderAccess(shaderAccess) {}

BufferDesc::BufferDesc() : ResourceDesc(ResourceType::Buffer, Guid(), "<unnamed_buffer>") {}

/**
 * @brief Construct a new SpecializationConstant object
 *
 * @param id
 * @param value
 */
SpecializationConstant::SpecializationConstant(int id, Constant value) : id(id), value(value) {}

/**
 * @brief Construct a new SpecializationConstantMap object
 *
 * @param specializationConstants
 * @param shaderTarget
 */
SpecializationConstantMap::SpecializationConstantMap(std::vector<SpecializationConstant> specializationConstants,
                                                     Guid shaderTarget)
    : specializationConstants(std::move(specializationConstants)), shaderTarget(shaderTarget) {}

/**
 * @brief Construct a new DataGraphDesc object
 *
 * @param guid
 * @param guidStr
 * @param src
 */
DataGraphDesc::DataGraphDesc(Guid guid, const std::string &guidStr, std::string src)
    : ResourceDesc(ResourceType::DataGraph, guid, guidStr) {
    setSrc(std::move(src));
}

DataGraphDesc::DataGraphDesc() : ResourceDesc(ResourceType::DataGraph, Guid(), "<unnamed_data_graph>") {}

/**
 * @brief Construct a new ShaderDesc object
 *
 * @param guid
 * @param guidStr
 * @param src
 * @param entry
 * @param type
 */
ShaderDesc::ShaderDesc(Guid guid, const std::string &guidStr, std::string src, std::string entry, ShaderType type)
    : ResourceDesc(ResourceType::Shader, guid, guidStr), entry(std::move(entry)), shaderType(type) {
    setSrc(std::move(src));
}

ShaderDesc::ShaderDesc() : ResourceDesc(ResourceType::Shader, Guid(), "<unnamed_shader>") {}

/**
 * @brief Construct a new RawDataDesc object
 *
 * @param guid
 * @param guidStr
 * @param src
 */
RawDataDesc::RawDataDesc(Guid guid, const std::string &guidStr, std::string src)
    : ResourceDesc(ResourceType::RawData, guid, guidStr) {
    setSrc(std::move(src));
}

RawDataDesc::RawDataDesc() : ResourceDesc(ResourceType::RawData, Guid(), "<unnamed_raw_data>") {}

/**
 * @brief Construct a new TensorDesc object
 *
 * @param guid
 * @param guidStr
 * @param dims
 * @param shaderAccess
 */
TensorDesc::TensorDesc(Guid guid, const std::string &guidStr, std::vector<int64_t> dims, ShaderAccessType shaderAccess)
    : ResourceDesc(ResourceType::Tensor, guid, guidStr), dims(std::move(dims)), shaderAccess(shaderAccess) {}

TensorDesc::TensorDesc() : ResourceDesc(ResourceType::Tensor, Guid(), "<unnamed_tensor>") {}

/**
 * @brief Construct a new ImageDesc object
 *
 * @param guid
 * @param guidStr
 * @param dims
 * @param mips
 * @param shaderAccess
 */
ImageDesc::ImageDesc(Guid guid, const std::string &guidStr, std::vector<uint32_t> dims, uint32_t mips,
                     ShaderAccessType shaderAccess)
    : ResourceDesc(ResourceType::Image, guid, guidStr), dims(std::move(dims)), mips(mips), shaderAccess(shaderAccess) {}

ImageDesc::ImageDesc() : ResourceDesc(ResourceType::Image, Guid(), "<unnamed_image>") {}

/**
 * @brief Construct a new BaseBarrierDesc object
 */
BaseBarrierDesc::BaseBarrierDesc(ResourceType resourceType, const std::string &guidStr, MemoryAccess srcAccess,
                                 MemoryAccess dstAccess)
    : ResourceDesc(resourceType, guidStr, guidStr), srcAccess(srcAccess), dstAccess(dstAccess) {}

/**
 * @brief Construct a new BufferBarrierDesc object
 *
 * @param guidStr
 * @param srcAccess
 * @param dstAccess
 * @param resource
 * @param offset
 * @param size
 */
BufferBarrierDesc::BufferBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess,
                                     std::string resource, uint64_t offset, uint64_t size)
    : BaseBarrierDesc(ResourceType::BufferBarrier, guidStr, srcAccess, dstAccess), bufferResource(std::move(resource)),
      offset(offset), size(size) {}

BufferBarrierDesc::BufferBarrierDesc()
    : BaseBarrierDesc(ResourceType::BufferBarrier, Guid(), "<unnamed_buffer_barrier>") {}

/**
 * @brief Construct a new MemoryBarrierDesc object
 *
 * @param guidStr
 * @param srcAccess
 * @param dstAccess
 */
MemoryBarrierDesc::MemoryBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess)
    : BaseBarrierDesc(ResourceType::MemoryBarrier, guidStr, srcAccess, dstAccess) {}

MemoryBarrierDesc::MemoryBarrierDesc()
    : BaseBarrierDesc(ResourceType::MemoryBarrier, Guid(), "<unnamed_memory_barrier>") {}

/**
 * @brief Construct a new ImageBarrierDesc object
 *
 * @param guidStr
 * @param srcAccess
 * @param dstAccess
 * @param oldLayout
 * @param newLayout
 * @param imageResource
 * @param imageRange
 */
ImageBarrierDesc::ImageBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess,
                                   ImageLayout oldLayout, ImageLayout newLayout, std::string imageResource,
                                   SubresourceRange imageRange)
    : BaseBarrierDesc(ResourceType::ImageBarrier, guidStr, srcAccess, dstAccess), oldLayout(oldLayout),
      newLayout(newLayout), imageResource(std::move(imageResource)), imageRange(imageRange) {}

ImageBarrierDesc::ImageBarrierDesc() : BaseBarrierDesc(ResourceType::ImageBarrier, Guid(), "<unnamed_image_barrier>") {}

TensorBarrierDesc::TensorBarrierDesc(const std::string &guidStr, MemoryAccess srcAccess, MemoryAccess dstAccess,
                                     std::string tensorResource)
    : BaseBarrierDesc(ResourceType::TensorBarrier, guidStr, srcAccess, dstAccess),
      tensorResource(std::move(tensorResource)) {}

TensorBarrierDesc::TensorBarrierDesc()
    : BaseBarrierDesc(ResourceType::TensorBarrier, Guid(), "<unnamed_tensor_barrier>") {}

} // namespace mlsdk::scenariorunner
