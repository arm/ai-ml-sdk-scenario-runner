/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "barrier.hpp"
#include "memory_map.hpp"
#include "numpy.hpp"
#include "utils.hpp"

#include "vgf/decoder.h"
#include "vgf/decoder.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <set>

namespace mlsdk::scenariorunner {

DataManager::DataManager(Context &ctx) : _ctx(ctx) {}

void DataManager::createBuffer(Guid guid, const BufferInfo &info, const std::vector<char> &values) {
    _buffers.insert({guid, Buffer(_ctx, info.debugName, info.size)});
    auto &buffer = _buffers[guid];
    buffer.fill(values.data(), values.size());
}

void DataManager::createBuffer(Guid guid, const BufferInfo &info, const mlsdk::numpy::data_ptr &dataPtr) {
    _buffers.insert({guid, Buffer(_ctx, info.debugName, info.size)});
    auto &buffer = _buffers[guid];
    buffer.fill(dataPtr.ptr, dataPtr.size());
}

void DataManager::createZeroedBuffer(Guid guid, const BufferInfo &info) {
    _buffers.insert({guid, Buffer(_ctx, info.debugName, info.size)});
    auto &buffer = _buffers[guid];
    buffer.fillZero();
}

void DataManager::createTensor(Guid guid, const TensorInfo &info, std::optional<Guid> aliasTarget) {
    if (aliasTarget.has_value()) {
        _tensors.insert(
            {guid, Tensor(_ctx, info.debugName, info.format, info.shape, info.isAliased,
                          Tensor::convertTiling(info.tiling), getMemoryManager(aliasTarget.value()), false)});
    } else {
        createMemoryManager(guid);
        _tensors.insert({guid, Tensor(_ctx, info.debugName, info.format, info.shape, info.isAliased,
                                      Tensor::convertTiling(info.tiling), getMemoryManager(guid), false)});
    }
}

void DataManager::createImage(Guid guid, const ImageInfo &info) {
    createMemoryManager(guid);
    _images.insert({guid, Image(_ctx, info, getMemoryManager(guid))});
}

void DataManager::createVgfView(Guid guid, const DataGraphDesc &desc) {
    std::unique_ptr<MemoryMap> mapped = std::make_unique<MemoryMap>(desc.src);

    // Create header decoder
    std::unique_ptr<vgflib::HeaderDecoder> headerDecoder = vgflib::CreateHeaderDecoder(mapped->ptr());
    if (!headerDecoder->IsValid()) {
        throw std::runtime_error("Invalid VGF header");
    }
    if (!headerDecoder->CheckVersion()) {
        throw std::runtime_error("Incompatible VGF header");
    }

    uint64_t moduleTableOffset = headerDecoder->GetModuleTableOffset();
    uint64_t sequenceTableOffset = headerDecoder->GetModelSequenceTableOffset();
    uint64_t resourceTableOffset = headerDecoder->GetModelResourceTableOffset();
    uint64_t constantsOffset = headerDecoder->GetConstantsOffset();

    std::unique_ptr<vgflib::ModuleTableDecoder> moduleTableDecoder =
        vgflib::CreateModuleTableDecoder(reinterpret_cast<const uint8_t *>(mapped->ptr(moduleTableOffset)));
    std::unique_ptr<vgflib::ModelSequenceTableDecoder> sequenceTableDecoder =
        vgflib::CreateModelSequenceTableDecoder(reinterpret_cast<const uint8_t *>(mapped->ptr(sequenceTableOffset)));
    std::unique_ptr<vgflib::ModelResourceTableDecoder> resourceTableDecoder =
        vgflib::CreateModelResourceTableDecoder(reinterpret_cast<const uint8_t *>(mapped->ptr(resourceTableOffset)));
    std::unique_ptr<vgflib::ConstantDecoder> constantTableDecoder =
        vgflib::CreateConstantDecoder(reinterpret_cast<const uint8_t *>(mapped->ptr(constantsOffset)));

    VgfView vgfView(std::move(mapped), std::move(moduleTableDecoder), std::move(sequenceTableDecoder),
                    std::move(resourceTableDecoder), std::move(constantTableDecoder));

    _vgfViews.insert({guid, std::move(vgfView)});
}

void DataManager::createImageBarrier(Guid guid, const ImageBarrierData &data) {
    _imageBarriers.insert({guid, VulkanImageBarrier(data)});
}

void DataManager::createTensorBarrier(Guid guid, const TensorBarrierData &data) {
    _tensorBarriers.insert({guid, VulkanTensorBarrier(data)});
}

void DataManager::createMemoryBarrier(Guid guid, const MemoryBarrierData &data) {
    _memoryBarriers.insert({guid, VulkanMemoryBarrier(data)});
}

void DataManager::createBufferBarrier(Guid guid, const BufferBarrierData &data) {
    _bufferBarriers.insert({guid, VulkanBufferBarrier(data)});
}

void DataManager::createRawData(Guid guid, const std::string &debugName, const std::string &src) {
    _rawData.insert({guid, RawData(debugName, src)});
}

bool DataManager::hasBuffer(Guid guid) const { return _buffers.find(guid) != _buffers.end(); }

bool DataManager::hasTensor(Guid guid) const { return _tensors.find(guid) != _tensors.end(); }

bool DataManager::hasImage(Guid guid) const { return _images.find(guid) != _images.end(); }

bool DataManager::hasRawData(Guid guid) const { return _rawData.find(guid) != _rawData.end(); }

bool DataManager::hasImageBarrier(Guid guid) const { return _imageBarriers.find(guid) != _imageBarriers.end(); }

bool DataManager::hasMemoryBarrier(Guid guid) const { return _memoryBarriers.find(guid) != _memoryBarriers.end(); }

bool DataManager::hasTensorBarrier(Guid guid) const { return _tensorBarriers.find(guid) != _tensorBarriers.end(); }

bool DataManager::hasBufferBarrier(Guid guid) const { return _bufferBarriers.find(guid) != _bufferBarriers.end(); }

uint32_t DataManager::numBuffers() const { return static_cast<uint32_t>(_buffers.size()); }

uint32_t DataManager::numTensors() const { return static_cast<uint32_t>(_tensors.size()); }

uint32_t DataManager::numImages() const { return static_cast<uint32_t>(_images.size()); }

Buffer &DataManager::getBufferMut(const Guid &guid) {
    if (_buffers.find(guid) == _buffers.end()) {
        throw std::runtime_error("Buffer not found");
    }
    return _buffers[guid];
}

Tensor &DataManager::getTensorMut(const Guid &guid) {
    if (_tensors.find(guid) == _tensors.end()) {
        throw std::runtime_error("Tensor not found");
    }
    return _tensors[guid];
}

Image &DataManager::getImageMut(const Guid &guid) {
    if (_images.find(guid) == _images.end()) {
        throw std::runtime_error("Image not found");
    }
    return _images[guid];
}

const Buffer &DataManager::getBuffer(const Guid &guid) const {
    if (_buffers.find(guid) == _buffers.end()) {
        throw std::runtime_error("Buffer not found");
    }
    return _buffers.at(guid);
}

const Tensor &DataManager::getTensor(const Guid &guid) const {
    if (_tensors.find(guid) == _tensors.end()) {
        throw std::runtime_error("Tensor not found");
    }
    return _tensors.at(guid);
}

const Image &DataManager::getImage(const Guid &guid) const {
    if (_images.find(guid) == _images.end()) {
        throw std::runtime_error("Image not found");
    }
    return _images.at(guid);
}

const RawData &DataManager::getRawData(const Guid &guid) const {
    if (_rawData.find(guid) == _rawData.end()) {
        throw std::runtime_error("RawData not found");
    }
    return _rawData.at(guid);
}

const VgfView &DataManager::getVgfView(const Guid &guid) const {
    if (_vgfViews.find(guid) == _vgfViews.end()) {
        throw std::runtime_error("Vgf not found");
    }
    return _vgfViews.at(guid);
}

const VulkanImageBarrier &DataManager::getImageBarrier(const Guid &guid) const {
    if (_imageBarriers.find(guid) == _imageBarriers.end()) {
        throw std::runtime_error("Image Barrier not found");
    }
    return _imageBarriers.at(guid);
}

const VulkanTensorBarrier &DataManager::getTensorBarrier(const Guid &guid) const {
    if (_tensorBarriers.find(guid) == _tensorBarriers.end()) {
        throw std::runtime_error("Tensor Barrier not found");
    }
    return _tensorBarriers.at(guid);
}

const VulkanMemoryBarrier &DataManager::getMemoryBarrier(const Guid &guid) const {
    if (_memoryBarriers.find(guid) == _memoryBarriers.end()) {
        throw std::runtime_error("Memory Barrier not found");
    }
    return _memoryBarriers.at(guid);
}

const VulkanBufferBarrier &DataManager::getBufferBarrier(const Guid &guid) const {
    if (_bufferBarriers.find(guid) == _bufferBarriers.end()) {
        throw std::runtime_error("Buffer Barrier not found");
    }
    return _bufferBarriers.at(guid);
}

void DataManager::storeResource(const ResourceDesc &resourceDesc) {
    auto &dst = resourceDesc.getDestination();
    if (!dst.has_value()) {
        return;
    }

    getResourceMut(resourceDesc).store(_ctx, dst.value());
}

Resource &DataManager::getResourceMut(const ResourceDesc &resourceDesc) {
    auto &guid = resourceDesc.guid;
    switch (resourceDesc.resourceType) {
    case ResourceType::Buffer:
        return getBufferMut(guid);
    case ResourceType::Tensor:
        return getTensorMut(guid);
    case ResourceType::Image:
        return getImageMut(guid);
    default:
        throw std::runtime_error("Resource not found");
    }
}

vk::DescriptorType DataManager::getResourceDescriptorType(const Guid &guid) const {
    if (hasBuffer(guid)) {
        return vk::DescriptorType::eStorageBuffer;
    } else if (hasTensor(guid)) {
        return vk::DescriptorType::eTensorARM;
    } else if (hasImage(guid)) {
        if (getImage(guid).isSampled()) {
            return vk::DescriptorType::eCombinedImageSampler;
        } else {
            return vk::DescriptorType::eStorageImage;
        }
    } else {
        throw std::runtime_error("Invalid resource descriptor type");
    }
}

void DataManager::createMemoryManager(Guid guid) {
    _memoryManagers.insert({guid, std::make_shared<ResourceMemoryManager>()});
}

std::shared_ptr<ResourceMemoryManager> DataManager::getMemoryManager(const Guid &guid) const {
    auto it = _memoryManagers.find(guid);
    if (it != _memoryManagers.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace mlsdk::scenariorunner
