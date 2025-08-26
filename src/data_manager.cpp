/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "barrier.hpp"
#include "utils.hpp"

#include "vgf/decoder.h"
#include "vgf/decoder.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <set>

namespace mlsdk::scenariorunner {
namespace {
constexpr vk::DescriptorType convertDescriptorType(const DescriptorType descriptorType) {
    switch (descriptorType) {
    case DescriptorType::StorageImage:
        return vk::DescriptorType::eStorageImage;
    case DescriptorType::Auto:
        throw std::runtime_error("Cannot infer the descriptor type without context");
    default:
        throw std::runtime_error("Descriptor type is invalid");
    }
}

} // namespace

DataManager::DataManager(Context &ctx) : _ctx(ctx) {}

void DataManager::createBuffer(Guid guid, const BufferInfo &info) {
    _buffers.insert({guid, Buffer(_ctx, info, getOrCreateMemoryManager(guid))});
}

void DataManager::createBuffer(Guid guid, const BufferInfo &info, std::vector<char> &values) {
    createBuffer(guid, info);
    auto &buffer = getBufferMut(guid);
    buffer.allocateMemory(_ctx);
    buffer.fill(values.data(), values.size());
}

void DataManager::createTensor(Guid guid, const TensorInfo &info) {
    _tensors.insert({guid, Tensor(_ctx, info, getOrCreateMemoryManager(guid))});
}

void DataManager::createImage(Guid guid, const ImageInfo &info) {
    _images.insert({guid, Image(_ctx, info, getOrCreateMemoryManager(guid))});
}

void DataManager::createVgfView(Guid guid, const DataGraphDesc &desc) {
    auto mapped = std::make_unique<MemoryMap>(desc.src.value());

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

    // Verify file content
    if (!vgflib::VerifyModuleTable(mapped->ptr(moduleTableOffset), headerDecoder->GetModuleTableSize())) {
        throw std::runtime_error("Invalid module table");
    }
    if (!vgflib::VerifyModelSequenceTable(mapped->ptr(sequenceTableOffset),
                                          headerDecoder->GetModelSequenceTableSize())) {
        throw std::runtime_error("Invalid model sequence table");
    }
    if (!vgflib::VerifyModelResourceTable(mapped->ptr(resourceTableOffset),
                                          headerDecoder->GetModelResourceTableSize())) {
        throw std::runtime_error("Invalid model resource table");
    }
    if (!vgflib::VerifyConstant(mapped->ptr(constantsOffset), headerDecoder->GetConstantsSize())) {
        throw std::runtime_error("Invalid constant section");
    }

    auto moduleTableDecoder = vgflib::CreateModuleTableDecoder(mapped->ptr(moduleTableOffset));
    auto sequenceTableDecoder = vgflib::CreateModelSequenceTableDecoder(mapped->ptr(sequenceTableOffset));
    auto resourceTableDecoder = vgflib::CreateModelResourceTableDecoder(mapped->ptr(resourceTableOffset));
    auto constantTableDecoder = vgflib::CreateConstantDecoder(mapped->ptr(constantsOffset));

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
vk::DescriptorType DataManager::getDescriptorType(const BindingDesc &bindingDesc) const {
    return bindingDesc.descriptorType == DescriptorType::Auto ? getResourceDescriptorType(bindingDesc.resourceRef)
                                                              : convertDescriptorType(bindingDesc.descriptorType);
}

std::shared_ptr<ResourceMemoryManager> DataManager::getOrCreateMemoryManager(const Guid &resourceGuid) {
    Guid groupGuid{};
    for (const auto &groupToResource : _groupToResources) {
        if (groupToResource.second.count(resourceGuid)) {
            groupGuid = groupToResource.first;
            break;
        }
    }
    auto memMan = getMemoryManager(groupGuid);
    if (memMan == nullptr) {
        _groupMemoryManagers.insert({groupGuid, std::make_shared<ResourceMemoryManager>()});
        return getMemoryManager(groupGuid);
    }
    return memMan;
}

std::shared_ptr<ResourceMemoryManager> DataManager::getMemoryManager(const Guid &groupGuid) const {
    auto it = _groupMemoryManagers.find(groupGuid);
    if (it != _groupMemoryManagers.end()) {
        return it->second;
    }
    return nullptr;
}

void DataManager::addResourceToGroup(const Guid &group, const Guid &resource) {
    _groupToResources[group].insert(resource);
}

const std::unordered_map<Guid, std::set<Guid>> &DataManager::getResourceMemoryGroups() const {
    return _groupToResources;
}

bool DataManager::isSingleMemoryGroup(const Guid &resource) const {
    for (const auto &groupToResource : _groupToResources) {
        if (groupToResource.second.count(resource) && groupToResource.second.size() == 1) {
            return true;
        }
    }
    return false;
}

} // namespace mlsdk::scenariorunner
