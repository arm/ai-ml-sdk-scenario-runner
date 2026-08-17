/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_manager.hpp"
#include "utils.hpp"

#include <utility>

namespace mlsdk::scenariorunner {
namespace {
void fill(const BaseBarrierInfo &info, BaseBarrierData &data) {
    data.debugName = info.debugName;
    data.srcAccess = info.srcAccess;
    data.dstAccess = info.dstAccess;
    data.srcStages = info.srcStages;
    data.dstStages = info.dstStages;
}

template <typename Resources, typename Id>
decltype(auto) getResource(Resources &resources, Id id, const char *errorMessage) {
    const auto resource = resources.find(id);
    if (resource == resources.end()) {
        throw std::runtime_error(errorMessage);
    }
    return (resource->second);
}
} // namespace

void DataManager::createBuffer(BufferId id, const BufferInfo &info) { _buffers.emplace(id, Buffer(info)); }

void DataManager::createBuffer(BufferId id, BufferInfo &&info) { _buffers.emplace(id, Buffer(std::move(info))); }

void DataManager::createTensor(TensorId id, const TensorInfo &info) { _tensors.emplace(id, Tensor(info)); }

void DataManager::createTensor(TensorId id, TensorInfo &&info) { _tensors.emplace(id, Tensor(std::move(info))); }

void DataManager::createImage(ImageId id, const ImageInfo &info) { _images.emplace(id, Image(info)); }

void DataManager::createImage(ImageId id, ImageInfo &&info) { _images.emplace(id, Image(std::move(info))); }

void DataManager::createVgfView(DataGraphId id, const DataGraphInfo &info) {
    _vgfViews.insert({id, VgfView::createVgfView(info.src)});
}

void DataManager::createImageBarrier(ImageBarrierId id, const ImageBarrierInfo &info) {
    ImageBarrierData data;
    fill(info, data);
    data.oldLayout = info.oldLayout;
    data.newLayout = info.newLayout;
    data.image = getImage(info.image).image();
    data.imageRange = info.range;
    _imageBarriers.insert({id, VulkanImageBarrier(data)});
}

void DataManager::createTensorBarrier(TensorBarrierId id, const TensorBarrierInfo &info) {
    TensorBarrierData data;
    fill(info, data);
    data.tensor = getTensor(info.tensor).tensor();
    _tensorBarriers.insert({id, VulkanTensorBarrier(data)});
}

void DataManager::createMemoryBarrier(MemoryBarrierId id, const MemoryBarrierInfo &info) {
    MemoryBarrierData data;
    fill(info, data);
    _memoryBarriers.insert({id, VulkanMemoryBarrier(data)});
}

void DataManager::createBufferBarrier(BufferBarrierId id, const BufferBarrierInfo &info) {
    BufferBarrierData data;
    fill(info, data);
    data.buffer = getBuffer(info.buffer).buffer();
    data.offset = info.offset;
    data.size = info.size;
    _bufferBarriers.insert({id, VulkanBufferBarrier(data)});
}

void DataManager::createRawData(RawDataId id, const RawDataInfo &info) {
    _rawData.insert({id, RawData(info.debugName, info.src)});
}

bool DataManager::hasBuffer(BufferId id) const { return _buffers.find(id) != _buffers.end(); }

bool DataManager::hasTensor(TensorId id) const { return _tensors.find(id) != _tensors.end(); }

bool DataManager::hasImage(ImageId id) const { return _images.find(id) != _images.end(); }

bool DataManager::hasRawData(RawDataId id) const { return _rawData.find(id) != _rawData.end(); }

bool DataManager::hasImageBarrier(ImageBarrierId id) const { return _imageBarriers.find(id) != _imageBarriers.end(); }

bool DataManager::hasMemoryBarrier(MemoryBarrierId id) const {
    return _memoryBarriers.find(id) != _memoryBarriers.end();
}

bool DataManager::hasTensorBarrier(TensorBarrierId id) const {
    return _tensorBarriers.find(id) != _tensorBarriers.end();
}

bool DataManager::hasBufferBarrier(BufferBarrierId id) const {
    return _bufferBarriers.find(id) != _bufferBarriers.end();
}

uint32_t DataManager::numBuffers() const { return static_cast<uint32_t>(_buffers.size()); }

uint32_t DataManager::numTensors() const { return static_cast<uint32_t>(_tensors.size()); }

uint32_t DataManager::numImages() const { return static_cast<uint32_t>(_images.size()); }

Buffer &DataManager::getBufferMut(BufferId id) { return getResource(_buffers, id, "Buffer not found"); }

Tensor &DataManager::getTensorMut(TensorId id) { return getResource(_tensors, id, "Tensor not found"); }

Image &DataManager::getImageMut(ImageId id) { return getResource(_images, id, "Image not found"); }

const Buffer &DataManager::getBuffer(BufferId id) const { return getResource(_buffers, id, "Buffer not found"); }

const Tensor &DataManager::getTensor(TensorId id) const { return getResource(_tensors, id, "Tensor not found"); }

const Image &DataManager::getImage(ImageId id) const { return getResource(_images, id, "Image not found"); }

const RawData &DataManager::getRawData(RawDataId id) const {
    if (_rawData.find(id) == _rawData.end()) {
        throw std::runtime_error("RawData not found");
    }
    return _rawData.at(id);
}

const VgfView &DataManager::getVgfView(DataGraphId id) const {
    if (_vgfViews.find(id) == _vgfViews.end()) {
        throw std::runtime_error("Vgf not found");
    }
    return _vgfViews.at(id);
}

const VulkanImageBarrier &DataManager::getImageBarrier(ImageBarrierId id) const {
    if (_imageBarriers.find(id) == _imageBarriers.end()) {
        throw std::runtime_error("Image Barrier not found");
    }
    return _imageBarriers.at(id);
}

const VulkanTensorBarrier &DataManager::getTensorBarrier(TensorBarrierId id) const {
    if (_tensorBarriers.find(id) == _tensorBarriers.end()) {
        throw std::runtime_error("Tensor Barrier not found");
    }
    return _tensorBarriers.at(id);
}

const VulkanMemoryBarrier &DataManager::getMemoryBarrier(MemoryBarrierId id) const {
    if (_memoryBarriers.find(id) == _memoryBarriers.end()) {
        throw std::runtime_error("Memory Barrier not found");
    }
    return _memoryBarriers.at(id);
}

const VulkanBufferBarrier &DataManager::getBufferBarrier(BufferBarrierId id) const {
    if (_bufferBarriers.find(id) == _bufferBarriers.end()) {
        throw std::runtime_error("Buffer Barrier not found");
    }
    return _bufferBarriers.at(id);
}

} // namespace mlsdk::scenariorunner
