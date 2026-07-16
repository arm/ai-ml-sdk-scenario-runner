/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"

#include <iostream>
#include <utility>

namespace mlsdk::scenariorunner {

Buffer::Buffer(BufferInfo bufferInfo)
    : _size(bufferInfo.size), _debugName(std::move(bufferInfo.debugName)), _memoryOffset(bufferInfo.memoryOffset) {}

void Buffer::setup(const Context &ctx, std::shared_ptr<ResourceMemoryManager> memoryManager) {
    _memoryManager = std::move(memoryManager);
    vk::BufferCreateInfo bufferCreateInfo{vk::BufferCreateFlags(),
                                          _size,
                                          vk::BufferUsageFlagBits::eStorageBuffer |
                                              vk::BufferUsageFlagBits::eTransferSrc |
                                              vk::BufferUsageFlagBits::eTransferDst,
                                          vk::SharingMode::eExclusive,
                                          ctx.familyQueueIdx(),
                                          nullptr};

    _buffer = vk::raii::Buffer(ctx.device(), bufferCreateInfo);

    trySetVkRaiiObjectDebugName(ctx, _buffer, _debugName);

    const vk::MemoryRequirements memReqs = _buffer.getMemoryRequirements();
    if (memReqs.alignment != 0 && (_memoryOffset % memReqs.alignment) != 0) {
        throw std::runtime_error("Buffer memory offset for '" + _debugName + "' must be aligned to " +
                                 std::to_string(memReqs.alignment) + " bytes, got " + std::to_string(_memoryOffset));
    }
    _memoryManager->updateMemSize(memReqs.size + _memoryOffset);
    _memoryManager->updateMemType(memReqs.memoryTypeBits);
}

void Buffer::allocateMemory(const Context &ctx) {
    if (!_memoryManager->isInitalized()) {
        _memoryManager->allocateDeviceMemory(ctx, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    _buffer.bindMemory(*_memoryManager->getDeviceMemory(), _memoryOffset);
}

const vk::Buffer &Buffer::buffer() const { return *_buffer; }

uint32_t Buffer::size() const { return _size; }

const std::string &Buffer::debugName() const { return _debugName; }

void Buffer::fillFromDescription(const Context &ctx, const BufferDesc &buffer) const {
    if (buffer.src.has_value()) {
        MemoryMap mapped(buffer.src.value());
        auto dataPtr = vgfutils::numpy::parse(mapped);
        BufferDataView view{dataPtr.ptr, dataPtr.size()};
        upload(ctx, view);
    } else {
        fillZero(ctx);
    }
}

void Buffer::fill(const Context &ctx, const void *ptr, size_t size) const {
    if (size != this->size()) {
        throw std::runtime_error("Buffer::fill: size mismatch");
    }
    void *pDeviceMemory = _memoryManager->mapStagingBufferMemory(_memoryOffset, size);
    std::memcpy(pDeviceMemory, ptr, size);
    _memoryManager->unmapStagingBufferMemory();

    _memoryManager->uploadData(ctx, _memoryOffset, size);
}

void Buffer::fillZero(const Context &ctx) const {
    void *pDeviceMemory = _memoryManager->mapStagingBufferMemory(_memoryOffset, size());
    std::memset(pDeviceMemory, 0, size());
    _memoryManager->unmapStagingBufferMemory();

    _memoryManager->uploadData(ctx, _memoryOffset, size());
}

void Buffer::upload(const Context &ctx, const BufferDataView &data) const {
    if (data.size != this->size()) {
        throw std::runtime_error("Buffer::upload: size mismatch");
    }
    fill(ctx, data.data, data.size);
}

BufferData Buffer::download(const Context &ctx) const {
    _memoryManager->downloadData(ctx, _memoryOffset, size());
    ScopeExit<void()> onScopeExitRun([&] { _memoryManager->unmapStagingBufferMemory(); });

    BufferData bd;
    bd.data.resize(size());
    const auto *mapped = static_cast<const char *>(_memoryManager->mapStagingBufferMemory(_memoryOffset, size()));
    std::memcpy(bd.data.data(), mapped, bd.data.size());
    return bd;
}

void Buffer::store(const Context &ctx, const std::string &filename) const {
    const auto bufferContents = download(ctx);
    vgfutils::numpy::DataPtr data(bufferContents.data.data(), {static_cast<int64_t>(bufferContents.data.size())},
                                  vgfutils::numpy::DType('i', 1));
    vgfutils::numpy::write(filename, data);
}

} // namespace mlsdk::scenariorunner
