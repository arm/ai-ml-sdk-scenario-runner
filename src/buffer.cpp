/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"
#include <iostream>

namespace mlsdk::scenariorunner {

Buffer::Buffer(const BufferInfo &bufferInfo)
    : _size(bufferInfo.size), _debugName(bufferInfo.debugName), _memoryOffset(bufferInfo.memoryOffset) {}

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
        fill(ctx, dataPtr.ptr, dataPtr.size());
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

void Buffer::store(const Context &ctx, const std::string &filename) const {
    _memoryManager->downloadData(ctx, _memoryOffset, size());

    ScopeExit<void()> on_scope_exit_run([&] { _memoryManager->unmapStagingBufferMemory(); });
    vgfutils::numpy::DataPtr data(
        reinterpret_cast<const char *>(_memoryManager->mapStagingBufferMemory(_memoryOffset, size())), {size()},
        vgfutils::numpy::DType('i', 1));
    vgfutils::numpy::write(filename, data);
}

} // namespace mlsdk::scenariorunner
