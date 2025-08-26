/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"

namespace mlsdk::scenariorunner {

Buffer::Buffer(const Context &ctx, const BufferInfo &bufferInfo, std::shared_ptr<ResourceMemoryManager> memoryManager)
    : _buffer(nullptr), _size(bufferInfo.size), _debugName(bufferInfo.debugName),
      _memoryManager(std::move(memoryManager)), _memoryOffset(bufferInfo.memoryOffset) {
    vk::BufferCreateInfo bufferCreateInfo{
        vk::BufferCreateFlags(), _size,  vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive,
        ctx.familyQueueIdx(),    nullptr};

    _buffer = vk::raii::Buffer(ctx.device(), bufferCreateInfo);

    trySetVkRaiiObjectDebugName(ctx, _buffer, _debugName);

    const vk::MemoryRequirements memReqs = _buffer.getMemoryRequirements();
    _memoryManager->updateMemSize(memReqs.size + _memoryOffset);
    _memoryManager->updateMemType(memReqs.memoryTypeBits);
}

void Buffer::allocateMemory(const Context &ctx) {
    if (!_memoryManager->isInitalized()) {
        _memoryManager->allocateDeviceMemory(ctx, vk::MemoryPropertyFlagBits::eHostVisible);
    }

    _buffer.bindMemory(*_memoryManager->getDeviceMemory(), _memoryOffset);
}

const vk::Buffer &Buffer::buffer() const { return *_buffer; }

uint32_t Buffer::size() const { return _size; }

const std::string &Buffer::debugName() const { return _debugName; }

void *Buffer::map() {
    if (!_memoryManager->isInitalized()) {
        throw std::runtime_error("Uninitialized MemoryManager for Buffer");
    }
    return _memoryManager->getDeviceMemory().mapMemory(_memoryOffset, _size);
}

void Buffer::unmap() {
    if (!_memoryManager->isInitalized()) {
        throw std::runtime_error("Uninitialized MemoryManager for Buffer");
    }
    _memoryManager->getDeviceMemory().unmapMemory();
}

void Buffer::fill(const void *ptr, size_t size) {
    if (size != this->size()) {
        throw std::runtime_error("Buffer::fill: size mismatch");
    }
    void *pDeviceMemory = map();
    std::memcpy(pDeviceMemory, ptr, size);
    unmap();
}

void Buffer::fillZero() {
    void *pDeviceMemory = map();
    std::memset(pDeviceMemory, 0, size());
    unmap();
}

void Buffer::store(Context &, const std::string &filename) {
    ScopeExit<void()> on_scope_exit_run([&] { unmap(); });
    mlsdk::numpy::data_ptr data(reinterpret_cast<const char *>(map()), {size()}, mlsdk::numpy::dtype('i', 1));
    mlsdk::numpy::write(filename, data);
}

} // namespace mlsdk::scenariorunner
