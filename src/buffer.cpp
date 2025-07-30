/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"

namespace mlsdk::scenariorunner {

Buffer::Buffer(const Context &ctx, const std::string &debugName, uint32_t size)
    : _buffer(nullptr), _deviceMemory(nullptr), _size(size), _debugName(debugName) {
    const vk::BufferCreateInfo bufferCreateInfo{
        vk::BufferCreateFlags(), size,   vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive,
        ctx.familyQueueIdx(),    nullptr};

    _buffer = vk::raii::Buffer(ctx.device(), bufferCreateInfo);

    trySetVkRaiiObjectDebugName(ctx, _buffer, debugName);

    const vk::MemoryRequirements memReqs = _buffer.getMemoryRequirements();
    const auto flags = vk::MemoryPropertyFlagBits::eHostVisible;
    const uint32_t memTypeIndex = findMemoryIdx(ctx, memReqs.memoryTypeBits, flags);
    if (memTypeIndex == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Cannot find a memory type with the required properties");
    }

    const vk::MemoryAllocateInfo memAllocInfo(memReqs.size, memTypeIndex);
    _deviceMemory = vk::raii::DeviceMemory(ctx.device(), memAllocInfo);

    _buffer.bindMemory(*_deviceMemory, 0);
}

const vk::Buffer &Buffer::buffer() const { return *_buffer; }

uint32_t Buffer::size() const { return _size; }

const std::string &Buffer::debugName() const { return _debugName; }

void *Buffer::map() { return _deviceMemory.mapMemory(0, this->size()); }

void Buffer::unmap() { _deviceMemory.unmapMemory(); }

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
