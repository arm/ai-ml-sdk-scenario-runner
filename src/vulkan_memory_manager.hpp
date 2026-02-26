/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "context.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <limits>

namespace mlsdk::scenariorunner {

class ResourceMemoryManager {
  public:
    bool isInitalized() const { return _initalized; }

    void allocateDeviceMemory(const Context &ctx, vk::MemoryPropertyFlags flags) {
        if (_memSize == 0) {
            throw std::runtime_error("Allocated memory size must be non-zero");
        }
        const vk::MemoryAllocateInfo memoryAllocateInfo(_memSize, findMemoryIdx(ctx, _memType, flags));
        _deviceMemory = vk::raii::DeviceMemory(ctx.device(), memoryAllocateInfo);

        // Create the staging buffer
        vk::BufferCreateInfo bufferCreateInfo{vk::BufferCreateFlags(),
                                              _memSize,
                                              vk::BufferUsageFlagBits::eTransferSrc |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              vk::SharingMode::eExclusive,
                                              ctx.familyQueueIdx(),
                                              nullptr};

        _stagingBuffer = vk::raii::Buffer(ctx.device(), bufferCreateInfo);
        const vk::MemoryRequirements memReqs = _stagingBuffer.getMemoryRequirements();
        const auto memoryFlags = vk::MemoryPropertyFlagBits::eHostVisible;
        const uint32_t memTypeIndex = findMemoryIdx(ctx, memReqs.memoryTypeBits, memoryFlags);
        if (memTypeIndex == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Cannot find a memory type with the required properties");
        }

        const vk::MemoryAllocateInfo memAllocInfo(memReqs.size, memTypeIndex);
        _stagingBufferDeviceMemory = vk::raii::DeviceMemory(ctx.device(), memAllocInfo);
        _stagingBuffer.bindMemory(*_stagingBufferDeviceMemory, 0);

        _initalized = true;
    }

    void updateMemSize(vk::DeviceSize newSize) {
        if (newSize > _memSize) {
            _memSize = newSize;
        }
    }

    void updateSubResourceOffset(vk::DeviceSize offset) { _subRecOffset = offset; }

    void updateSubResourceRowPitch(vk::DeviceSize rowPitch) { _rowPitch = rowPitch; }

    void updateSubResourceDepthPitch(vk::DeviceSize depthPitch) { _depthPitch = depthPitch; }

    void updateSubResourceArrayPitch(vk::DeviceSize arrayPitch) { _arrayPitch = arrayPitch; }

    void updateFormat(vk::Format format) { _format = format; }

    void updateImageType(vk::ImageType imType) { _imType = imType; }

    void updateMemType(uint32_t type) { _memType &= type; }

    vk::DeviceSize getMemSize() const { return _memSize; }

    vk::DeviceSize getSubresourceOffset() const { return _subRecOffset; }

    vk::DeviceSize getSubResourceRowPitch() const { return _rowPitch; }

    vk::DeviceSize getSubResourceDepthPitch() const { return _depthPitch; }

    vk::DeviceSize getSubResourceArrayPitch() const { return _arrayPitch; }

    vk::Format getFormat() const { return _format; }

    vk::ImageType getImageType() const { return _imType; }

    uint32_t getMemType() const { return _memType; }

    const vk::raii::DeviceMemory &getDeviceMemory() const { return _deviceMemory; }

    const vk::raii::Buffer &getStagingBuffer() const { return _stagingBuffer; }

    void *mapStagingBufferMemory(uint64_t offset, uint64_t size) const {
        if (!isInitalized()) {
            throw std::runtime_error("Staging buffer memory has not been allocated");
        }
        if (offset + size > _memSize) {
            throw std::runtime_error("Attempt to map staging buffer memory out of bounds");
        }
        return _stagingBufferDeviceMemory.mapMemory(offset, size);
    }

    void unmapStagingBufferMemory() const {
        if (!isInitalized()) {
            throw std::runtime_error("Staging buffer memory has not been allocated");
        }
        _stagingBufferDeviceMemory.unmapMemory();
    }

  private:
    vk::DeviceSize _memSize{0};
    vk::DeviceSize _subRecOffset{0};
    vk::DeviceSize _rowPitch{0};
    vk::DeviceSize _depthPitch{0};
    vk::DeviceSize _arrayPitch{0};
    vk::ImageType _imType{vk::ImageType::e2D};
    vk::Format _format{vk::Format::eUndefined};
    uint32_t _memType{UINT32_MAX};
    vk::raii::DeviceMemory _deviceMemory{nullptr};
    bool _initalized{false};
    vk::raii::Buffer _stagingBuffer{nullptr};
    vk::raii::DeviceMemory _stagingBufferDeviceMemory{nullptr};
};
} // namespace mlsdk::scenariorunner
