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
        const auto memoryFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
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

    void uploadData(const Context &ctx, vk::DeviceSize offset, vk::DeviceSize size) const {
        // Create device buffer to copy data to
        vk::BufferCreateInfo bufferCreateInfo{
            vk::BufferCreateFlags(), size,   vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive,
            ctx.familyQueueIdx(),    nullptr};
        vk::raii::Buffer deviceBuffer = vk::raii::Buffer(ctx.device(), bufferCreateInfo);
        deviceBuffer.bindMemory(*_deviceMemory, offset);

        // Create host visible buffer to copy data from
        vk::BufferCreateInfo stagingBufferCreateInfo{
            vk::BufferCreateFlags(), size,   vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
            ctx.familyQueueIdx(),    nullptr};
        vk::raii::Buffer stagingBuffer = vk::raii::Buffer(ctx.device(), stagingBufferCreateInfo);
        stagingBuffer.bindMemory(*_stagingBufferDeviceMemory, offset);

        // Copy data from staging buffer to device local buffer
        const vk::CommandPoolCreateInfo cmdPoolCreateInfo({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                                          ctx.familyQueueIdx());
        auto cmdPool = ctx.device().createCommandPool(cmdPoolCreateInfo);
        const vk::CommandBufferAllocateInfo cmdBufferAllocInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::raii::CommandBuffer cmdBuffer = std::move(ctx.device().allocateCommandBuffers(cmdBufferAllocInfo).front());
        const vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdBuffer.begin(CmdBufferBeginInfo);
        vk::BufferCopy copyRegion{0, 0, size};
        cmdBuffer.copyBuffer(stagingBuffer, deviceBuffer, copyRegion);
        cmdBuffer.end();

        vk::SubmitInfo submitInfo({}, {}, *cmdBuffer);
        auto queue = ctx.device().getQueue(ctx.familyQueueIdx(), 0);
        auto fence = ctx.device().createFence({});
        queue.submit(submitInfo, *fence);
        const auto timeout = WAIT_FOR_FENCE_TIMEOUT;
        auto res = ctx.device().waitForFences({*fence}, true, timeout);
        if (res != vk::Result::eSuccess) {
            throw std::runtime_error("Error while waiting for fence.");
        }
    }

    void downloadData(const Context &ctx, vk::DeviceSize offset, vk::DeviceSize size) const {
        // Create device buffer to copy data from
        vk::BufferCreateInfo bufferCreateInfo{
            vk::BufferCreateFlags(), size,   vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
            ctx.familyQueueIdx(),    nullptr};
        vk::raii::Buffer deviceBuffer = vk::raii::Buffer(ctx.device(), bufferCreateInfo);
        deviceBuffer.bindMemory(*_deviceMemory, offset);

        // Create host visible buffer to copy data to
        vk::BufferCreateInfo stagingBufferCreateInfo{
            vk::BufferCreateFlags(), size,   vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive,
            ctx.familyQueueIdx(),    nullptr};
        vk::raii::Buffer stagingBuffer = vk::raii::Buffer(ctx.device(), stagingBufferCreateInfo);
        stagingBuffer.bindMemory(*_stagingBufferDeviceMemory, offset);

        // Copy data from device local buffer to staging buffer
        const vk::CommandPoolCreateInfo cmdPoolCreateInfo({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                                          ctx.familyQueueIdx());
        auto cmdPool = ctx.device().createCommandPool(cmdPoolCreateInfo);
        const vk::CommandBufferAllocateInfo cmdBufferAllocInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::raii::CommandBuffer cmdBuffer = std::move(ctx.device().allocateCommandBuffers(cmdBufferAllocInfo).front());
        const vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdBuffer.begin(CmdBufferBeginInfo);
        vk::BufferCopy copyRegion{0, 0, size};
        cmdBuffer.copyBuffer(deviceBuffer, stagingBuffer, copyRegion);
        cmdBuffer.end();

        vk::SubmitInfo submitInfo({}, {}, *cmdBuffer);
        auto queue = ctx.device().getQueue(ctx.familyQueueIdx(), 0);
        auto fence = ctx.device().createFence({});
        queue.submit(submitInfo, *fence);
        const auto timeout = WAIT_FOR_FENCE_TIMEOUT;
        auto res = ctx.device().waitForFences({*fence}, true, timeout);
        if (res != vk::Result::eSuccess) {
            throw std::runtime_error("Error while waiting for fence.");
        }
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
