/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "context.hpp"
#include "types.hpp"
#include "utils.hpp"

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
};
} // namespace mlsdk::scenariorunner
