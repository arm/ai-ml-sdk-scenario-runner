/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "barrier.hpp"
#include "buffer.hpp"
#include "context.hpp"
#include "image.hpp"
#include "numpy.hpp"
#include "raw_data.hpp"
#include "resource_desc.hpp"
#include "tensor.hpp"
#include "vgf_view.hpp"

#include <unordered_map>

namespace mlsdk::scenariorunner {

class DataManager {
  public:
    explicit DataManager(Context &ctx);

    void createBuffer(Guid guid, const BufferInfo &info, const std::vector<char> &values);
    void createBuffer(Guid guid, const BufferInfo &info, const mlsdk::numpy::data_ptr &dataPtr);
    void createZeroedBuffer(Guid guid, const BufferInfo &info);
    void createTensor(Guid guid, const TensorInfo &info, std::optional<Guid> aliasTarget = std::nullopt);
    void createImage(Guid guid, const ImageInfo &info);
    void createRawData(Guid guid, const std::string &debugName, const std::string &src);
    void createVgfView(Guid guid, const DataGraphDesc &desc);
    void createImageBarrier(Guid guid, const ImageBarrierData &data);
    void createTensorBarrier(Guid guid, const TensorBarrierData &data);
    void createMemoryBarrier(Guid guid, const MemoryBarrierData &data);
    void createBufferBarrier(Guid guid, const BufferBarrierData &data);

    bool hasBuffer(Guid guid) const;
    bool hasTensor(Guid guid) const;
    bool hasImage(Guid guid) const;
    bool hasRawData(Guid guid) const;
    bool hasImageBarrier(Guid guid) const;
    bool hasTensorBarrier(Guid guid) const;
    bool hasMemoryBarrier(Guid guid) const;
    bool hasBufferBarrier(Guid guid) const;

    Buffer &getBufferMut(const Guid &guid);
    Tensor &getTensorMut(const Guid &guid);
    Image &getImageMut(const Guid &guid);

    const Buffer &getBuffer(const Guid &guid) const;
    const Tensor &getTensor(const Guid &guid) const;
    const Image &getImage(const Guid &guid) const;
    const RawData &getRawData(const Guid &guid) const;
    const VgfView &getVgfView(const Guid &guid) const;
    const VulkanImageBarrier &getImageBarrier(const Guid &guid) const;
    const VulkanMemoryBarrier &getMemoryBarrier(const Guid &guid) const;
    const VulkanBufferBarrier &getBufferBarrier(const Guid &guid) const;
    const VulkanTensorBarrier &getTensorBarrier(const Guid &guid) const;

    Resource &getResourceMut(const ResourceDesc &resourceDesc);
    void storeResource(const ResourceDesc &resourceDesc);

    uint32_t numBuffers() const;
    uint32_t numTensors() const;
    uint32_t numImages() const;

    vk::DescriptorType getResourceDescriptorType(const Guid &guid) const;

    void createMemoryManager(Guid guid);
    std::shared_ptr<ResourceMemoryManager> getMemoryManager(const Guid &guid) const;

  private:
    Context &_ctx;

    std::unordered_map<Guid, Buffer> _buffers{};
    std::unordered_map<Guid, Tensor> _tensors{};
    std::unordered_map<Guid, Image> _images{};
    std::unordered_map<Guid, RawData> _rawData{};
    std::unordered_map<Guid, VgfView> _vgfViews{};
    std::unordered_map<Guid, VulkanImageBarrier> _imageBarriers{};
    std::unordered_map<Guid, VulkanMemoryBarrier> _memoryBarriers{};
    std::unordered_map<Guid, VulkanBufferBarrier> _bufferBarriers{};
    std::unordered_map<Guid, VulkanTensorBarrier> _tensorBarriers{};

    std::unordered_map<Guid, std::shared_ptr<ResourceMemoryManager>> _memoryManagers{};
};
} // namespace mlsdk::scenariorunner
