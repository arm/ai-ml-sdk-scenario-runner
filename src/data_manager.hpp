/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "barrier.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "raw_data.hpp"
#include "resource_id.hpp"
#include "tensor.hpp"
#include "vgf_view.hpp"

#include <unordered_map>

namespace mlsdk::scenariorunner {

class DataManager {
  public:
    void createBuffer(BufferId id, const BufferInfo &info);
    void createBuffer(BufferId id, BufferInfo &&info);
    void createTensor(TensorId id, const TensorInfo &info);
    void createTensor(TensorId id, TensorInfo &&info);
    void createImage(ImageId id, const ImageInfo &info);
    void createImage(ImageId id, ImageInfo &&info);
    void createRawData(RawDataId id, const RawDataInfo &info);
    void createVgfView(DataGraphId id, const DataGraphInfo &info);
    void createImageBarrier(Guid guid, const ImageBarrierData &data);
    void createTensorBarrier(Guid guid, const TensorBarrierData &data);
    void createMemoryBarrier(Guid guid, const MemoryBarrierData &data);
    void createBufferBarrier(Guid guid, const BufferBarrierData &data);

    bool hasBuffer(BufferId id) const;
    bool hasTensor(TensorId id) const;
    bool hasImage(ImageId id) const;
    bool hasRawData(RawDataId id) const;
    bool hasImageBarrier(Guid guid) const;
    bool hasTensorBarrier(Guid guid) const;
    bool hasMemoryBarrier(Guid guid) const;
    bool hasBufferBarrier(Guid guid) const;

    Buffer &getBufferMut(BufferId id);
    Tensor &getTensorMut(TensorId id);
    Image &getImageMut(ImageId id);

    const Buffer &getBuffer(BufferId id) const;
    const Tensor &getTensor(TensorId id) const;
    const Image &getImage(ImageId id) const;
    const RawData &getRawData(RawDataId id) const;
    const VgfView &getVgfView(DataGraphId id) const;
    const VulkanImageBarrier &getImageBarrier(const Guid &guid) const;
    const VulkanMemoryBarrier &getMemoryBarrier(const Guid &guid) const;
    const VulkanBufferBarrier &getBufferBarrier(const Guid &guid) const;
    const VulkanTensorBarrier &getTensorBarrier(const Guid &guid) const;

    uint32_t numBuffers() const;
    uint32_t numTensors() const;
    uint32_t numImages() const;

  private:
    std::unordered_map<BufferId, Buffer> _buffers;
    std::unordered_map<TensorId, Tensor> _tensors;
    std::unordered_map<ImageId, Image> _images;
    std::unordered_map<RawDataId, RawData> _rawData;
    std::unordered_map<DataGraphId, VgfView> _vgfViews;
    std::unordered_map<Guid, VulkanImageBarrier> _imageBarriers;
    std::unordered_map<Guid, VulkanMemoryBarrier> _memoryBarriers;
    std::unordered_map<Guid, VulkanBufferBarrier> _bufferBarriers;
    std::unordered_map<Guid, VulkanTensorBarrier> _tensorBarriers;
};
} // namespace mlsdk::scenariorunner
