/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resource_id.hpp"
#include "types.hpp"

#include <vector>

namespace mlsdk::scenariorunner {

template <typename Id, typename Info> class ResourceEntries {
  public:
    struct Entry {
        Id id;
        const Info &info;
    };

    class Iterator {
      public:
        Iterator(const std::vector<Info> &resources, size_t index) : _resources{resources}, _index{index} {}

        Entry operator*() const { return {Id{_index}, _resources[_index]}; }
        Iterator &operator++() {
            ++_index;
            return *this;
        }
        bool operator!=(const Iterator &other) const { return _index != other._index; }

      private:
        const std::vector<Info> &_resources;
        size_t _index;
    };

    explicit ResourceEntries(const std::vector<Info> &resources) : _resources{resources} {}

    Iterator begin() const { return {_resources, 0}; }
    Iterator end() const { return {_resources, _resources.size()}; }

  private:
    const std::vector<Info> &_resources;
};

class ResourceManager {
  public:
    BufferId addBuffer(const BufferInfo &info);
    BufferId addBuffer(BufferInfo &&info);
    ImageId addImage(const ImageInfo &info);
    ImageId addImage(ImageInfo &&info);
    TensorId addTensor(const TensorInfo &info);
    TensorId addTensor(TensorInfo &&info);
    ShaderId addShader(const ShaderInfo &info);
    ShaderId addShader(ShaderInfo &&info);
    RawDataId addRawData(const RawDataInfo &info);
    RawDataId addRawData(RawDataInfo &&info);
    DataGraphId addDataGraph(const DataGraphInfo &info);
    DataGraphId addDataGraph(DataGraphInfo &&info);
    GraphConstantResourceId addGraphConstant(const GraphConstantInfo &info);
    GraphConstantResourceId addGraphConstant(GraphConstantInfo &&info);
    ImageBarrierId addImageBarrier(const ImageBarrierInfo &info);
    ImageBarrierId addImageBarrier(ImageBarrierInfo &&info);
    BufferBarrierId addBufferBarrier(const BufferBarrierInfo &info);
    BufferBarrierId addBufferBarrier(BufferBarrierInfo &&info);
    TensorBarrierId addTensorBarrier(const TensorBarrierInfo &info);
    TensorBarrierId addTensorBarrier(TensorBarrierInfo &&info);
    MemoryBarrierId addMemoryBarrier(const MemoryBarrierInfo &info);
    MemoryBarrierId addMemoryBarrier(MemoryBarrierInfo &&info);

    const BufferInfo &get(BufferId id) const;
    const ImageInfo &get(ImageId id) const;
    const TensorInfo &get(TensorId id) const;
    const ShaderInfo &get(ShaderId id) const;
    const RawDataInfo &get(RawDataId id) const;
    const DataGraphInfo &get(DataGraphId id) const;
    const GraphConstantInfo &get(GraphConstantResourceId id) const;
    const ImageBarrierInfo &get(ImageBarrierId id) const;
    const BufferBarrierInfo &get(BufferBarrierId id) const;
    const TensorBarrierInfo &get(TensorBarrierId id) const;
    const MemoryBarrierInfo &get(MemoryBarrierId id) const;

    ResourceEntries<BufferId, BufferInfo> buffers() const { return ResourceEntries<BufferId, BufferInfo>{_buffers}; }
    ResourceEntries<ImageId, ImageInfo> images() const { return ResourceEntries<ImageId, ImageInfo>{_images}; }
    ResourceEntries<TensorId, TensorInfo> tensors() const { return ResourceEntries<TensorId, TensorInfo>{_tensors}; }
    ResourceEntries<ShaderId, ShaderInfo> shaders() const { return ResourceEntries<ShaderId, ShaderInfo>{_shaders}; }
    ResourceEntries<RawDataId, RawDataInfo> rawData() const {
        return ResourceEntries<RawDataId, RawDataInfo>{_rawData};
    }
    ResourceEntries<DataGraphId, DataGraphInfo> dataGraphs() const {
        return ResourceEntries<DataGraphId, DataGraphInfo>{_dataGraphs};
    }
    ResourceEntries<GraphConstantResourceId, GraphConstantInfo> graphConstants() const {
        return ResourceEntries<GraphConstantResourceId, GraphConstantInfo>{_graphConstants};
    }
    ResourceEntries<ImageBarrierId, ImageBarrierInfo> imageBarriers() const {
        return ResourceEntries<ImageBarrierId, ImageBarrierInfo>{_imageBarriers};
    }
    ResourceEntries<BufferBarrierId, BufferBarrierInfo> bufferBarriers() const {
        return ResourceEntries<BufferBarrierId, BufferBarrierInfo>{_bufferBarriers};
    }
    ResourceEntries<TensorBarrierId, TensorBarrierInfo> tensorBarriers() const {
        return ResourceEntries<TensorBarrierId, TensorBarrierInfo>{_tensorBarriers};
    }
    ResourceEntries<MemoryBarrierId, MemoryBarrierInfo> memoryBarriers() const {
        return ResourceEntries<MemoryBarrierId, MemoryBarrierInfo>{_memoryBarriers};
    }

  private:
    std::vector<BufferInfo> _buffers;
    std::vector<ImageInfo> _images;
    std::vector<TensorInfo> _tensors;
    std::vector<ShaderInfo> _shaders;
    std::vector<RawDataInfo> _rawData;
    std::vector<DataGraphInfo> _dataGraphs;
    std::vector<GraphConstantInfo> _graphConstants;
    std::vector<ImageBarrierInfo> _imageBarriers;
    std::vector<BufferBarrierInfo> _bufferBarriers;
    std::vector<TensorBarrierInfo> _tensorBarriers;
    std::vector<MemoryBarrierInfo> _memoryBarriers;
};

} // namespace mlsdk::scenariorunner
