/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resource_id.hpp"
#include "types.hpp"

#include <vector>

namespace mlsdk::scenariorunner {

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

    const BufferInfo &get(BufferId id) const;
    const ImageInfo &get(ImageId id) const;
    const TensorInfo &get(TensorId id) const;
    const ShaderInfo &get(ShaderId id) const;
    const RawDataInfo &get(RawDataId id) const;
    const DataGraphInfo &get(DataGraphId id) const;

  private:
    std::vector<BufferInfo> _buffers;
    std::vector<ImageInfo> _images;
    std::vector<TensorInfo> _tensors;
    std::vector<ShaderInfo> _shaders;
    std::vector<RawDataInfo> _rawData;
    std::vector<DataGraphInfo> _dataGraphs;
};

} // namespace mlsdk::scenariorunner
