/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resource_manager.hpp"

#include <utility>

namespace mlsdk::scenariorunner {
namespace {
template <typename Id, typename StoredInfo, typename InputInfo>
Id addResource(std::vector<StoredInfo> &resources, InputInfo &&info) {
    const Id id{resources.size()};
    resources.emplace_back(std::forward<InputInfo>(info));
    return id;
}

template <typename Info, typename Id> const Info &getResource(const std::vector<Info> &resources, Id id) {
    return resources.at(id.value());
}
} // namespace

BufferId ResourceManager::addBuffer(const BufferInfo &info) { return addResource<BufferId>(_buffers, info); }

BufferId ResourceManager::addBuffer(BufferInfo &&info) { return addResource<BufferId>(_buffers, std::move(info)); }

ImageId ResourceManager::addImage(const ImageInfo &info) { return addResource<ImageId>(_images, info); }

ImageId ResourceManager::addImage(ImageInfo &&info) { return addResource<ImageId>(_images, std::move(info)); }

TensorId ResourceManager::addTensor(const TensorInfo &info) { return addResource<TensorId>(_tensors, info); }

TensorId ResourceManager::addTensor(TensorInfo &&info) { return addResource<TensorId>(_tensors, std::move(info)); }

ShaderId ResourceManager::addShader(const ShaderInfo &info) { return addResource<ShaderId>(_shaders, info); }

ShaderId ResourceManager::addShader(ShaderInfo &&info) { return addResource<ShaderId>(_shaders, std::move(info)); }

RawDataId ResourceManager::addRawData(const RawDataInfo &info) { return addResource<RawDataId>(_rawData, info); }

RawDataId ResourceManager::addRawData(RawDataInfo &&info) { return addResource<RawDataId>(_rawData, std::move(info)); }

DataGraphId ResourceManager::addDataGraph(const DataGraphInfo &info) {
    return addResource<DataGraphId>(_dataGraphs, info);
}

DataGraphId ResourceManager::addDataGraph(DataGraphInfo &&info) {
    return addResource<DataGraphId>(_dataGraphs, std::move(info));
}

GraphConstantResourceId ResourceManager::addGraphConstant(const GraphConstantInfo &info) {
    return addResource<GraphConstantResourceId>(_graphConstants, info);
}

GraphConstantResourceId ResourceManager::addGraphConstant(GraphConstantInfo &&info) {
    return addResource<GraphConstantResourceId>(_graphConstants, std::move(info));
}

const BufferInfo &ResourceManager::get(BufferId id) const { return getResource(_buffers, id); }

const ImageInfo &ResourceManager::get(ImageId id) const { return getResource(_images, id); }

const TensorInfo &ResourceManager::get(TensorId id) const { return getResource(_tensors, id); }

const ShaderInfo &ResourceManager::get(ShaderId id) const { return getResource(_shaders, id); }

const RawDataInfo &ResourceManager::get(RawDataId id) const { return getResource(_rawData, id); }

const DataGraphInfo &ResourceManager::get(DataGraphId id) const { return getResource(_dataGraphs, id); }

const GraphConstantInfo &ResourceManager::get(GraphConstantResourceId id) const {
    return getResource(_graphConstants, id);
}

} // namespace mlsdk::scenariorunner
