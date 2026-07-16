/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resource_manager.hpp"

#include <utility>

namespace mlsdk::scenariorunner {
namespace {
template <typename Id, typename Info> Id addResource(std::vector<Info> &resources, Info info) {
    const Id id{resources.size()};
    resources.push_back(std::move(info));
    return id;
}

template <typename Info, typename Id> const Info &getResource(const std::vector<Info> &resources, Id id) {
    return resources.at(id.value());
}
} // namespace

BufferId ResourceManager::addBuffer(BufferInfo info) { return addResource<BufferId>(_buffers, std::move(info)); }

ImageId ResourceManager::addImage(ImageInfo info) { return addResource<ImageId>(_images, std::move(info)); }

TensorId ResourceManager::addTensor(TensorInfo info) { return addResource<TensorId>(_tensors, std::move(info)); }

ShaderId ResourceManager::addShader(ShaderInfo info) { return addResource<ShaderId>(_shaders, std::move(info)); }

const BufferInfo &ResourceManager::get(BufferId id) const { return getResource(_buffers, id); }

const ImageInfo &ResourceManager::get(ImageId id) const { return getResource(_images, id); }

const TensorInfo &ResourceManager::get(TensorId id) const { return getResource(_tensors, id); }

const ShaderInfo &ResourceManager::get(ShaderId id) const { return getResource(_shaders, id); }

} // namespace mlsdk::scenariorunner
