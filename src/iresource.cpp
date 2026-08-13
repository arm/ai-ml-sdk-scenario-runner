/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "iresource.hpp"

#include "data_manager.hpp"

namespace mlsdk::scenariorunner {

DataManagerResourceViewer::DataManagerResourceViewer(const DataManager &dataManager, MemoryResourceId resource)
    : _dataManager(dataManager), _resource(resource) {}

bool DataManagerResourceViewer::hasBuffer() const {
    const auto *id = std::get_if<BufferId>(&_resource);
    return id != nullptr && _dataManager.hasBuffer(*id);
}

bool DataManagerResourceViewer::hasImage() const {
    const auto *id = std::get_if<ImageId>(&_resource);
    return id != nullptr && _dataManager.hasImage(*id);
}

bool DataManagerResourceViewer::hasTensor() const {
    const auto *id = std::get_if<TensorId>(&_resource);
    return id != nullptr && _dataManager.hasTensor(*id);
}

const Buffer &DataManagerResourceViewer::getBuffer() const {
    if (!hasBuffer()) {
        throw std::runtime_error("Identifier does not reference a buffer");
    }
    return _dataManager.getBuffer(std::get<BufferId>(_resource));
}

const Image &DataManagerResourceViewer::getImage() const {
    if (!hasImage()) {
        throw std::runtime_error("Identifier does not reference an image");
    }
    return _dataManager.getImage(std::get<ImageId>(_resource));
}

const Tensor &DataManagerResourceViewer::getTensor() const {
    if (!hasTensor()) {
        throw std::runtime_error("Identifier does not reference a tensor");
    }
    return _dataManager.getTensor(std::get<TensorId>(_resource));
}

} // namespace mlsdk::scenariorunner
