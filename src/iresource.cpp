/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "iresource.hpp"

#include "data_manager.hpp"

namespace mlsdk::scenariorunner {

DataManagerResourceViewer::DataManagerResourceViewer(const DataManager &dataManager, Guid resourceRef)
    : _dataManager(dataManager), _resourceRef(resourceRef) {}

bool DataManagerResourceViewer::hasBuffer() const { return _dataManager.hasBuffer(_resourceRef); }

bool DataManagerResourceViewer::hasImage() const { return _dataManager.hasImage(_resourceRef); }

bool DataManagerResourceViewer::hasTensor() const { return _dataManager.hasTensor(_resourceRef); }

const Buffer &DataManagerResourceViewer::getBuffer() const {
    if (!hasBuffer()) {
        throw std::runtime_error("Identifier does not reference a buffer");
    }
    return _dataManager.getBuffer(_resourceRef);
}

const Image &DataManagerResourceViewer::getImage() const {
    if (!hasImage()) {
        throw std::runtime_error("Identifier does not reference an image");
    }
    return _dataManager.getImage(_resourceRef);
}

const Tensor &DataManagerResourceViewer::getTensor() const {
    if (!hasTensor()) {
        throw std::runtime_error("Identifier does not reference a tensor");
    }
    return _dataManager.getTensor(_resourceRef);
}

} // namespace mlsdk::scenariorunner