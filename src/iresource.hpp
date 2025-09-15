/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "buffer.hpp"
#include "guid.hpp"
#include "image.hpp"
#include "tensor.hpp"

namespace mlsdk::scenariorunner {
class DataManager;

class IResourceCreator {
  public:
    virtual ~IResourceCreator() = default;

    // Create buffer resource and setup and allocate memory in buffer
    virtual void createBuffer(Guid guid, const BufferInfo &info) = 0;
    virtual void createTensor(Guid guid, const TensorInfo &info) = 0;
};

/// @brief Interface for accessing resources in an identifier agnostic way. Derived class handle the resource
/// identification.
class IResourceViewer {
  public:
    virtual ~IResourceViewer() = default;

    virtual bool hasBuffer() const = 0;
    virtual bool hasImage() const = 0;
    virtual bool hasTensor() const = 0;

    virtual const Buffer &getBuffer() const = 0;
    virtual const Image &getImage() const = 0;
    virtual const Tensor &getTensor() const = 0;
};

/// @brief Base implementation for DataManager
class DataManagerResourceViewer : public IResourceViewer {
  public:
    DataManagerResourceViewer(const DataManager &dataManager, Guid resourceRef);

    bool hasBuffer() const override;
    bool hasImage() const override;
    bool hasTensor() const override;

    const Buffer &getBuffer() const override;
    const Image &getImage() const override;
    const Tensor &getTensor() const override;

  protected:
    const DataManager &_dataManager;
    Guid _resourceRef;
};

} // namespace mlsdk::scenariorunner
