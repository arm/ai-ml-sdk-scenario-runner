/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "buffer.hpp"
#include "tensor.hpp"

namespace mlsdk::scenariorunner {

class IResourceCreator {
  public:
    virtual ~IResourceCreator() = default;

    // Create buffer resource and setup and allocate memory in buffer
    virtual void createBuffer(Guid guid, const BufferInfo &info) = 0;
    virtual void createTensor(Guid guid, const TensorInfo &info) = 0;
};

} // namespace mlsdk::scenariorunner
