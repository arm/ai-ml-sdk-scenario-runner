/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resource_data.hpp"
#include "resource_id.hpp"

#include <string_view>

namespace mlsdk::scenariorunner {

/// @brief Public interface for executing a built scenario and transferring resource data.
///
/// Implementations own the built resources and pipelines. A caller can upload new input data,
/// execute the scenario, and download output data repeatedly without rebuilding those resources.
class IScenario {
  public:
    virtual ~IScenario() = default;

    /// @brief Execute the scenario one or more times.
    /// @param repeatCount Number of executions; must be greater than zero.
    /// @param dryRun Skip workload execution and output-resource saving.
    virtual void run(int repeatCount = 1, bool dryRun = false) = 0;

    virtual BufferId getBufferId(std::string_view uid) const = 0;
    virtual ImageId getImageId(std::string_view uid) const = 0;
    virtual TensorId getTensorId(std::string_view uid) const = 0;

    virtual void upload(BufferId id, const BufferDataView &data) = 0;
    virtual void upload(ImageId id, const ImageDataView &data) = 0;
    virtual void upload(TensorId id, const TensorDataView &data) = 0;

    virtual BufferData download(BufferId id) const = 0;
    virtual ImageData download(ImageId id) = 0;
    virtual TensorData download(TensorId id) const = 0;
};

} // namespace mlsdk::scenariorunner
