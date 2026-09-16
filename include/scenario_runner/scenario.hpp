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
class Scenario {
  public:
    virtual ~Scenario() = default;

    /// @brief Execute the scenario once.
    ///
    /// This is equivalent to run(1, false).
    virtual void run() = 0;

    /// @brief Execute the scenario one or more times.
    /// @param repeatCount Number of executions; must be greater than zero.
    /// @param dryRun Skip workload execution and output-resource saving.
    /// @throws std::invalid_argument If repeatCount is not positive.
    virtual void run(int repeatCount, bool dryRun) = 0;

    /// @brief Get a buffer resource ID from its UID.
    /// @param uid UID declared by the JSON scenario.
    /// @return The corresponding buffer ID.
    /// @note This lookup is available for scenarios created by ScenarioJsonFactory.
    /// @throws std::runtime_error If the UID is unknown or identifies another resource type.
    virtual BufferId getBufferId(std::string_view uid) const = 0;

    /// @brief Get an image resource ID from its UID.
    /// @param uid UID declared by the JSON scenario.
    /// @return The corresponding image ID.
    /// @note This lookup is available for scenarios created by ScenarioJsonFactory.
    /// @throws std::runtime_error If the UID is unknown or identifies another resource type.
    virtual ImageId getImageId(std::string_view uid) const = 0;

    /// @brief Get a tensor resource ID from its UID.
    /// @param uid UID declared by the JSON scenario.
    /// @return The corresponding tensor ID.
    /// @note This lookup is available for scenarios created by ScenarioJsonFactory.
    /// @throws std::runtime_error If the UID is unknown or identifies another resource type.
    virtual TensorId getTensorId(std::string_view uid) const = 0;

    /// @brief Upload data to an existing buffer resource.
    /// @param id Buffer registered in this scenario.
    /// @param data Non-owning view of the bytes to upload.
    /// @note The data is copied before this function returns.
    /// @throws std::runtime_error If the ID is invalid or the data size does not match the resource.
    virtual void upload(BufferId id, const BufferDataView &data) = 0;

    /// @brief Upload data to an existing image resource.
    /// @param id Image registered in this scenario.
    /// @param data Non-owning view of the image bytes and validation metadata.
    /// @note The data is copied before this function returns.
    /// @throws std::runtime_error If the ID or supplied resource metadata is invalid.
    virtual void upload(ImageId id, const ImageDataView &data) = 0;

    /// @brief Upload data to an existing tensor resource.
    /// @param id Tensor registered in this scenario.
    /// @param data Non-owning view of the tensor bytes and validation metadata.
    /// @note The data is copied before this function returns.
    /// @throws std::runtime_error If the ID or supplied resource metadata is invalid.
    virtual void upload(TensorId id, const TensorDataView &data) = 0;

    /// @brief Download data from an existing buffer resource.
    /// @param id Buffer registered in this scenario.
    /// @return An owning copy of the buffer contents.
    /// @throws std::runtime_error If the ID is invalid.
    virtual BufferData download(BufferId id) const = 0;

    /// @brief Download data from an existing image resource.
    /// @param id Image registered in this scenario.
    /// @return An owning copy of the base image mip and its metadata.
    /// @throws std::runtime_error If the ID is invalid.
    virtual ImageData download(ImageId id) = 0;

    /// @brief Download data from an existing tensor resource.
    /// @param id Tensor registered in this scenario.
    /// @return An owning copy of the tensor contents and its metadata.
    /// @throws std::runtime_error If the ID is invalid.
    virtual TensorData download(TensorId id) const = 0;
};

} // namespace mlsdk::scenariorunner
