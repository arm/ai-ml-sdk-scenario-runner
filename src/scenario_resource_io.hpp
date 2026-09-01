/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "resource_data.hpp"
#include "types.hpp"

#include <optional>
#include <string>
#include <variant>

namespace mlsdk::scenariorunner::detail {

struct InitializationBase {
    explicit InitializationBase(std::string debugName) : debugName{std::move(debugName)} {}
    std::string debugName;
};

struct BufferInitialization : InitializationBase {
    BufferInitialization(BufferId id, BufferData data, std::string debugName)
        : InitializationBase{std::move(debugName)}, id{id}, data{std::move(data)} {}
    BufferId id;
    BufferData data;
};

struct ImageInitialization : InitializationBase {
    ImageInitialization(ImageId id, std::optional<ImageData> data, std::string debugName)
        : InitializationBase{std::move(debugName)}, id{id}, data{std::move(data)} {}
    ImageId id;
    std::optional<ImageData> data;
};

struct TensorInitialization : InitializationBase {
    TensorInitialization(TensorId id, TensorData data, std::string debugName)
        : InitializationBase{std::move(debugName)}, id{id}, data{std::move(data)} {}
    TensorId id;
    TensorData data;
};

using ResourceInitialization = std::variant<BufferInitialization, ImageInitialization, TensorInitialization>;

struct ResourceOutput {
    TypedResourceId id;
    std::string destination;
    std::string debugName;
};

} // namespace mlsdk::scenariorunner::detail
