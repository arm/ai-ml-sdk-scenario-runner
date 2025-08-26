/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "context.hpp"
#include "resource.hpp"
#include "resource_desc.hpp"
#include "types.hpp"
#include "vulkan_memory_manager.hpp"

namespace mlsdk::scenariorunner {

class Tensor : public Resource {
  public:
    /// \brief Constructor
    ///
    /// \param ctx                    Contextual information about the Vulkan instance
    /// \param tensorInfo             Tensor info
    /// \param memoryManager          Memory manager for this resource
    Tensor(Context &ctx, const TensorInfo &tensorInfo, std::shared_ptr<ResourceMemoryManager> memoryManager);
    Tensor() = default;

    /// \brief Tensor accessor
    /// \return The underlying Vulkan tensor
    const vk::TensorARM &tensor() const;

    /// \brief Tensor view accessor
    /// \return The underlying Vulkan tensor view
    const vk::TensorViewARM &tensorView() const;

    /// \brief Get total size of memory object associated with this tensor
    /// \return Size of memory in bytes
    uint64_t memSize() const;

    /// \brief Get tensor packed data size in bytes
    /// \return Size of tensor data in bytes
    uint64_t dataSize() const;

    /// \brief Get tensor dimensions strides
    /// \return dimensions strides
    const std::vector<int64_t> &dimStrides() const;

    /// \brief Get tensor data type
    /// \return Data type of tensor
    vk::Format dataType() const;

    /// \brief Get tensor shape
    /// \return Vector containing the shape of tensor
    const std::vector<int64_t> &shape() const;

    /// \brief Get tensor tiling setting
    /// \return tiling
    vk::TensorTilingARM tiling() const;

    /// \brief Maps buffer memory to host
    /// \return Pointer to the mapped host memory
    void *map();

    /// \brief Un-maps a buffer from the host
    void unmap();

    /// \brief checks if tensor's shape has been converted from dims=[] to dims=[1]
    bool isRankConverted() const { return _rankConverted; };

    void allocateMemory(const Context &ctx);

    void fillFromDescription(const TensorDesc &desc);

    void store(Context &ctx, const std::string &filename) override;

    const std::string &debugName() const;

  private:
    void fill(const void *data, size_t size);
    void fillZero();

    std::string _debugName{};
    vk::raii::TensorARM _tensor{nullptr};
    vk::raii::TensorViewARM _tensorView{nullptr};
    std::vector<int64_t> _shape{};
    vk::Format _dataType = vk::Format::eUndefined;
    std::vector<int64_t> _strides{};
    std::shared_ptr<ResourceMemoryManager> _memoryManager{nullptr};
    vk::TensorTilingARM _tiling = vk::TensorTilingARM::eLinear;
    uint64_t _memoryOffset{0};
    bool _rankConverted{false};
};

} // namespace mlsdk::scenariorunner
