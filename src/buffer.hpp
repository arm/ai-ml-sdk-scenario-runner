/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "context.hpp"
#include "resource.hpp"
#include "types.hpp"
#include "vulkan_memory_manager.hpp"

namespace mlsdk::scenariorunner {

class Buffer : public Resource {
  public:
    /// \brief Constructor
    ///
    /// \param ctx  Contextual information about the Vulkan instance
    /// \param debugName Debug name
    /// \param size Total size of buffer in bytes
    /// \param memoryManager Information about (possibly shared) underlying memory
    Buffer(const Context &ctx, const std::string &debugName, uint32_t size,
           std::shared_ptr<ResourceMemoryManager> memoryManager);
    Buffer() = default;

    /// \brief Buffer accessor
    /// \return The underlying Vulkan buffer
    const vk::Buffer &buffer() const;

    /// \brief Get buffer size in bytes
    /// \return Size of buffer in bytes
    uint32_t size() const;

    void allocateMemory(const Context &ctx);

    /// \brief Get buffer debug name
    /// \return Debug name associated with the buffer
    const std::string &debugName() const;

    /// \brief Maps buffer memory to host
    /// \return Pointer to the mapped host memory
    void *map();

    /// \brief Un-maps a buffer from the host
    void unmap();

    /// \brief Fills the buffer with the given data
    ///
    /// \param ptr  pointer to data to fill the buffer with
    /// \param size size of data in bytes
    void fill(const void *ptr, size_t size);

    /// \brief Fills the buffer with zeros
    void fillZero();

    void store(Context &ctx, const std::string &filename) override;

  private:
    vk::raii::Buffer _buffer{nullptr};
    uint32_t _size{0};
    std::string _debugName{};
    std::shared_ptr<ResourceMemoryManager> _memoryManager;
};

} // namespace mlsdk::scenariorunner
