/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "context.hpp"
#include "resource_data.hpp"
#include "resource_desc.hpp"
#include "types.hpp"
#include "vulkan_memory_manager.hpp"

namespace mlsdk::scenariorunner {

class Buffer {
  public:
    /// \brief Constructor
    ///
    /// \param bufferInfo buffer Information
    explicit Buffer(BufferInfo bufferInfo);
    Buffer() = default;

    /// \brief Setup instance, assumes all aliasing objects have been constructed
    /// \param ctx           Contextual information about the Vulkan instance
    /// \param memoryManager Memory manager for this resource
    void setup(const Context &ctx,
               std::shared_ptr<ResourceMemoryManager> memoryManager = std::make_shared<ResourceMemoryManager>());

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

    void fillFromDescription(const Context &ctx, const BufferDesc &buffer) const;

    /// \brief Fills the buffer with the given data
    ///
    /// \param ctx  Contextual information about the Vulkan instance
    /// \param ptr  pointer to data to fill the buffer with
    /// \param size size of data in bytes
    void fill(const Context &ctx, const void *ptr, size_t size) const;

    /// \brief Fills the buffer with zeros
    void fillZero(const Context &ctx) const;

    /// \brief Upload in‑memory data into this buffer (host → device)
    ///
    /// Copies the provided bytes to device‑local memory via a staging transfer
    /// The payload size must exactly match `size()`
    /// \param ctx   Vulkan context
    /// \param data  BufferDataView containing bytes for upload
    void upload(const Context &ctx, const BufferDataView &data) const;

    /// \brief Download buffer contents (device → host)
    ///
    /// Reads the current buffer contents into an owned byte vector
    /// Buffers are untyped; only raw bytes are returned
    /// \param ctx Vulkan context
    /// \return BufferData containing a copy of the bytes
    BufferData download(const Context &ctx) const;

    /// \brief Retrieves the buffer data and writes it to a file
    void store(const Context &ctx, const std::string &filename) const;

    std::shared_ptr<ResourceMemoryManager> memoryManager() const { return _memoryManager; }

  private:
    vk::raii::Buffer _buffer{nullptr};
    uint32_t _size{0};
    std::string _debugName;
    std::shared_ptr<ResourceMemoryManager> _memoryManager;
    uint64_t _memoryOffset{0};
};

} // namespace mlsdk::scenariorunner
