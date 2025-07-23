/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "context.hpp"
#include "types.hpp"

namespace mlsdk::scenariorunner {

class VulkanImageBarrier {
  public:
    /// \brief Constructor
    ///
    /// \param imageBarrierData Struct that contains needed information to create a VulkanImageBarrier
    explicit VulkanImageBarrier(const ImageBarrierData &imageBarrierData);
    VulkanImageBarrier() = default;

    /// \brief VulkanImageBarrier accessor
    /// \return The underlying Vulkan® VulkanImageBarrier
    const vk::ImageMemoryBarrier2 &imageBarrier() const;

    /// \brief Get VulkanImageBarrier debug name
    /// \return Debug name associated with the VulkanImageBarrier
    const std::string &debugName() const;

  private:
    vk::ImageMemoryBarrier2 _imageBarrier;
    std::string _debugName{};
};

class VulkanTensorBarrier {
  public:
    /// \brief Constructor
    ///
    /// \param tensorBarrierData tensor barrier information struct
    explicit VulkanTensorBarrier(const TensorBarrierData &tensorBarrierData);
    VulkanTensorBarrier() = default;

    /// \brief VulkanTensorBarrier accessor
    /// \return The underlying Vulkan® TensorMemoryBarrierARM
    const vk::TensorMemoryBarrierARM &tensorBarrier() const;

    /// \brief Get VulkanTensorBarrier debug name
    /// \return Debug name associated with the VulkanTensorBarrier
    const std::string &debugName() const;

  private:
    vk::TensorMemoryBarrierARM _tensorBarrier;
    std::string _debugName{};
};

class VulkanMemoryBarrier {
  public:
    /// \brief Constructor
    ///
    /// \param memoryBarrierData Struct containing source and destination access info
    explicit VulkanMemoryBarrier(const MemoryBarrierData &memoryBarrierData);
    VulkanMemoryBarrier() = default;

    /// \brief MemoryBarrier accessor
    /// \return The underlying Vulkan® MemoryBarrier
    const vk::MemoryBarrier2 &memoryBarrier() const;

    /// \brief Get MemoryBarrier debug name
    /// \return Debug name associated with the MemoryBarrier
    const std::string &debugName() const;

  private:
    vk::MemoryBarrier2 _memoryBarrier;
    std::string _debugName{};
};

class VulkanBufferBarrier {
  public:
    /// \param bufferBarrierData Struct which contains needed information to construct a VulkanBufferBarrier
    explicit VulkanBufferBarrier(const BufferBarrierData &bufferBarrierData);
    VulkanBufferBarrier() = default;

    /// \brief VulkanBufferBarrier accessor
    /// \return The underlying Vulkan® VulkanBufferBarrier
    const vk::BufferMemoryBarrier2 &bufferBarrier() const;

    /// \brief Get VulkanBufferBarrier debug name
    /// \return Debug name associated with the VulkanBufferBarrier
    const std::string &debugName() const;

  private:
    vk::BufferMemoryBarrier2 _bufferBarrier;
    std::string _debugName{};
};
} // namespace mlsdk::scenariorunner
