/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vgf_runtime/workload_frontends/vgf.hpp>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <memory>

namespace mlsdk::vgf_runtime {

/**
 * @brief Configures and runs VGF graph segments on a Vulkan device.
 *
 * A session binds tensors to VGF resources, creates the required Vulkan state
 * once with configure(), and then executes the graph with run().
 */
class Session {
  public:
    struct BoundMemoryInfo {
        vk::DeviceMemory memory;
        vk::DeviceSize offset;
        vk::DeviceSize size;
    };

    /** @brief Create a session bound to a Vulkan device, queue, and decoded VGF. */
    Session(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, uint32_t queueFamilyIndex,
            const vk::raii::Queue &queue, const VGF &vgf);
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    /** @brief Bind a tensor to the descriptor binding described by @p binding. */
    void bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding,
                    BoundMemoryInfo memory = BoundMemoryInfo());

    /** @brief Bind a buffer to the descriptor binding described by @p binding. */
    void bindBuffer(const vk::raii::Buffer &buffer, DescriptorBindingInfo binding,
                    BoundMemoryInfo memory = BoundMemoryInfo());

    /**
     * @brief Bind an image to the descriptor binding described by @p binding.
     *
     * The image is assumed to already be in the layout required by @p binding.
     * Use the overload with @p currentLayout when the image needs a transition
     * before the first session dispatch.
     */
    void bindImage(const vk::raii::Image &image, DescriptorBindingInfo binding,
                   BoundMemoryInfo memory = BoundMemoryInfo());

    /** @brief Bind an image and describe its layout before the first session dispatch. */
    void bindImage(const vk::raii::Image &image, DescriptorBindingInfo binding, vk::ImageLayout currentLayout,
                   BoundMemoryInfo memory = BoundMemoryInfo());

    /** @brief Create the Vulkan objects needed to execute the decoded graph. */
    void configure();

    /** @brief Submit the configured graph to the session queue and wait for completion. */
    void run();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mlsdk::vgf_runtime
