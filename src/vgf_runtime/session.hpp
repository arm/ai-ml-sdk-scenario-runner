/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "vgf.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <vector>

namespace mlsdk::vgf_runtime {

class Session {
  public:
    Session(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, uint32_t queueFamilyIndex,
            const vk::raii::Queue &queue, const VGF &vgf);
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    void bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding);

    // Init cmd buffer, pipelines etc
    void configure();

    // Submit and wait
    void run();

  private:
    struct BoundTensor {
        DescriptorBindingInfo binding;
        vk::raii::TensorViewARM tensorView{nullptr};
    };

    const vk::raii::PhysicalDevice &physicalDevice_;
    const vk::raii::Device &device_;
    const VGF &vgf_;

    uint32_t queueFamilyIndex_ = 0;
    const vk::raii::Queue &queue_;

    vk::raii::ShaderModule shaderModule_{nullptr};
    std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts_;
    vk::raii::PipelineLayout pipelineLayout_{nullptr};
    vk::raii::Pipeline pipeline_{nullptr};
    std::vector<vk::raii::DeviceMemory> sessionMemory_;
    vk::raii::DataGraphPipelineSessionARM graphSession_{nullptr};

    vk::raii::DescriptorPool descriptorPool_{nullptr};
    std::vector<vk::raii::DescriptorSet> descriptorSets_;
    std::vector<BoundTensor> boundTensors_;

    vk::raii::CommandPool commandPool_{nullptr};
    vk::raii::CommandBuffer commandBuffer_{nullptr};
    vk::raii::Fence fence_{nullptr};
    bool configured_ = false;
};

} // namespace mlsdk::vgf_runtime
