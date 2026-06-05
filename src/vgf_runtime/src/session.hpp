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
    struct BoundMemoryInfo {
        vk::DeviceMemory memory;
        vk::DeviceSize offset;
        vk::DeviceSize size;
    };

    Session(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, uint32_t queueFamilyIndex,
            const vk::raii::Queue &queue, const VGF &vgf);
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    void bindTensor(const vk::raii::TensorARM &tensor, DescriptorBindingInfo binding,
                    BoundMemoryInfo memory = BoundMemoryInfo());
    void bindBuffer(const vk::raii::Buffer &buffer, DescriptorBindingInfo binding,
                    BoundMemoryInfo memory = BoundMemoryInfo());

    // Init cmd buffer, pipelines etc
    void configure();

    // Submit and wait
    void run();

  private:
    struct BoundTensor;
    struct BoundBuffer;
    struct SegmentState;

    void configureSegment(uint32_t segmentIndex);
    void allocateResources();
    vk::raii::TensorARM createIntermediateTensor(const DescriptorBindingInfo &binding) const;
    vk::raii::Buffer createIntermediateBuffer(const DescriptorBindingInfo &binding) const;
    void allocateIntermediateTensor(const DescriptorBindingInfo &binding);
    void allocateIntermediateBuffer(const DescriptorBindingInfo &binding);
    void allocateAliasedResources(const std::vector<DescriptorBindingInfo> &bindings);
    void addBoundTensor(vk::TensorARM tensor, DescriptorBindingInfo binding,
                        BoundMemoryInfo memory = BoundMemoryInfo());
    void addBoundBuffer(vk::Buffer buffer, DescriptorBindingInfo binding, BoundMemoryInfo memory = BoundMemoryInfo());
    const BoundTensor *findBoundTensor(uint32_t resourceIndex) const;
    const BoundBuffer *findBoundBuffer(uint32_t resourceIndex) const;
    const BoundTensor *findBoundTensorInAliasGroup(uint32_t aliasGroupId) const;
    const BoundBuffer *findBoundBufferInAliasGroup(uint32_t aliasGroupId) const;
    void updateDescriptorSets(const std::vector<vk::raii::DescriptorSet> &descriptorSets,
                              const std::vector<DescriptorBindingInfo> &bindings) const;
    void insertSegmentBarrier(vk::raii::CommandBuffer &commandBuffer, const SegmentState &producer,
                              const SegmentState &consumer) const;

    const vk::raii::PhysicalDevice &physicalDevice_;
    const vk::raii::Device &device_;
    const VGF &vgf_;

    uint32_t queueFamilyIndex_ = 0;
    const vk::raii::Queue &queue_;

    std::vector<vk::raii::DeviceMemory> ownedMemory_;
    std::vector<vk::raii::TensorARM> ownedTensors_;
    std::vector<vk::raii::Buffer> ownedBuffers_;
    std::vector<BoundTensor> boundTensors_;
    std::vector<BoundBuffer> boundBuffers_;
    std::vector<SegmentState> segments_;

    vk::raii::CommandPool commandPool_{nullptr};
    vk::raii::CommandBuffer commandBuffer_{nullptr};
    vk::raii::Fence fence_{nullptr};
    bool configured_ = false;
};

} // namespace mlsdk::vgf_runtime
