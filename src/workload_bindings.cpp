/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils.hpp"

#include "workload_bindings.hpp"

#include <stdexcept>
#include <variant>

namespace mlsdk::scenariorunner {

vk::ImageSubresourceRange WorkloadBindings::imageSubresourceRange(const TypedBinding &binding) const {
    const auto &image = _resources.getImage(std::get<ImageId>(binding.resource));
    if (binding.lod && *binding.lod >= image.getInfo().mips) {
        throw std::runtime_error("Workload image binding mip level is out of range");
    }
    return {getImageAspectMaskForVkFormat(image.dataType()), binding.lod.value_or(0),
            binding.lod ? 1u : image.getInfo().mips, 0, 1};
}

void WorkloadBindings::bind(workloadlib::BindingSet &bindings, workloadlib::ResourceView resource,
                            const TypedBinding &binding, workloadlib::BoundMemoryInfo memory) const {
    if (resource.requirements().descriptorType() != binding.vkDescriptorType) {
        throw std::runtime_error("Scenario descriptor type does not match workload resource");
    }
    switch (binding.vkDescriptorType) {
    case vk::DescriptorType::eStorageBuffer:
        bindings.bindBuffer(resource, {_resources.getBuffer(std::get<BufferId>(binding.resource)).buffer(), memory});
        return;
    case vk::DescriptorType::eTensorARM:
        bindings.bindTensor(resource, {_resources.getTensor(std::get<TensorId>(binding.resource)).tensor(), memory});
        return;
    case vk::DescriptorType::eStorageImage:
    case vk::DescriptorType::eCombinedImageSampler: {
        const auto &image = _resources.getImage(std::get<ImageId>(binding.resource));
        workloadlib::ImageBindingInfo info;
        info.image = image.image();
        info.memory = memory;
        info.imageView = binding.lod ? image.imageView(*binding.lod) : image.imageView();
        if (resource.requirements().asImage().requiresSamplerBinding()) {
            info.sampler = image.sampler();
        }
        info.subresourceRange = imageSubresourceRange(binding);
        // Leave layout unset to keep transitions and tracking in SR. Descriptors
        // still use the workload's required layout; the adapter must ensure SR
        // establishes it for every use, including repeats and mixed commands.
        bindings.bindImage(resource, info);
        return;
    }
    default:
        throw std::runtime_error("Unsupported scenario workload descriptor type");
    }
}

} // namespace mlsdk::scenariorunner
