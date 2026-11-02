/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "workload_compute.hpp"

#include "data_manager.hpp"
#include "utils.hpp"
#include "workload_bindings.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace mlsdk::scenariorunner {
namespace {

workloadlib::ResourceRequirements resourceRequirements(const DataManager &resources, const TypedBinding &binding) {
    workloadlib::ResourceRequirements requirements;
    requirements.descriptorType = binding.vkDescriptorType;
    switch (binding.vkDescriptorType) {
    case vk::DescriptorType::eStorageBuffer:
        requirements.buffer.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        requirements.buffer.byteSize = resources.getBuffer(std::get<BufferId>(binding.resource)).size();
        break;
    case vk::DescriptorType::eTensorARM: {
        const auto &tensor = resources.getTensor(std::get<TensorId>(binding.resource));
        requirements.format = tensor.dataType();
        requirements.tensor.usage = vk::TensorUsageFlagBitsARM::eShader;
        requirements.tensor.shape = tensor.shape();
        requirements.tensor.stride = tensor.dimStrides();
        break;
    }
    case vk::DescriptorType::eStorageImage:
    case vk::DescriptorType::eCombinedImageSampler: {
        const auto &image = resources.getImage(std::get<ImageId>(binding.resource));
        const auto &shape = image.shape();
        requirements.format = image.dataType();
        requirements.image.usage = binding.vkDescriptorType == vk::DescriptorType::eStorageImage
                                       ? vk::ImageUsageFlagBits::eStorage
                                       : vk::ImageUsageFlagBits::eSampled;
        requirements.image.extent = vk::Extent3D(static_cast<uint32_t>(shape.at(1)), static_cast<uint32_t>(shape.at(2)),
                                                 static_cast<uint32_t>(shape.at(3)));
        requirements.image.imageType = vk::ImageType::e2D;
        // SR establishes and tracks layouts outside the prepared execution.
        requirements.image.requiredLayout = image.getImageLayout();
        requirements.image.range = WorkloadBindings(resources).imageSubresourceRange(binding);
        break;
    }
    default:
        throw std::runtime_error("Unsupported standalone compute descriptor type");
    }
    return requirements;
}

workloadlib::ComputeShaderDescription computeDescription(const DataManager &resources, const ShaderInfo &shader,
                                                         const DispatchComputeData &dispatch) {
    if (shader.stage != ShaderStage::Compute && shader.stage != ShaderStage::Unknown) {
        throw std::runtime_error("DispatchCompute requires a compute shader stage, given: " +
                                 std::to_string(static_cast<int>(shader.stage)));
    }
    if (!shader.src || shader.src->empty()) {
        throw std::runtime_error("Shader resource missing src: " + shader.debugName);
    }
    // Keep SR validation and diagnostics before handing code to workload-lib.
    spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
    tools.SetMessageConsumer(SPIRVMessageConsumer);
    if (!tools.Validate(shader.src->data(), shader.src->size())) {
        throw std::runtime_error("Failed to validate SPIR-V module");
    }

    workloadlib::ComputeShaderDescription description;
    description.module.codeKind = workloadlib::ModuleCodeKind::Spirv;
    description.module.spirv = *shader.src;
    description.entryPoint = shader.entry;
    description.dispatch = {dispatch.computeDispatch.gwcx, dispatch.computeDispatch.gwcy,
                            dispatch.computeDispatch.gwcz};
    description.pushConstantSize = shader.pushConstantsSize;
    // There is only one executable. SR supplies the optional trailing dependency.
    description.implicitBarrier = false;

    using ConstantValue = decltype(SpecializationConstant::value);
    auto &specialization = description.specializationInfo;
    specialization.data.resize(shader.specializationConstants.size() * sizeof(ConstantValue));
    for (const auto &constant : shader.specializationConstants) {
        const auto offset = specialization.mapEntries.size() * sizeof(ConstantValue);
        specialization.mapEntries.emplace_back(static_cast<uint32_t>(constant.id), static_cast<uint32_t>(offset),
                                               sizeof(ConstantValue));
        std::memcpy(specialization.data.data() + offset, &constant.value, sizeof(ConstantValue));
    }
    for (const auto &binding : dispatch.bindings) {
        description.resources.push_back({{},
                                         binding.set,
                                         binding.id,
                                         workloadlib::ResourceAccess::ReadWrite,
                                         resourceRequirements(resources, binding)});
    }
    return description;
}

} // namespace

std::shared_ptr<WorkloadExecution> prepareComputeExecution(std::shared_ptr<WorkloadRuntime> runtime,
                                                           const DataManager &resources, const ShaderInfo &shader,
                                                           const DispatchComputeData &dispatch,
                                                           workloadlib::SessionOptions options) {
    auto description = computeDescription(resources, shader, dispatch);
    auto execution = std::make_shared<WorkloadExecution>(
        std::move(runtime), [&] { return workloadlib::Workload::fromComputeShader(std::move(description)); }, options);
    auto &session = execution->session();
    const auto &workload = execution->workload();
    // Standalone descriptions expose one public resource per supplied binding,
    // in the same order, even when multiple bindings refer to the same SR object.
    for (uint32_t index = 0; index < workload.resourceCount(); ++index) {
        const auto &binding = dispatch.bindings.at(index);
        const auto resource = workload.resource(index);
        if (const auto *tensor = std::get_if<TensorId>(&binding.resource)) {
            session.setTensorTiling(resource, resources.getTensor(*tensor).tiling());
        } else if (const auto *image = std::get_if<ImageId>(&binding.resource)) {
            session.setImageTiling(resource, resources.getImage(*image).tiling());
        }
    }
    session.configure();

    auto bindings = session.createBindingSet();
    const WorkloadBindings adapter(resources);
    for (uint32_t index = 0; index < workload.resourceCount(); ++index) {
        adapter.bind(bindings, workload.resource(index), dispatch.bindings.at(index));
    }
    if (dispatch.pushData) {
        const auto payload = resources.getRawData(*dispatch.pushData).numpyData();
        if (shader.pushConstantsSize != 0) {
            bindings.bindPushConstants(workload.executable(0),
                                       {reinterpret_cast<const std::byte *>(payload.ptr), payload.size()});
        }
    } else if (shader.pushConstantsSize != 0) {
        bindings.bindPushConstants(workload.executable(0), {});
    }
    execution->prepare(bindings);
    return execution;
}

} // namespace mlsdk::scenariorunner
