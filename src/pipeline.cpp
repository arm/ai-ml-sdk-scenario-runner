/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include "spirv-tools/libspirv.hpp"
#include "vulkan_debug_utils.hpp"

namespace mlsdk::scenariorunner {

namespace {

template <typename T, typename U> inline void insertAfter(T *current, U *next) {
    next->pNext = current->pNext;
    current->pNext = next;
}

vk::raii::ShaderModule createShaderModuleFromCode(const Context &ctx, const uint32_t *spvCode, const size_t spvSize) {
    const vk::ShaderModuleCreateInfo shaderCreateInfo({}, spvSize * sizeof(uint32_t), spvCode);
    return vk::raii::ShaderModule(ctx.device(), shaderCreateInfo);
}

vk::raii::ShaderModule createShaderModule(const Context &ctx, const ShaderInfo &shaderInfo) {
    const std::vector<uint32_t> code = readShaderCode(shaderInfo);
    return createShaderModuleFromCode(ctx, code.data(), code.size());
}

void validateShaderModule(const uint32_t *spvCode, const size_t spvSize) {
    spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
    tools.SetMessageConsumer(SPIRVMessageConsumer);
    if (!tools.Validate(spvCode, spvSize)) {
        throw std::runtime_error("Failed to validate SPIR-V module");
    }
}

std::vector<vk::DescriptorSetLayout>
rawLayouts(const std::vector<vk::raii::DescriptorSetLayout> &descriptorSetLayouts) {
    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.reserve(descriptorSetLayouts.size());
    for (auto &raiiLayout : descriptorSetLayouts) {
        layouts.push_back(*raiiLayout);
    }
    return layouts;
}

vk::raii::PipelineLayout createPipelineLayout(const Context &ctx,
                                              const std::vector<vk::raii::DescriptorSetLayout> &descriptorSetLayouts,
                                              uint32_t pushConstantsSize = 0) {
    auto layouts = rawLayouts(descriptorSetLayouts);
    if (pushConstantsSize > 0) {
        const vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, pushConstantsSize);
        const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, layouts, pushConstantRange);
        return vk::raii::PipelineLayout(ctx.device(), pipelineLayoutCreateInfo);
    }

    const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, layouts);
    return vk::raii::PipelineLayout(ctx.device(), pipelineLayoutCreateInfo);
}

vk::raii::DescriptorSetLayout createDescriptorSetLayout(const Context &ctx, const std::vector<TypedBinding> &bindings) {
    std::vector<vk::DescriptorSetLayoutBinding> descBindings;
    for (const auto &binding : bindings) {
        const vk::DescriptorSetLayoutBinding descBinding(binding.id, binding.vkDescriptorType, 1,
                                                         vk::ShaderStageFlagBits::eAll);
        descBindings.emplace_back(descBinding);
    }
    const vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo({}, descBindings);
    return vk::raii::DescriptorSetLayout(ctx.device(), descSetLayoutCreateInfo);
}

std::vector<std::vector<TypedBinding>> splitOutSets(const std::vector<TypedBinding> &allBindings) {
    std::vector<std::vector<TypedBinding>> setBindings;

    for (const auto &bindingDesc : allBindings) {
        while (setBindings.size() <= static_cast<size_t>(bindingDesc.set)) {
            setBindings.emplace_back(0);
        }

        setBindings[bindingDesc.set].push_back(bindingDesc);
    }

    return setBindings;
}

} // namespace

void Pipeline::createDescriptorSetLayouts(const Context &ctx, const std::vector<TypedBinding> &bindings) {
    for (const auto &setBindings : splitOutSets(bindings)) {
        _descriptorSetLayouts.push_back(createDescriptorSetLayout(ctx, setBindings));
    }
}

void Pipeline::computePipelineCommon(const Context &ctx, const ShaderInfo &shaderInfo,
                                     std::optional<PipelineCache> &pipelineCache) {
    _pipelineLayout = createPipelineLayout(ctx, _descriptorSetLayouts, shaderInfo.pushConstantsSize);

    std::vector<vk::SpecializationMapEntry> specMapEntries(shaderInfo.specializationConstants.size());
    std::vector<decltype(SpecializationConstant::value)> specConstValues(shaderInfo.specializationConstants.size());
    const auto specConstSize = sizeof(SpecializationConstant::value);
    for (uint32_t i = 0, offset = 0; i < static_cast<uint32_t>(shaderInfo.specializationConstants.size());
         ++i, offset += specConstSize) {
        const auto &specConst = shaderInfo.specializationConstants[i];
        specMapEntries[i] = vk::SpecializationMapEntry(static_cast<uint32_t>(specConst.id), offset, specConstSize);
        specConstValues[i] = specConst.value;
    }

    const vk::SpecializationInfo specInfo(static_cast<uint32_t>(specMapEntries.size()), specMapEntries.data(),
                                          specConstValues.size() * specConstSize, specConstValues.data());

    const vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(
        {}, vk::ShaderStageFlagBits::eCompute, *_shader, shaderInfo.entry.c_str(), &specInfo);

    vk::PipelineCreateFlags flags{};
    const void *pNext{nullptr};
    const vk::raii::PipelineCache *vkPipelineCache{nullptr};
    if (pipelineCache.has_value()) {
        if (pipelineCache.value().failOnCacheMiss()) {
            flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
        }
        pNext = pipelineCache.value().getCacheFeedbackCreateInfo();
        vkPipelineCache = pipelineCache.value().get();
    }

    const vk::ComputePipelineCreateInfo computePipelineCreateInfo(flags, pipelineShaderStageCreateInfo,
                                                                  *_pipelineLayout, {}, {}, pNext);
    _pipeline = vk::raii::Pipeline(ctx.device(), vkPipelineCache, computePipelineCreateInfo);

    trySetVkRaiiObjectDebugName(ctx, _pipeline, _debugName);
}

Pipeline::Pipeline(const CommonArguments &args, const uint32_t *spvCode, const size_t spvSize,
                   const ShaderInfo &shaderInfo)
    : _type{PipelineType::Compute}, _debugName(args.debugName) {

    validateShaderModule(spvCode, spvSize);

    _shader = createShaderModuleFromCode(args.ctx, spvCode, spvSize);
    trySetVkRaiiObjectDebugName(args.ctx, _shader, _debugName + " shader");

    createDescriptorSetLayouts(args.ctx, args.bindings);
    computePipelineCommon(args.ctx, shaderInfo, args.pipelineCache);
}

Pipeline::Pipeline(const CommonArguments &args, const ShaderInfo &shaderInfo)
    : _type{PipelineType::Compute}, _shader(createShaderModule(args.ctx, shaderInfo)), _debugName(args.debugName) {

    trySetVkRaiiObjectDebugName(args.ctx, _shader, shaderInfo.debugName);

    createDescriptorSetLayouts(args.ctx, args.bindings);
    computePipelineCommon(args.ctx, shaderInfo, args.pipelineCache);
}

Pipeline::Pipeline(const CommonArguments &args, const uint32_t segmentIndex, const VgfView &vgfView,
                   const DataManager &dataManager)
    : _type{PipelineType::GraphCompute}, _debugName(args.debugName) {

    createDescriptorSetLayouts(args.ctx, args.bindings);
    _pipelineLayout = createPipelineLayout(args.ctx, _descriptorSetLayouts);

    // Setup tensor resource info
    std::vector<vk::TensorDescriptionARM> tensorDescriptions;
    tensorDescriptions.reserve(args.bindings.size());
    std::vector<vk::DataGraphPipelineResourceInfoARM> resourceInfos;
    resourceInfos.reserve(args.bindings.size());

    for (const auto &binding : args.bindings) {
        if (!dataManager.hasTensor(binding.resourceRef)) {
            throw std::runtime_error("Unsupported graph pipeline resource");
        }

        const auto &tensor = dataManager.getTensor(binding.resourceRef);

        const auto *strides = tensor.dimStrides().data();
        if (tensor.dimStrides().empty()) {
            strides = nullptr;
        }

        tensorDescriptions.emplace_back(
            vk::TensorDescriptionARM(tensor.tiling(), vk::Format(tensor.dataType()),
                                     static_cast<uint32_t>(tensor.shape().size()), // dimensions
                                     tensor.shape().data(),
                                     strides, // pStrides
                                     vk::TensorUsageFlagBitsARM::eDataGraph));
        resourceInfos.emplace_back(static_cast<uint32_t>(binding.set), static_cast<uint32_t>(binding.id),
                                   /*arrayElement=*/0, &tensorDescriptions.back());
    }

    graphComputePipelineCommon(args.ctx, segmentIndex, vgfView, args.pipelineCache, resourceInfos);
}

void Pipeline::graphComputePipelineCommon(const Context &ctx, uint32_t segmentIndex, const VgfView &vgfView,
                                          std::optional<PipelineCache> &pipelineCache,
                                          const std::vector<vk::DataGraphPipelineResourceInfoARM> &resourceInfos) {
    // Setup constant resource info
    auto constantIndexes = vgfView.getSegmentConstantIndexes(segmentIndex);
    std::vector<vk::TensorDescriptionARM> constantTensorDescriptions;
    constantTensorDescriptions.reserve(constantIndexes.size());

    std::vector<vk::DataGraphPipelineConstantARM> constantInfos;
    constantInfos.reserve(constantIndexes.size());

    std::vector<vk::DataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM> sparsityInfos;
    sparsityInfos.reserve(constantIndexes.size());

    for (uint32_t constantIndex : constantIndexes) {
        void *pNext = nullptr;

        auto constantData = vgfView.getConstantData(constantIndex);
        auto shape = vgfView.getConstantShape(constantIndex);
        auto vkFormat = vk::Format(vgfView.getConstantFormat(constantIndex));
        int64_t sparsityDimension = vgfView.getConstantSparsityDimension(constantIndex);

        if (sparsityDimension >= 0) {
            constexpr uint32_t zeroCount = 2;
            constexpr uint32_t groupSize = 4;

            sparsityInfos.emplace_back(static_cast<uint32_t>(sparsityDimension), zeroCount, groupSize, nullptr);
            pNext = &sparsityInfos.back();
        }

        constantTensorDescriptions.emplace_back(vk::TensorTilingARM::eLinear, vkFormat,
                                                static_cast<uint32_t>(shape.size()), shape.begin(),
                                                nullptr, // pStrides
                                                vk::TensorUsageFlagBitsARM::eDataGraph, pNext);
        constantInfos.emplace_back(constantIndex, constantData.begin(), &constantTensorDescriptions.back());
    }

    // Compile SPIR-V code
    auto spv = vgfView.getSPVModule(segmentIndex);
    validateShaderModule(spv.begin(), spv.size());
    _shader = createShaderModuleFromCode(ctx, spv.begin(), spv.size());
    trySetVkRaiiObjectDebugName(ctx, _shader, _debugName + " shader");

    auto entryPoint = vgfView.getSPVModuleEntryPoint(segmentIndex);

    vk::DataGraphPipelineShaderModuleCreateInfoARM pipelineShaderModuleCreateInfo(
        *_shader, entryPoint.c_str(), nullptr, static_cast<uint32_t>(constantInfos.size()), constantInfos.data(),
        nullptr);

    const vk::raii::DeferredOperationKHR deferredOperation(nullptr);

    vk::PipelineCreateFlags2KHR flags{};
    const vk::raii::PipelineCache *vkPipelineCache{nullptr};
    if (pipelineCache.has_value()) {
        if (pipelineCache.value().failOnCacheMiss()) {
            flags |= vk::PipelineCreateFlagBits2KHR::eFailOnPipelineCompileRequired;
        }
        insertAfter(&pipelineShaderModuleCreateInfo, pipelineCache.value().getCacheFeedbackCreateInfo());
        vkPipelineCache = pipelineCache.value().get();
    }

    const vk::DataGraphPipelineCreateInfoARM pipelineCreateInfo(flags, *_pipelineLayout,
                                                                static_cast<uint32_t>(resourceInfos.size()),
                                                                resourceInfos.data(), &pipelineShaderModuleCreateInfo);
    _pipeline = vk::raii::Pipeline(ctx.device(), deferredOperation, vkPipelineCache, pipelineCreateInfo);

    trySetVkRaiiObjectDebugName(ctx, _pipeline, _debugName);

    initSession(ctx);
}

void Pipeline::initSession(const Context &ctx) {
    // Create session for the pipeline
    vk::DataGraphPipelineSessionCreateInfoARM sessionCreateInfo{/*flags=*/{}, *_pipeline};

    _session = vk::raii::DataGraphPipelineSessionARM{ctx.device(), sessionCreateInfo};

    // Get memory requirements
    vk::DataGraphPipelineSessionBindPointRequirementsInfoARM bindpointRequirementsInfo(*_session);
    auto bindPointReqs = ctx.device().getDataGraphPipelineSessionBindPointRequirementsARM(bindpointRequirementsInfo);

    std::vector<vk::BindDataGraphPipelineSessionMemoryInfoARM> bindInfos;
    for (auto &bindPointReq : bindPointReqs) {
        if (bindPointReq.bindPointType != vk::DataGraphPipelineSessionBindPointTypeARM::eMemory) {
            continue;
        }

        vk::DataGraphPipelineSessionMemoryRequirementsInfoARM memoryRequirementsInfo(*_session, bindPointReq.bindPoint);
        VkMemoryRequirements2 memoryReqs =
            ctx.device().getDataGraphPipelineSessionMemoryRequirementsARM(memoryRequirementsInfo);

        //  Allocate memory for the session
        if (memoryReqs.memoryRequirements.size > 0) {
            vk::MemoryPropertyFlags memoryFlags = {};
            if (ctx.sessionMemoryDumpEnabled()) {
                mlsdk::logging::warning("Enabling session memory dumping is known to cause issues on certain GPUs.");
                memoryFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
            }
            auto memoryTypeIdx = findMemoryIdx(ctx, memoryReqs.memoryRequirements.memoryTypeBits, memoryFlags);

            vk::MemoryAllocateInfo allocateInfo(memoryReqs.memoryRequirements.size, memoryTypeIdx);
            _sessionMemory.emplace_back(vk::raii::DeviceMemory(ctx.device(), allocateInfo));
            _sessionMemoryDataSizes.push_back(memoryReqs.memoryRequirements.size);

            // Bind memory to session
            uint32_t resourceIndex = 0;
            bindInfos.emplace_back(*_session, bindPointReq.bindPoint, resourceIndex, *_sessionMemory.back());
        }
    }

    if (!bindInfos.empty()) {
        ctx.device().bindDataGraphPipelineSessionMemoryARM(bindInfos);
    }
}

const std::string &Pipeline::debugName() const { return _debugName; }

} // namespace mlsdk::scenariorunner
