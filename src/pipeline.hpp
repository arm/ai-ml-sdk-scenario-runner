/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "context.hpp"
#include "data_manager.hpp"
#include "pipeline_cache.hpp"
#include "types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace mlsdk::scenariorunner {

class Pipeline {
  public:
    struct CommonArguments {
        const Context &ctx;
        const std::string &debugName;
        const std::vector<TypedBinding> &bindings;
        std::shared_ptr<PipelineCache> pipelineCache;
    };

    /// \brief Constructor
    ///
    /// \param args Common arguments struct
    /// \param shaderInfo Shader related meta-data
    /// \param spvCode Pointer to SPIR-V code
    /// \param spvSize Size of SPIR-V code in number of uint32_t
    Pipeline(const CommonArguments &args, const ShaderInfo &shaderInfo, const uint32_t *spvCode, size_t spvSize);

    Pipeline(const CommonArguments &args, uint32_t segmentIndex, const VgfView &vgfView,
             const DataManager &dataManager);

    // Create DataGraph pipeline directly from SPIR-V module + constants (no VGF)
    Pipeline(const CommonArguments &args, const ShaderInfo &shaderInfo, const DataManager &dataManager,
             const std::vector<GraphConstantInfo> &constants);

    Pipeline(const CommonArguments &args, const ShaderInfo &vertexShaderInfo, const ShaderInfo &fragmentShaderInfo,
             const std::vector<vk::Format> &colorAttachmentFormats);

    /// \brief Vulkan® pipeline accessor
    /// \return The underlying Vulkan® pipeline of the object
    const vk::Pipeline &pipeline() const { return *_pipeline; }

    const vk::PipelineLayout &pipelineLayout() const { return *_pipelineLayout; }

    const vk::DescriptorSetLayout &descriptorSetLayout(uint32_t setIdx) const { return *_descriptorSetLayouts[setIdx]; }

    const vk::DataGraphPipelineSessionARM &session() const { return *_session; }

    const std::vector<vk::raii::DeviceMemory> &sessionMemory() const { return _sessionMemory; }

    const std::vector<vk::DeviceSize> &sessionMemoryDataSizes() const { return _sessionMemoryDataSizes; }

    uint64_t getDataGraphPipelineMemoryRequirement() const { return _dataGraphPipelineMemoryRequirement; }

    bool isDataGraphPipeline() const { return _type == PipelineType::GraphCompute; };

    bool isGraphicsPipeline() const { return _type == PipelineType::Graphics; };

    vk::ShaderStageFlags pushConstantStages() const { return _pushConstantStages; }

    template <typename T>
    std::vector<T> getGraphPipelinePropertyData(const vk::raii::Device &device,
                                                vk::DataGraphPipelinePropertyARM property) const;

    const std::string &debugName() const;

  private:
    enum class PipelineType { Unknown, Compute, GraphCompute, Graphics };

    PipelineType _type{PipelineType::Unknown};
    std::vector<vk::raii::DescriptorSetLayout> _descriptorSetLayouts;
    vk::raii::PipelineLayout _pipelineLayout{nullptr};
    vk::raii::Pipeline _pipeline{nullptr};
    vk::raii::DataGraphPipelineSessionARM _session{nullptr};
    std::vector<vk::raii::DeviceMemory> _sessionMemory;
    std::vector<vk::DeviceSize> _sessionMemoryDataSizes;
    vk::raii::ShaderModule _shader{nullptr};
    vk::raii::ShaderModule _fragmentShader{nullptr};
    std::string _debugName{};
    uint64_t _dataGraphPipelineMemoryRequirement{};
    vk::ShaderStageFlags _pushConstantStages{};

    void initSession(const Context &ctx);

    void createDescriptorSetLayouts(const Context &ctx, const std::vector<TypedBinding> &bindings);

    void computePipelineCommon(const Context &ctx, const ShaderInfo &shaderInfo,
                               std::shared_ptr<PipelineCache> pipelineCache);

    void graphicsPipelineCommon(const Context &ctx, const ShaderInfo &vertexShaderInfo,
                                const ShaderInfo &fragmentShaderInfo,
                                const std::vector<vk::Format> &colorAttachmentFormats,
                                std::shared_ptr<PipelineCache> pipelineCache);

    void graphComputePipelineCommon(const Context &ctx, uint32_t segmentIndex, const VgfView &vgfView,
                                    std::shared_ptr<PipelineCache> pipelineCache,
                                    const std::vector<vk::DataGraphPipelineResourceInfoARM> &resourceInfos);

    // Helper to build a Datagraph pipeline that has been dispatched through SPIR-V
    void graphComputePipelineCommon(const Context &ctx, const ShaderInfo &shaderInfo,
                                    const std::vector<vk::DataGraphPipelineResourceInfoARM> &resourceInfos,
                                    const std::vector<vk::DataGraphPipelineConstantARM> &constantInfos,
                                    std::shared_ptr<PipelineCache> pipelineCache);

    // Helper to build a DataGraph pipeline once shader module, entry, resources and constants are prepared
    void buildDataGraphPipeline(const Context &ctx, const std::string &entry,
                                const std::vector<vk::DataGraphPipelineResourceInfoARM> &resourceInfos,
                                const std::vector<vk::DataGraphPipelineConstantARM> &constantInfos,
                                std::shared_ptr<PipelineCache> pipelineCache);
};

template <typename T>
std::vector<T> Pipeline::getGraphPipelinePropertyData(const vk::raii::Device &device,
                                                      vk::DataGraphPipelinePropertyARM property) const {
    if (!isDataGraphPipeline()) {
        throw std::runtime_error("getDataGraphPipelinePropertiesARM called on a non DataGraphPipeline");
    }

    vk::DataGraphPipelineInfoARM pipelineInfo{*_pipeline, nullptr};
    uint32_t propCount = 1;
    vk::DataGraphPipelinePropertyQueryResultARM propQuery{property, false, 0, nullptr};

    auto res = device.getDataGraphPipelinePropertiesARM(&pipelineInfo, propCount, &propQuery);
    if (res != vk::Result::eSuccess && res != vk::Result::eIncomplete) {
        throw std::runtime_error("getDataGraphPipelinePropertiesARM returned failure");
    }

    std::vector<T> propData = {};
    if (propQuery.dataSize > 0) {
        propData.resize(propQuery.dataSize / sizeof(T));
        propQuery.pData = propData.data();
        res = device.getDataGraphPipelinePropertiesARM(&pipelineInfo, propCount, &propQuery);
        if (res != vk::Result::eSuccess && res != vk::Result::eIncomplete) {
            throw std::runtime_error("getDataGraphPipelinePropertiesARM returned failure");
        }

        if (propQuery.isText == vk::True && propData.back() != '\0') {
            propData.push_back('\0');
        }
    }

    return propData;
}

} // namespace mlsdk::scenariorunner
