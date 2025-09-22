/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "commands.hpp"
#include "context.hpp"
#include "data_manager.hpp"
#include "pipeline_cache.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlsdk::scenariorunner {

enum class PipelineType { Unknown, Compute, GraphCompute };

class Pipeline {
  public:
    /// \brief Constructor
    ///
    /// \param ctx GPU context
    /// \param debugName Debug name
    /// \param bindings Bindings related meta-data
    /// \param shaderDesc Shader related meta-data
    /// \param pipelineCache optional pipeline cache object
    Pipeline(const Context &ctx, const std::string &debugName, const std::vector<TypedBinding> &bindings,
             const ShaderDesc &shaderDesc, std::optional<PipelineCache> &pipelineCache);

    Pipeline(const Context &ctx, const std::string &debugName, const uint32_t *spvCode, const size_t spvSize,
             const std::vector<TypedBinding> &sequenceBindings, const ShaderDesc &shaderDesc,
             std::optional<PipelineCache> &pipelineCache);

    Pipeline(const Context &ctx, const std::string &debugName, const uint32_t segmentIndex,
             const std::vector<TypedBinding> &sequenceBindings, const VgfView &vgfView, const DataManager &dataManager,
             std::optional<PipelineCache> &pipelineCache);

    /// \brief Vulkan® pipeline accessor
    /// \return The underlying Vulkan® pipeline of the object
    const vk::Pipeline &pipeline() const { return *_pipeline; }

    const vk::PipelineLayout &pipelineLayout() const { return *_pipelineLayout; }

    const vk::DescriptorSetLayout &descriptorSetLayout(uint32_t setIdx) const { return *_descriptorSetLayouts[setIdx]; }

    const vk::DataGraphPipelineSessionARM &session() const { return *_session; }

    const std::vector<vk::raii::DeviceMemory> &sessionMemory() const { return _sessionMemory; }

    const std::vector<vk::DeviceSize> &sessionMemoryDataSizes() const { return _sessionMemoryDataSizes; }

    bool isDataGraphPipeline() const { return _type == PipelineType::GraphCompute; };

    template <typename T>
    std::vector<T> getGraphPipelinePropertyData(const vk::raii::Device &device,
                                                vk::DataGraphPipelinePropertyARM property) const;

    const std::string &debugName() const;

  private:
    PipelineType _type{PipelineType::Unknown};
    std::vector<vk::raii::DescriptorSetLayout> _descriptorSetLayouts;
    vk::raii::PipelineLayout _pipelineLayout{nullptr};
    vk::raii::Pipeline _pipeline{nullptr};
    vk::raii::DataGraphPipelineSessionARM _session{nullptr};
    std::vector<vk::raii::DeviceMemory> _sessionMemory;
    std::vector<vk::DeviceSize> _sessionMemoryDataSizes;
    vk::raii::ShaderModule _shader{nullptr};
    std::string _debugName{};

    void initSession(const Context &ctx);

    void createDescriptorSetLayouts(const Context &ctx, const std::vector<TypedBinding> &bindings);

    void computePipelineCommon(const Context &ctx, const ShaderDesc &shaderDesc,
                               std::optional<PipelineCache> &pipelineCache);

    void graphComputePipelineCommon(const Context &ctx, uint32_t segmentIndex, const VgfView &vgfView,
                                    std::optional<PipelineCache> &pipelineCache,
                                    const std::vector<vk::DataGraphPipelineResourceInfoARM> &resourceInfos);
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
