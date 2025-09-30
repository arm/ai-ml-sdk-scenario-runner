/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "context.hpp"
#include "data_manager.hpp"
#include "pipeline_cache.hpp"
#include "types.hpp"

#include <optional>
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
        std::optional<PipelineCache> &pipelineCache;
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
    enum class PipelineType { Unknown, Compute, GraphCompute };

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

    void computePipelineCommon(const Context &ctx, const ShaderInfo &shaderInfo,
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
