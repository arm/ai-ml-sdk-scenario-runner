/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vgf-utils/memory_map.hpp>
#include <vgf/decoder.hpp>

#include <vulkan/vulkan.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlsdk::vgf_runtime {

struct SPIRVModule {
    uint32_t index = 0;
    std::string name;
    std::string entryPoint;
    vgflib::DataView<uint32_t> code;
};

/*******************************************************************************
 * VGF File sections
 *******************************************************************************/

struct SegmentInfo {
    uint32_t index = 0;
    std::string name;
    vgflib::ModuleType type = vgflib::ModuleType::GRAPH;
    uint32_t moduleIndex = 0;
    std::string moduleName;
    std::string entryPoint;
};

struct DescriptorBindingInfo {
    uint32_t set = 0;
    uint32_t binding = 0;
    uint32_t resourceIndex = 0;
    vk::DescriptorType descriptorType = {};
    vgflib::ResourceCategory resourceCategory = vgflib::ResourceCategory::INPUT;
};

struct SamplerInfo {
    uint32_t minFilter = 0;
    uint32_t magFilter = 0;
    uint32_t addressModeU = 0;
    uint32_t addressModeV = 0;
    uint32_t borderColor = 0;
};

struct ResourceInfo {
    uint32_t index = 0;
    vgflib::ResourceCategory category = vgflib::ResourceCategory::INPUT;
    std::optional<vk::DescriptorType> descriptorType;
    vk::Format format = vk::Format::eUndefined;
    vgflib::DataView<int64_t> shape;
    vgflib::DataView<int64_t> stride;
    std::optional<SamplerInfo> sampler;
    std::optional<uint32_t> aliasGroupId;
};

struct ConstantInfo {
    uint32_t index = 0;
    uint32_t resourceIndex = 0;
    vk::Format format = vk::Format::eUndefined;
    vgflib::DataView<int64_t> shape;
    vgflib::DataView<int64_t> stride;
    vgflib::DataView<uint8_t> data;
    int64_t sparsityDimension = -1;
};

struct PushConstantRangeInfo {
    uint32_t stageFlags = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
};

/*******************************************************************************
 * VGF
 *******************************************************************************/

class VGF {
  public:
    explicit VGF(const std::filesystem::path &vgf);
    explicit VGF(const void *data, std::size_t size);
    ~VGF();

    VGF(const VGF &) = delete;
    VGF &operator=(const VGF &) = delete;
    VGF(VGF &&) noexcept;
    VGF &operator=(VGF &&) noexcept;

    std::vector<DescriptorBindingInfo> getDescriptorBindings(uint32_t segmentIndex) const;

    uint32_t getNumSegments() const;
    uint32_t getNumSPIRVModules() const;
    uint32_t getNumResources() const;
    uint32_t getNumConstants() const;
    uint32_t getNumConstants(uint32_t segmentIndex) const;

    SegmentInfo getSegment(uint32_t segmentIndex) const;
    SPIRVModule getSPIRVModule(uint32_t moduleIndex) const;
    ResourceInfo getResource(uint32_t resourceIndex) const;
    vgflib::DataView<uint32_t> getConstantIndexes(uint32_t segmentIndex) const;
    ConstantInfo getConstant(uint32_t segmentIndex, uint32_t index) const;

    vgflib::DataView<uint32_t> getDispatchShape(uint32_t segmentIndex) const;

    uint32_t getNumPushConstantRanges(uint32_t segmentIndex) const;
    PushConstantRangeInfo getPushConstantRange(uint32_t segmentIndex, uint32_t index) const;

  private:
    std::unique_ptr<MemoryMap> mapped_;
    std::unique_ptr<vgflib::HeaderDecoder> header_;
    std::unique_ptr<vgflib::ModuleTableDecoder> moduleTable_;
    std::unique_ptr<vgflib::ModelSequenceTableDecoder> modelSequenceTable_;
    std::unique_ptr<vgflib::ModelResourceTableDecoder> modelResourceTable_;
    std::unique_ptr<vgflib::ConstantDecoder> constants_;
};

} // namespace mlsdk::vgf_runtime
