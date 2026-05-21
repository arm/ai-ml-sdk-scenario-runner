/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vulkan/vulkan.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// From vgf-utils
class MemoryMap;

namespace mlsdk::vgflib {

class ConstantDecoder;
class HeaderDecoder;
class ModelResourceTableDecoder;
class ModelSequenceTableDecoder;
class ModuleTableDecoder;

} // namespace mlsdk::vgflib

namespace mlsdk::vgf_runtime {

template <typename T> class DataView {
  public:
    constexpr DataView() noexcept = default;
    constexpr DataView(const T *data, std::size_t size) noexcept : data_(data), size_(size) {}

    constexpr const T *data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr const T *begin() const noexcept { return data_; }
    constexpr const T *end() const noexcept { return data_ == nullptr ? nullptr : data_ + size_; }
    constexpr const T &operator[](std::size_t index) const noexcept { return data_[index]; }

  private:
    const T *data_ = nullptr;
    std::size_t size_ = 0;
};

struct SPIRVModule {
    uint32_t index = 0;
    std::string name;
    std::string entryPoint;
    DataView<uint32_t> code;
};

enum class ModuleType : uint32_t {
    UNKNOWN = 0,
    GRAPH = 1,
    SHADER = 2,
};

enum class ResourceCategory : uint32_t {
    UNKNOWN = 0,
    INPUT = 1,
    OUTPUT = 2,
    INTERMEDIATE = 3,
    CONSTANT = 4,
};

/*******************************************************************************
 * VGF File sections
 *******************************************************************************/

struct SegmentInfo {
    uint32_t index = 0;
    std::string name;
    ModuleType type = ModuleType::GRAPH;
    uint32_t moduleIndex = 0;
    std::string moduleName;
    std::string entryPoint;
};

struct DescriptorBindingInfo {
    uint32_t set = 0;
    uint32_t binding = 0;
    uint32_t resourceIndex = 0;
    vk::DescriptorType descriptorType = {};
    ResourceCategory resourceCategory = ResourceCategory::INPUT;
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
    ResourceCategory category = ResourceCategory::INPUT;
    std::optional<vk::DescriptorType> descriptorType;
    vk::Format format = vk::Format::eUndefined;
    DataView<int64_t> shape;
    DataView<int64_t> stride;
    std::optional<SamplerInfo> sampler;
};

struct ConstantInfo {
    uint32_t index = 0;
    uint32_t resourceIndex = 0;
    vk::Format format = vk::Format::eUndefined;
    DataView<int64_t> shape;
    DataView<int64_t> stride;
    DataView<uint8_t> data;
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
    DataView<uint32_t> getConstantIndexes(uint32_t segmentIndex) const;
    ConstantInfo getConstant(uint32_t segmentIndex, uint32_t index) const;

    DataView<uint32_t> getDispatchShape(uint32_t segmentIndex) const;

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
