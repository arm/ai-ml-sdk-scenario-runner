/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

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

/** @brief Description of a decoded SPIR-V module contained in a VGF file. */
struct SPIRVModule {
    uint32_t index = 0;
    std::string name;
    std::string entryPoint;
    vgflib::DataView<uint32_t> code;
};

/** @brief Metadata describing one executable segment in a VGF file. */
struct SegmentInfo {
    uint32_t index = 0;
    std::string name;
    vgflib::ModuleType type = vgflib::ModuleType::GRAPH;
    uint32_t moduleIndex = 0;
    std::string moduleName;
    std::string entryPoint;
};

/** @brief Descriptor binding used by a segment resource. */
struct DescriptorBindingInfo {
    uint32_t set = 0;
    uint32_t binding = 0;
    uint32_t resourceIndex = 0;
    vk::DescriptorType descriptorType = {};
    vgflib::ResourceCategory resourceCategory = vgflib::ResourceCategory::INPUT;
};

/** @brief Metadata describing a VGF resource entry. */
struct ResourceInfo {
    struct SamplerConfig {
        vk::Filter minFilter = vk::Filter::eNearest;
        vk::Filter magFilter = vk::Filter::eNearest;
        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eClampToEdge;
        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eClampToEdge;
        vk::BorderColor borderColor = vk::BorderColor::eFloatTransparentBlack;
    };

    uint32_t index = 0;
    vgflib::ResourceCategory category = vgflib::ResourceCategory::INPUT;
    std::optional<vk::DescriptorType> descriptorType;
    vk::Format format = vk::Format::eUndefined;
    vgflib::DataView<int64_t> shape;
    vgflib::DataView<int64_t> stride;
    std::optional<uint32_t> aliasGroupId;
    std::optional<SamplerConfig> samplerConfig;
};

/** @brief Decoded constant payload and metadata for a segment resource. */
struct ConstantInfo {
    uint32_t index = 0;
    uint32_t resourceIndex = 0;
    vk::Format format = vk::Format::eUndefined;
    vgflib::DataView<int64_t> shape;
    vgflib::DataView<int64_t> stride;
    vgflib::DataView<uint8_t> data;
    int64_t sparsityDimension = -1;
};

/**
 * @brief Decodes and exposes metadata and binary payloads stored in a VGF file.
 *
 * A `VGF` instance provides read-only access to modules, segments, resources,
 * constants, and dispatch metadata referenced by the file.
 */
class VGF {
  public:
    /** @brief Map and decode a VGF file from disk. */
    explicit VGF(const std::filesystem::path &vgf);

    /** @brief Decode a VGF blob from memory. */
    explicit VGF(const void *data, std::size_t size);
    ~VGF();

    VGF(const VGF &) = delete;
    VGF &operator=(const VGF &) = delete;
    VGF(VGF &&) noexcept;
    VGF &operator=(VGF &&) noexcept;

    /** @brief Return descriptor bindings used by the segment at @p segmentIndex. */
    std::vector<DescriptorBindingInfo> getDescriptorBindings(uint32_t segmentIndex) const;

    /** @brief Return the number of segments described by the VGF. */
    uint32_t getNumSegments() const;

    /** @brief Return the number of SPIR-V modules stored in the VGF. */
    uint32_t getNumSPIRVModules() const;

    /** @brief Return the number of resource entries stored in the VGF. */
    uint32_t getNumResources() const;

    /** @brief Return the total number of constants stored in the VGF. */
    uint32_t getNumConstants() const;

    /** @brief Return the number of constants referenced by a segment. */
    uint32_t getNumConstants(uint32_t segmentIndex) const;

    /** @brief Return metadata for the segment at @p segmentIndex. */
    SegmentInfo getSegment(uint32_t segmentIndex) const;

    /** @brief Return metadata and code for the module at @p moduleIndex. */
    SPIRVModule getSPIRVModule(uint32_t moduleIndex) const;

    /** @brief Return metadata for the resource at @p resourceIndex. */
    ResourceInfo getResource(uint32_t resourceIndex) const;

    /** @brief Return constant metadata and data for a segment-local constant index. */
    ConstantInfo getConstant(uint32_t segmentIndex, uint32_t index) const;

    /** @brief Return the dispatch shape for the segment at @p segmentIndex. */
    vgflib::DataView<uint32_t> getDispatchShape(uint32_t segmentIndex) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mlsdk::vgf_runtime
