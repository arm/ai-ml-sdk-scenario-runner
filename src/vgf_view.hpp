/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "scenario_runner/types.hpp"

#include "iresource.hpp"

#include "vgf-utils/memory_map.hpp"
#include "vgf/decoder.hpp"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlsdk::scenariorunner {

class DataManager;
enum class ModuleType { SHADER, GRAPH };

struct VgfResourceCreationResult {
    // VGF model resource table index to the created runtime resource ID.
    std::unordered_map<uint32_t, MemoryResourceId> intermediateResources;
    // VGF alias group ID to its created runtime resource IDs. Ordered for deterministic group creation.
    std::map<uint32_t, std::vector<MemoryResourceId>> memoryGroups;
};

class VgfView {
  public:
    using DescriptorBinding = std::pair<uint32_t, uint32_t>;
    using DescriptorBindingMrtMap = std::map<DescriptorBinding, uint32_t>;

    static VgfView createVgfView(std::string vgfFile);

    size_t getNumSegments() const;
    ModuleType getSegmentType(uint32_t segmentIndex) const;
    std::string getSegmentName(uint32_t segmentIndex) const;

    bool hasSPVModule(uint32_t segmentIndex) const;
    bool hasGLSLModule(uint32_t segmentIndex) const;
    bool hasHLSLModule(uint32_t segmentIndex) const;
    std::string getModuleName(uint32_t segmentIndex) const;
    std::string getModuleEntryPoint(uint32_t segmentIndex) const;
    vgflib::DataView<uint32_t> getSPVModuleCode(uint32_t segmentIndex) const;
    void setSPVModuleCode(uint32_t segmentIndex, std::vector<uint32_t> moduleCode);
    std::string getGLSLModuleCode(uint32_t segmentIndex) const;
    std::string getHLSLModuleCode(uint32_t segmentIndex) const;
    vgflib::DataView<uint32_t> getDispatchShape(uint32_t segmentIndex) const;
    DescriptorBindingMrtMap getSegmentMrtIndexes(uint32_t segmentIndex) const;
    std::optional<uint32_t> getModelInterfaceMrtIndex(uint32_t bindingId) const;
    vgflib::DataView<int64_t> getTensorShape(uint32_t mrtIndex) const;
    void setTensorShape(uint32_t mrtIndex, std::vector<int64_t> shape);

    std::vector<vgflib::GraphConstantBinding> getSegmentConstantBindings(uint32_t segmentIndex) const;
    vgflib::FormatType getConstantFormat(uint32_t constantIndex) const;
    int64_t getConstantSparsityDimension(uint32_t constantIndex) const;
    vgflib::DataView<int64_t> getConstantShape(uint32_t constantIndex) const;
    vgflib::DataView<uint8_t> getConstantData(uint32_t constantIndex) const;

    std::vector<TypedBinding>
    resolveBindings(uint32_t segmentIndex, const DataManager &dataManager,
                    const std::vector<TypedBinding> &externalBindings,
                    const std::unordered_map<uint32_t, MemoryResourceId> &intermediates) const;
    std::optional<uint32_t> getModelResourceAliasGroup(uint32_t bindingId) const;
    VgfResourceCreationResult createIntermediateResources(IResourceCreator &creator) const;

  private:
    struct VgfBinding {
        uint32_t set;
        uint32_t id;
        // Index into the VGF model resource table.
        uint32_t resourceIndex;
        vk::DescriptorType descriptorType;
    };

    std::vector<VgfBinding> getBindings(uint32_t segmentIndex) const;
    void validateResource(const IResourceViewer &resourceViewer, uint32_t vgfMrtIndex, std::string_view vgfDirection,
                          uint32_t vgfSlotIndex) const;

    std::string _vgfFileName;
    std::unique_ptr<MemoryMap> mapped;
    std::unique_ptr<vgflib::ModuleTableDecoder> moduleTableDecoder;
    std::unique_ptr<vgflib::ModelSequenceTableDecoder> sequenceTableDecoder;
    std::unique_ptr<vgflib::ModelResourceTableDecoder> resourceTableDecoder;
    std::unique_ptr<vgflib::ConstantDecoder> constantTableDecoder;
    std::map<uint32_t, std::vector<int64_t>> resolvedMrtShapes;
    std::map<uint32_t, std::vector<uint32_t>> shapedSpirvModules;

    VgfView(std::string vgfFileName, std::unique_ptr<MemoryMap> mapped,
            std::unique_ptr<vgflib::ModuleTableDecoder> moduleTableDecoder,
            std::unique_ptr<vgflib::ModelSequenceTableDecoder> sequenceTableDecoder,
            std::unique_ptr<vgflib::ModelResourceTableDecoder> resourceTableDecoder,
            std::unique_ptr<vgflib::ConstantDecoder> constantTableDecoder);
};
} // namespace mlsdk::scenariorunner
