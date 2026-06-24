/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "iresource.hpp"
#include "scenario_runner.hpp"
#include "types.hpp"

#include "vgf-utils/memory_map.hpp"
#include "vgf/decoder.hpp"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mlsdk::scenariorunner {

class DataManager;

struct ResourceAlias {
    Guid group;
    Guid resource;
    ResourceIdType resourceType;
};

class VgfView {
  public:
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
    std::string getGLSLModuleCode(uint32_t segmentIndex) const;
    std::string getHLSLModuleCode(uint32_t segmentIndex) const;
    vgflib::DataView<uint32_t> getDispatchShape(uint32_t segmentIndex) const;

    vgflib::DataView<uint32_t> getSegmentConstantIndexes(uint32_t segmentIndex) const;
    vgflib::FormatType getConstantFormat(uint32_t constantIndex) const;
    int64_t getConstantSparsityDimension(uint32_t constantIndex) const;
    vgflib::DataView<int64_t> getConstantShape(uint32_t constantIndex) const;
    vgflib::DataView<uint8_t> getConstantData(uint32_t constantIndex) const;

    std::vector<TypedBinding> resolveBindings(uint32_t segmentIndex, const DataManager &dataManager,
                                              const std::vector<TypedBinding> &externalBindings) const;
    std::vector<ResourceAlias> getResourceAliases(const std::vector<TypedBinding> &externalBindings) const;
    void createIntermediateResources(IResourceCreator &creator) const;

  private:
    // Map of (set, binding) to vgfMrtIndex
    using MrtIndexes = std::map<std::tuple<uint32_t, uint32_t>, uint32_t>;

    std::pair<std::vector<TypedBinding>, MrtIndexes> getBindings(uint32_t segmentIndex) const;
    void validateResource(const IResourceViewer &resourceViewer, uint32_t vgfMrtIndex, std::string_view vgfDirection,
                          uint32_t vgfSlotIndex) const;

    std::string _vgfFileName;
    std::unique_ptr<MemoryMap> mapped;
    std::unique_ptr<vgflib::ModuleTableDecoder> moduleTableDecoder;
    std::unique_ptr<vgflib::ModelSequenceTableDecoder> sequenceTableDecoder;
    std::unique_ptr<vgflib::ModelResourceTableDecoder> resourceTableDecoder;
    std::unique_ptr<vgflib::ConstantDecoder> constantTableDecoder;

    VgfView(std::string vgfFileName, std::unique_ptr<MemoryMap> mapped,
            std::unique_ptr<vgflib::ModuleTableDecoder> moduleTableDecoder,
            std::unique_ptr<vgflib::ModelSequenceTableDecoder> sequenceTableDecoder,
            std::unique_ptr<vgflib::ModelResourceTableDecoder> resourceTableDecoder,
            std::unique_ptr<vgflib::ConstantDecoder> constantTableDecoder);
};
} // namespace mlsdk::scenariorunner
