/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "types.hpp"
#include "vgf-utils/memory_map.hpp"
#include "vgf/decoder.hpp"

#include <memory>

namespace mlsdk::scenariorunner {

class Context;
class DataManager;

class VgfView {
  public:
    VgfView(std::unique_ptr<MemoryMap> &&mapped, std::unique_ptr<vgflib::ModuleTableDecoder> &&moduleTableDecoder,
            std::unique_ptr<vgflib::ModelSequenceTableDecoder> &&sequenceTableDecoder,
            std::unique_ptr<vgflib::ModelResourceTableDecoder> &&resourceTableDecoder,
            std::unique_ptr<vgflib::ConstantDecoder> &&constantTableDecoder)
        : mapped(std::move(mapped)), moduleTableDecoder(std::move(moduleTableDecoder)),
          sequenceTableDecoder(std::move(sequenceTableDecoder)), resourceTableDecoder(std::move(resourceTableDecoder)),
          constantTableDecoder(std::move(constantTableDecoder)) {}
    VgfView() = default;

    size_t getNumSegments() const;
    ModuleType getSegmentType(uint32_t segmentIndex) const;

    bool hasSPVModule(uint32_t segmentIndex) const;
    std::string getSPVModuleName(uint32_t segmentIndex) const;
    std::string getSPVModuleEntryPoint(uint32_t segmentIndex) const;
    vgflib::DataView<uint32_t> getSPVModule(uint32_t segmentIndex) const;
    vgflib::DataView<uint32_t> getDispatchShape(uint32_t segmentIndex) const;

    vgflib::DataView<uint32_t> getSegmentConstantIndexes(uint32_t segmentIndex) const;
    vgflib::FormatType getConstantFormat(uint32_t constantIndex) const;
    int64_t getConstantSparsityDimension(uint32_t constantIndex) const;
    vgflib::DataView<int64_t> getConstantShape(uint32_t constantIndex) const;
    vgflib::DataView<uint8_t> getConstantData(uint32_t constantIndex) const;

    std::vector<BindingDesc> resolveBindings(uint32_t segmentIndex, const DataManager &dataManager,
                                             const std::vector<BindingDesc> &externalBindings) const;
    void createIntermediateResources(Context &ctx, DataManager &dataManager) const;

  private:
    void validateResource(const DataManager &dataManager, uint32_t vgfMrtIndex, Guid externalResourceRef) const;

    std::unique_ptr<MemoryMap> mapped;
    std::unique_ptr<vgflib::ModuleTableDecoder> moduleTableDecoder;
    std::unique_ptr<vgflib::ModelSequenceTableDecoder> sequenceTableDecoder;
    std::unique_ptr<vgflib::ModelResourceTableDecoder> resourceTableDecoder;
    std::unique_ptr<vgflib::ConstantDecoder> constantTableDecoder;
};
} // namespace mlsdk::scenariorunner
