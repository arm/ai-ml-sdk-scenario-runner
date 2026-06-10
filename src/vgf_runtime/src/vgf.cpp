/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf.hpp"

#include "vgf-utils/memory_map.hpp"
#include "vgf/decoder.hpp"

#include <string>

namespace mlsdk::vgf_runtime {
namespace {

void decode(const void *data, std::size_t size, std::unique_ptr<vgflib::HeaderDecoder> &header,
            std::unique_ptr<vgflib::ModuleTableDecoder> &moduleTable,
            std::unique_ptr<vgflib::ModelSequenceTableDecoder> &modelSequenceTable,
            std::unique_ptr<vgflib::ModelResourceTableDecoder> &modelResourceTable,
            std::unique_ptr<vgflib::ConstantDecoder> &constants) {
    header = vgflib::CreateHeaderDecoder(data, vgflib::HeaderSize(), size);

    const auto *bytes = static_cast<const std::byte *>(data);
    moduleTable =
        vgflib::CreateModuleTableDecoder(bytes + header->GetModuleTableOffset(), header->GetModuleTableSize());
    modelSequenceTable = vgflib::CreateModelSequenceTableDecoder(bytes + header->GetModelSequenceTableOffset(),
                                                                 header->GetModelSequenceTableSize());
    modelResourceTable = vgflib::CreateModelResourceTableDecoder(bytes + header->GetModelResourceTableOffset(),
                                                                 header->GetModelResourceTableSize());
    constants = vgflib::CreateConstantDecoder(bytes + header->GetConstantsOffset(), header->GetConstantsSize());
}

} // namespace

VGF::VGF(const std::filesystem::path &vgf) : mapped_(std::make_unique<MemoryMap>(vgf.string())) {
    decode(mapped_->ptr(), mapped_->size(), header_, moduleTable_, modelSequenceTable_, modelResourceTable_,
           constants_);
}

VGF::VGF(const void *data, std::size_t size) {
    decode(data, size, header_, moduleTable_, modelSequenceTable_, modelResourceTable_, constants_);
}

VGF::~VGF() = default;

VGF::VGF(VGF &&) noexcept = default;

VGF &VGF::operator=(VGF &&) noexcept = default;

std::vector<DescriptorBindingInfo> VGF::getDescriptorBindings(uint32_t segmentIndex) const {
    std::vector<DescriptorBindingInfo> bindings;
    for (uint32_t descIdx = 0; descIdx < modelSequenceTable_->getSegmentDescriptorSetInfosSize(segmentIndex);
         ++descIdx) {
        const uint32_t set = modelSequenceTable_->getSegmentDescriptorSetIndex(segmentIndex, descIdx);
        const auto *handle = modelSequenceTable_->getDescriptorBindingSlotsHandle(segmentIndex, descIdx);
        for (uint32_t slot = 0; slot < modelSequenceTable_->getBindingsSize(handle); ++slot) {
            const uint32_t resourceIndex = modelSequenceTable_->getBindingSlotMrtIndex(handle, slot);
            bindings.push_back({set, modelSequenceTable_->getBindingSlotBinding(handle, slot), resourceIndex,
                                vk::DescriptorType(*modelResourceTable_->getDescriptorType(resourceIndex)),
                                modelResourceTable_->getCategory(resourceIndex)});
        }
    }
    return bindings;
}

uint32_t VGF::getNumSegments() const { return static_cast<uint32_t>(modelSequenceTable_->modelSequenceTableSize()); }

uint32_t VGF::getNumSPIRVModules() const { return static_cast<uint32_t>(moduleTable_->size()); }

uint32_t VGF::getNumResources() const { return static_cast<uint32_t>(modelResourceTable_->size()); }

uint32_t VGF::getNumConstants() const { return static_cast<uint32_t>(constants_->size()); }

uint32_t VGF::getNumConstants(uint32_t segmentIndex) const {
    return static_cast<uint32_t>(modelSequenceTable_->getSegmentConstantIndexes(segmentIndex).size());
}

SegmentInfo VGF::getSegment(uint32_t segmentIndex) const {
    const uint32_t moduleIndex = modelSequenceTable_->getSegmentModuleIndex(segmentIndex);
    return {segmentIndex,
            std::string(modelSequenceTable_->getSegmentName(segmentIndex)),
            modelSequenceTable_->getSegmentType(segmentIndex),
            moduleIndex,
            std::string(moduleTable_->getModuleName(moduleIndex)),
            std::string(moduleTable_->getModuleEntryPoint(moduleIndex))};
}

SPIRVModule VGF::getSPIRVModule(uint32_t moduleIndex) const {
    return {moduleIndex, std::string(moduleTable_->getModuleName(moduleIndex)),
            std::string(moduleTable_->getModuleEntryPoint(moduleIndex)), moduleTable_->getSPIRVModuleCode(moduleIndex)};
}

ResourceInfo VGF::getResource(uint32_t resourceIndex) const {
    const auto descriptorType = modelResourceTable_->getDescriptorType(resourceIndex);
    ResourceInfo resource{resourceIndex,
                          modelResourceTable_->getCategory(resourceIndex),
                          descriptorType ? std::optional<vk::DescriptorType>(vk::DescriptorType(*descriptorType))
                                         : std::nullopt,
                          vk::Format(modelResourceTable_->getVkFormat(resourceIndex)),
                          modelResourceTable_->getTensorShape(resourceIndex),
                          modelResourceTable_->getTensorStride(resourceIndex),
                          std::nullopt,
                          std::nullopt};
    if (const auto *sampler = modelResourceTable_->getSamplerConfigHandle(resourceIndex)) {
        resource.sampler = SamplerInfo{modelResourceTable_->getSamplerConfigMinFilter(sampler),
                                       modelResourceTable_->getSamplerConfigMagFilter(sampler),
                                       modelResourceTable_->getSamplerConfigAddressModeU(sampler),
                                       modelResourceTable_->getSamplerConfigAddressModeV(sampler),
                                       modelResourceTable_->getSamplerConfigBorderColor(sampler)};
    }
    resource.aliasGroupId = modelResourceTable_->getAliasGroupId(resourceIndex);
    return resource;
}

vgflib::DataView<uint32_t> VGF::getConstantIndexes(uint32_t segmentIndex) const {
    return modelSequenceTable_->getSegmentConstantIndexes(segmentIndex);
}

ConstantInfo VGF::getConstant(uint32_t segmentIndex, uint32_t index) const {
    const auto constantIndexes = modelSequenceTable_->getSegmentConstantIndexes(segmentIndex);
    const uint32_t constantIndex = constantIndexes[index];
    const uint32_t resourceIndex = constants_->getConstantMrtIndex(constantIndex);
    return {constantIndex,
            resourceIndex,
            vk::Format(modelResourceTable_->getVkFormat(resourceIndex)),
            modelResourceTable_->getTensorShape(resourceIndex),
            modelResourceTable_->getTensorStride(resourceIndex),
            constants_->getConstant(constantIndex),
            constants_->getConstantSparsityDimension(constantIndex)};
}

vgflib::DataView<uint32_t> VGF::getDispatchShape(uint32_t segmentIndex) const {
    return modelSequenceTable_->getSegmentDispatchShape(segmentIndex);
}

uint32_t VGF::getNumPushConstantRanges(uint32_t segmentIndex) const {
    const auto *handle = modelSequenceTable_->getSegmentPushConstRange(segmentIndex);
    return static_cast<uint32_t>(modelSequenceTable_->getPushConstRangesSize(handle));
}

PushConstantRangeInfo VGF::getPushConstantRange(uint32_t segmentIndex, uint32_t index) const {
    const auto *handle = modelSequenceTable_->getSegmentPushConstRange(segmentIndex);
    return {modelSequenceTable_->getPushConstRangeStageFlags(handle, index),
            modelSequenceTable_->getPushConstRangeOffset(handle, index),
            modelSequenceTable_->getPushConstRangeSize(handle, index)};
}

} // namespace mlsdk::vgf_runtime
