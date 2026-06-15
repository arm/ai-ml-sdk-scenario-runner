/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <vgf_runtime/runtime.hpp>

#include "vgf-utils/memory_map.hpp"
#include "vgf/decoder.hpp"

#include <string>

namespace mlsdk::vgf_runtime {

struct VGF::Impl {
    std::unique_ptr<MemoryMap> mapped;
    std::unique_ptr<vgflib::HeaderDecoder> header;
    std::unique_ptr<vgflib::ModuleTableDecoder> moduleTable;
    std::unique_ptr<vgflib::ModelSequenceTableDecoder> modelSequenceTable;
    std::unique_ptr<vgflib::ModelResourceTableDecoder> modelResourceTable;
    std::unique_ptr<vgflib::ConstantDecoder> constants;

    void decode(const std::string &filename) {
        mapped = std::make_unique<MemoryMap>(filename);
        decode(mapped->ptr(), mapped->size());
    }

    void decode(const void *data, std::size_t size) {
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
};

VGF::VGF(const std::filesystem::path &vgf) : impl_(std::make_unique<Impl>()) { impl_->decode(vgf.string()); }

VGF::VGF(const void *data, std::size_t size) : impl_(std::make_unique<Impl>()) { impl_->decode(data, size); }

VGF::~VGF() = default;

VGF::VGF(VGF &&) noexcept = default;

VGF &VGF::operator=(VGF &&) noexcept = default;

std::vector<DescriptorBindingInfo> VGF::getDescriptorBindings(uint32_t segmentIndex) const {
    std::vector<DescriptorBindingInfo> bindings;
    for (uint32_t descIdx = 0; descIdx < impl_->modelSequenceTable->getSegmentDescriptorSetInfosSize(segmentIndex);
         ++descIdx) {
        const uint32_t set = impl_->modelSequenceTable->getSegmentDescriptorSetIndex(segmentIndex, descIdx);
        const auto *handle = impl_->modelSequenceTable->getDescriptorBindingSlotsHandle(segmentIndex, descIdx);
        for (uint32_t slot = 0; slot < impl_->modelSequenceTable->getBindingsSize(handle); ++slot) {
            const uint32_t resourceIndex = impl_->modelSequenceTable->getBindingSlotMrtIndex(handle, slot);
            bindings.push_back({set, impl_->modelSequenceTable->getBindingSlotBinding(handle, slot), resourceIndex,
                                vk::DescriptorType(*impl_->modelResourceTable->getDescriptorType(resourceIndex)),
                                impl_->modelResourceTable->getCategory(resourceIndex)});
        }
    }
    return bindings;
}

uint32_t VGF::getNumSegments() const {
    return static_cast<uint32_t>(impl_->modelSequenceTable->modelSequenceTableSize());
}

uint32_t VGF::getNumSPIRVModules() const { return static_cast<uint32_t>(impl_->moduleTable->size()); }

uint32_t VGF::getNumResources() const { return static_cast<uint32_t>(impl_->modelResourceTable->size()); }

uint32_t VGF::getNumConstants() const { return static_cast<uint32_t>(impl_->constants->size()); }

uint32_t VGF::getNumConstants(uint32_t segmentIndex) const {
    return static_cast<uint32_t>(impl_->modelSequenceTable->getSegmentConstantIndexes(segmentIndex).size());
}

SegmentInfo VGF::getSegment(uint32_t segmentIndex) const {
    const uint32_t moduleIndex = impl_->modelSequenceTable->getSegmentModuleIndex(segmentIndex);
    return {segmentIndex,
            std::string(impl_->modelSequenceTable->getSegmentName(segmentIndex)),
            impl_->modelSequenceTable->getSegmentType(segmentIndex),
            moduleIndex,
            std::string(impl_->moduleTable->getModuleName(moduleIndex)),
            std::string(impl_->moduleTable->getModuleEntryPoint(moduleIndex))};
}

SPIRVModule VGF::getSPIRVModule(uint32_t moduleIndex) const {
    return {moduleIndex, std::string(impl_->moduleTable->getModuleName(moduleIndex)),
            std::string(impl_->moduleTable->getModuleEntryPoint(moduleIndex)),
            impl_->moduleTable->getSPIRVModuleCode(moduleIndex)};
}

ResourceInfo VGF::getResource(uint32_t resourceIndex) const {
    const auto descriptorType = impl_->modelResourceTable->getDescriptorType(resourceIndex);
    ResourceInfo resource{resourceIndex,
                          impl_->modelResourceTable->getCategory(resourceIndex),
                          descriptorType ? std::optional<vk::DescriptorType>(vk::DescriptorType(*descriptorType))
                                         : std::nullopt,
                          vk::Format(impl_->modelResourceTable->getVkFormat(resourceIndex)),
                          impl_->modelResourceTable->getTensorShape(resourceIndex),
                          impl_->modelResourceTable->getTensorStride(resourceIndex),
                          std::nullopt,
                          std::nullopt};
    resource.aliasGroupId = impl_->modelResourceTable->getAliasGroupId(resourceIndex);
    if (const auto samplerConfig = impl_->modelResourceTable->getSamplerConfigHandle(resourceIndex)) {
        resource.samplerConfig = ResourceInfo::SamplerConfig{
            vk::Filter(impl_->modelResourceTable->getSamplerConfigMinFilter(samplerConfig)),
            vk::Filter(impl_->modelResourceTable->getSamplerConfigMagFilter(samplerConfig)),
            vk::SamplerAddressMode(impl_->modelResourceTable->getSamplerConfigAddressModeU(samplerConfig)),
            vk::SamplerAddressMode(impl_->modelResourceTable->getSamplerConfigAddressModeV(samplerConfig)),
            vk::BorderColor(impl_->modelResourceTable->getSamplerConfigBorderColor(samplerConfig))};
    }
    return resource;
}

ConstantInfo VGF::getConstant(uint32_t segmentIndex, uint32_t index) const {
    const auto constantIndexes = impl_->modelSequenceTable->getSegmentConstantIndexes(segmentIndex);
    const uint32_t constantIndex = constantIndexes[index];
    const uint32_t resourceIndex = impl_->constants->getConstantMrtIndex(constantIndex);
    return {constantIndex,
            resourceIndex,
            vk::Format(impl_->modelResourceTable->getVkFormat(resourceIndex)),
            impl_->modelResourceTable->getTensorShape(resourceIndex),
            impl_->modelResourceTable->getTensorStride(resourceIndex),
            impl_->constants->getConstant(constantIndex),
            impl_->constants->getConstantSparsityDimension(constantIndex)};
}

vgflib::DataView<uint32_t> VGF::getDispatchShape(uint32_t segmentIndex) const {
    return impl_->modelSequenceTable->getSegmentDispatchShape(segmentIndex);
}

} // namespace mlsdk::vgf_runtime
