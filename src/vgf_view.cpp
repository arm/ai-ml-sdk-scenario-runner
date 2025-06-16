/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf_view.hpp"

#include "context.hpp"
#include "data_manager.hpp"

namespace mlsdk::scenariorunner {
namespace {
std::string categoryToSuffix(vgflib::ResourceCategory category) {
    switch (category) {
    case vgflib::ResourceCategory::INPUT:
        return "_input";
    case vgflib::ResourceCategory::OUTPUT:
        return "_output";
    case vgflib::ResourceCategory::INTERMEDIATE:
        return "_intermediate";
    case vgflib::ResourceCategory::CONSTANT:
        return "_constant";
    default:
        throw std::runtime_error("Unknown resource category");
    }
}

std::string createResourceGuidStr(uint32_t index, vgflib::ResourceCategory category) {
    return "Resource_" + std::to_string(index) + categoryToSuffix(category);
}
} // namespace

size_t VgfView::getNumSegments() const { return sequenceTableDecoder->modelSequenceTableSize(); }

ModuleType VgfView::getSegmentType(uint32_t segmentIndex) const {
    switch (sequenceTableDecoder->getSegmentType(segmentIndex)) {
    case vgflib::ModuleType::GRAPH:
        return ModuleType::GRAPH;
    case vgflib::ModuleType::COMPUTE:
        return ModuleType::SHADER;
    default:
        throw std::runtime_error("Unknown module type");
    }
}

bool VgfView::hasSPVModule(uint32_t segmentIndex) const {
    uint32_t moduleIndex = sequenceTableDecoder->getSegmentModuleIndex(segmentIndex);
    return moduleTableDecoder->hasSPIRV(moduleIndex);
}

std::string VgfView::getSPVModuleName(uint32_t segmentIndex) const {
    uint32_t moduleIndex = sequenceTableDecoder->getSegmentModuleIndex(segmentIndex);
    return std::string(moduleTableDecoder->getModuleName(moduleIndex));
}

std::string VgfView::getSPVModuleEntryPoint(uint32_t segmentIndex) const {
    uint32_t moduleIndex = sequenceTableDecoder->getSegmentModuleIndex(segmentIndex);
    return std::string(moduleTableDecoder->getModuleEntryPoint(moduleIndex));
}

vgflib::DataView<uint32_t> VgfView::getSPVModule(uint32_t segmentIndex) const {
    uint32_t moduleIndex = sequenceTableDecoder->getSegmentModuleIndex(segmentIndex);
    return moduleTableDecoder->getModuleCode(moduleIndex);
}

vgflib::DataView<uint32_t> VgfView::getDispatchShape(uint32_t segmentIndex) const {
    return sequenceTableDecoder->getSegmentDispatchShape(segmentIndex);
}

vgflib::DataView<uint32_t> VgfView::getSegmentConstantIndexes(uint32_t segmentIndex) const {
    return sequenceTableDecoder->getSegmentConstantIndexes(segmentIndex);
}

vgflib::FormatType VgfView::getConstantFormat(uint32_t constantIndex) const {
    uint32_t mrtIndex = constantTableDecoder->getConstantMrtIndex(constantIndex);
    if (resourceTableDecoder->getCategory(mrtIndex) != vgflib::ResourceCategory::CONSTANT) {
        throw std::runtime_error("Resource not marked as constant");
    }
    return resourceTableDecoder->getVkFormat(mrtIndex);
}

int64_t VgfView::getConstantSparsityDimension(uint32_t constantIndex) const {
    return constantTableDecoder->getConstantSparsityDimension(constantIndex);
}

vgflib::DataView<int64_t> VgfView::getConstantShape(uint32_t constantIndex) const {
    uint32_t mrtIndex = constantTableDecoder->getConstantMrtIndex(constantIndex);
    if (resourceTableDecoder->getCategory(mrtIndex) != vgflib::ResourceCategory::CONSTANT) {
        throw std::runtime_error("Resource not marked as constant");
    }
    return resourceTableDecoder->getTensorShape(mrtIndex);
}

vgflib::DataView<uint8_t> VgfView::getConstantData(uint32_t constantIndex) const {
    return constantTableDecoder->getConstant(constantIndex);
}

std::vector<BindingDesc> VgfView::resolveBindings(uint32_t segmentIndex, const DataManager &dataManager,
                                                  const std::vector<BindingDesc> &externalBindings) const {
    // Get segment binding infos
    std::vector<BindingDesc> bindings;
    auto descSetSize = sequenceTableDecoder->getSegmentDescriptorSetInfosSize(segmentIndex);
    std::map<std::tuple<uint32_t, uint32_t>, uint32_t> mrtIndexes;
    // For each segment descriptor set:
    for (uint32_t set = 0; set < descSetSize; ++set) {
        auto handle = sequenceTableDecoder->getDescriptorBindingSlotsHandle(segmentIndex, set);
        // For each descriptor set binding:
        for (uint32_t slot = 0; slot < sequenceTableDecoder->getBindingsSize(handle); ++slot) {
            auto bindingId = sequenceTableDecoder->getBindingSlotBinding(handle, slot);
            auto mrtIndex = sequenceTableDecoder->getBindingSlotMrtIndex(handle, slot);
            auto guidStr = createResourceGuidStr(bindingId, resourceTableDecoder->getCategory(mrtIndex));
            bindings.emplace_back(BindingDesc(set, bindingId, guidStr));
            mrtIndexes.insert({{set, bindingId}, mrtIndex});
        }
    }

    for (const auto &externalBinding : externalBindings) {
        if (!(dataManager.hasTensor(externalBinding.resourceRef) ||
              dataManager.hasBuffer(externalBinding.resourceRef))) {
            // All tensors should have been created when this function is called
            throw std::runtime_error("No resource with this guid found");
        }

        for (auto &binding : bindings) {
            if (binding.set == externalBinding.set && binding.id == externalBinding.id) {
                binding.resourceRef = externalBinding.resourceRef;

                auto mrtIndexSearch = mrtIndexes.find({externalBinding.set, externalBinding.id});
                if (mrtIndexSearch == mrtIndexes.end()) {
                    throw std::runtime_error("No resource found in MRT Table");
                }
                validateResource(dataManager, mrtIndexSearch->second, externalBinding.resourceRef);
            }
        }
    }
    return bindings;
}

void VgfView::validateResource(const DataManager &dataManager, uint32_t vgfMrtIndex, Guid externalResourceRef) const {
    std::optional<vgflib::DescriptorType> expectedType = resourceTableDecoder->getDescriptorType(vgfMrtIndex);
    if (!expectedType.has_value()) {
        throw std::runtime_error("Descriptor type not found from VGF file");
    }

    constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6;
    constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000;

    switch (expectedType.value()) {
    case DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT: {
        // Check if resource types match
        if (!dataManager.hasBuffer(externalResourceRef)) {
            throw std::runtime_error("This VGF resource is linked to a buffer resource,"
                                     "but JSON file states otherwise");
        }
        const Buffer &buffer = dataManager.getBuffer(externalResourceRef);

        // Check if buffer sizes match
        auto dims = resourceTableDecoder->getTensorShape(vgfMrtIndex);
        auto expectedBufferSize = static_cast<uint32_t>(
            std::abs(std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>())));
        if (buffer.size() != expectedBufferSize) {
            throw std::runtime_error("Mismatch of buffer size declarations between JSON and VGF file");
        }
    } break;
    case DESCRIPTOR_TYPE_TENSOR_ARM: {
        // Check if resource types matches
        if (!dataManager.hasTensor(externalResourceRef)) {
            throw std::runtime_error("This VGF resource is linked to a tensor resource,"
                                     "but JSON file states otherwise");
        }
        const Tensor &tensor = dataManager.getTensor(externalResourceRef);
        const std::vector<int64_t> actualTensorShape =
            tensor.isRankConverted() ? std::vector<int64_t>(0) : tensor.shape();

        auto dims = resourceTableDecoder->getTensorShape(vgfMrtIndex);
        const std::vector<int64_t> expectedTensorShape(dims.begin(), dims.end());

        // Check if tensor shapes match
        if (actualTensorShape != expectedTensorShape) {
            throw std::runtime_error("Mismatch of tensor shape declarations "
                                     "between JSON and VGF file");
        }

        // Check if tensor data formats match
        auto format = resourceTableDecoder->getVkFormat(vgfMrtIndex);
        if (static_cast<int32_t>(tensor.dataType()) != format) {
            throw std::runtime_error("Mismatch of tensor data type declarations"
                                     "between JSON and VGF file");
        }
    } break;
    default:
        throw std::runtime_error(
            "No resource validation should be performed for resources different from tensors and buffers");
        break;
    }
}

void VgfView::createIntermediateResources(Context &ctx, DataManager &dataManager) const {
    // Iterate over all VGF Resources, create intermediates
    constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_UNKNOWN = 0;
    constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT = 6;
    constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000;

    size_t numResources = resourceTableDecoder->size();
    for (uint32_t resourceIndex = 0; resourceIndex < numResources; ++resourceIndex) {
        auto resourceCategory = resourceTableDecoder->getCategory(resourceIndex);
        if (resourceCategory == vgflib::ResourceCategory::INTERMEDIATE) {
            auto guidStr = createResourceGuidStr(resourceIndex, resourceCategory);
            auto type = resourceTableDecoder->getDescriptorType(resourceIndex);
            switch (type.value_or(DESCRIPTOR_TYPE_UNKNOWN)) {
            case (DESCRIPTOR_TYPE_STORAGE_BUFFER_EXT): {
                auto shape = resourceTableDecoder->getTensorShape(resourceIndex);
                auto bufferSize = static_cast<uint32_t>(
                    std::abs(std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>())));

                // Create Scenario Runner buffer resource
                BufferInfo info{guidStr, bufferSize};
                dataManager.createZeroedBuffer(guidStr, info);
            } break;
            case (DESCRIPTOR_TYPE_TENSOR_ARM): {
                auto shape = resourceTableDecoder->getTensorShape(resourceIndex);
                auto format = resourceTableDecoder->getVkFormat(resourceIndex);

                auto info = TensorInfo{guidStr, std::vector<int64_t>(shape.begin(), shape.end()), vk::Format(format),
                                       -1, false};

                // Create Scenario Runner tensor
                dataManager.createTensor(guidStr, info);
                auto &tensor = dataManager.getTensorMut(guidStr);
                tensor.allocateMemory(ctx);
            } break;
            default:
                throw std::runtime_error("Unknown resource type read from VGF file");
            }
        }
    }
}

} // namespace mlsdk::scenariorunner
