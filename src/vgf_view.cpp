/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "vgf_view.hpp"

#include "data_manager.hpp"
#include "iresource.hpp"

#include <numeric>

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

uint32_t bufferSize(const vgflib::DataView<int64_t> &shape) {
    return static_cast<uint32_t>(
        std::abs(std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>())));
}

constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_UNKNOWN = 0;
constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6;
constexpr vgflib::DescriptorType DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000;

constexpr vk::DescriptorType getVkDescriptorType(vgflib::DescriptorType descriptorType) {
    switch (descriptorType) {
    case DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return vk::DescriptorType::eStorageBuffer;
    case DESCRIPTOR_TYPE_TENSOR_ARM:
        return vk::DescriptorType::eTensorARM;
    default:
        throw std::runtime_error("Descriptor type from VGF file not found");
    }
}

class DataManagerResourceViewerImpl final : public DataManagerResourceViewer {
  public:
    using DataManagerResourceViewer::DataManagerResourceViewer;

    bool hasImage() const override { return false; }

    const Image &getImage() const override { throw std::runtime_error("Image is an invalid resource type."); }
};

} // namespace

VgfView::VgfView(std::unique_ptr<MemoryMap> mapped, std::unique_ptr<vgflib::ModuleTableDecoder> moduleTableDecoder,
                 std::unique_ptr<vgflib::ModelSequenceTableDecoder> sequenceTableDecoder,
                 std::unique_ptr<vgflib::ModelResourceTableDecoder> resourceTableDecoder,
                 std::unique_ptr<vgflib::ConstantDecoder> constantTableDecoder)
    : mapped(std::move(mapped)), moduleTableDecoder(std::move(moduleTableDecoder)),
      sequenceTableDecoder(std::move(sequenceTableDecoder)), resourceTableDecoder(std::move(resourceTableDecoder)),
      constantTableDecoder(std::move(constantTableDecoder)) {}

VgfView VgfView::createVgfView(const std::string &vgfFile) {

    auto mapped = std::make_unique<MemoryMap>(vgfFile);
    auto headerDecoder = vgflib::CreateHeaderDecoder(mapped->ptr(), mapped->size());
    if (!headerDecoder) {
        throw std::runtime_error("Invalid VGF header");
    }

    const auto moduleTableOffset = headerDecoder->GetModuleTableOffset();
    const auto moduleTableSize = headerDecoder->GetModuleTableSize();

    const auto resourceTableOffset = headerDecoder->GetModelResourceTableOffset();
    const auto resourceTableSize = headerDecoder->GetModelResourceTableSize();

    const auto sequenceTableOffset = headerDecoder->GetModelSequenceTableOffset();
    const auto sequenceTableSize = headerDecoder->GetModelSequenceTableSize();

    const auto constantsOffset = headerDecoder->GetConstantsOffset();
    const auto constantsSize = headerDecoder->GetConstantsSize();

    auto moduleTableDecoder = vgflib::CreateModuleTableDecoder(mapped->ptr(moduleTableOffset), moduleTableSize);
    if (!moduleTableDecoder) {
        throw std::runtime_error("Invalid module table section");
    }
    auto sequenceTableDecoder =
        vgflib::CreateModelSequenceTableDecoder(mapped->ptr(sequenceTableOffset), sequenceTableSize);
    if (!sequenceTableDecoder) {
        throw std::runtime_error("Invalid model sequence table section");
    }
    auto resourceTableDecoder =
        vgflib::CreateModelResourceTableDecoder(mapped->ptr(resourceTableOffset), resourceTableSize);
    if (!resourceTableDecoder) {
        throw std::runtime_error("Invalid model resource table section");
    }
    auto constantTableDecoder = vgflib::CreateConstantDecoder(mapped->ptr(constantsOffset), constantsSize);
    if (!constantTableDecoder) {
        throw std::runtime_error("Invalid constant section");
    }

    VgfView vgfView(std::move(mapped), std::move(moduleTableDecoder), std::move(sequenceTableDecoder),
                    std::move(resourceTableDecoder), std::move(constantTableDecoder));
    return vgfView;
}

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
    if (mrtIndex == vgflib::CONSTANT_INVALID_MRT_INDEX) {
        throw std::runtime_error("Invalid constant metadata at index " + std::to_string(constantIndex));
    }
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
    if (mrtIndex == vgflib::CONSTANT_INVALID_MRT_INDEX) {
        throw std::runtime_error("Invalid constant metadata at index " + std::to_string(constantIndex));
    }
    if (resourceTableDecoder->getCategory(mrtIndex) != vgflib::ResourceCategory::CONSTANT) {
        throw std::runtime_error("Resource not marked as constant");
    }
    return resourceTableDecoder->getTensorShape(mrtIndex);
}

vgflib::DataView<uint8_t> VgfView::getConstantData(uint32_t constantIndex) const {
    return constantTableDecoder->getConstant(constantIndex);
}

std::pair<std::vector<TypedBinding>, VgfView::MrtIndexes> VgfView::getBindings(uint32_t segmentIndex) const {
    // Get segment binding infos
    std::vector<TypedBinding> bindings;
    auto descSetSize = sequenceTableDecoder->getSegmentDescriptorSetInfosSize(segmentIndex);
    MrtIndexes mrtIndexes;
    // For each segment descriptor set:
    for (uint32_t set = 0; set < descSetSize; ++set) {
        auto handle = sequenceTableDecoder->getDescriptorBindingSlotsHandle(segmentIndex, set);
        // For each descriptor set binding:
        for (uint32_t slot = 0; slot < sequenceTableDecoder->getBindingsSize(handle); ++slot) {
            auto mrtIndex = sequenceTableDecoder->getBindingSlotMrtIndex(handle, slot);
            const auto expectedType = resourceTableDecoder->getDescriptorType(mrtIndex);
            const auto vkDescriptorType = getVkDescriptorType(expectedType.value_or(DESCRIPTOR_TYPE_UNKNOWN));
            auto bindingId = sequenceTableDecoder->getBindingSlotBinding(handle, slot);
            auto guidStr = createResourceGuidStr(bindingId, resourceTableDecoder->getCategory(mrtIndex));
            TypedBinding binding;
            binding.set = set;
            binding.id = bindingId;
            binding.resourceRef = guidStr;
            binding.vkDescriptorType = vkDescriptorType;
            bindings.emplace_back(binding);
            mrtIndexes.insert({{set, bindingId}, mrtIndex});
        }
    }

    return {std::move(bindings), std::move(mrtIndexes)};
}

std::vector<TypedBinding> VgfView::resolveBindings(uint32_t segmentIndex, const DataManager &dataManager,
                                                   const std::vector<TypedBinding> &externalBindings) const {

    auto [bindings, mrtIndexes] = getBindings(segmentIndex);

    for (const auto &externalBinding : externalBindings) {
        if (!(dataManager.hasTensor(externalBinding.resourceRef) ||
              dataManager.hasBuffer(externalBinding.resourceRef))) {
            // All tensors should have been created when this function is called
            throw std::runtime_error("No resource with this guid found");
        }

        const DataManagerResourceViewerImpl resourceViewer(dataManager, externalBinding.resourceRef);
        for (auto &binding : bindings) {
            if (binding.set == externalBinding.set && binding.id == externalBinding.id) {
                binding.resourceRef = externalBinding.resourceRef;

                auto mrtIndexSearch = mrtIndexes.find({externalBinding.set, externalBinding.id});
                if (mrtIndexSearch == mrtIndexes.end()) {
                    throw std::runtime_error("No resource found in MRT Table");
                }
                validateResource(resourceViewer, mrtIndexSearch->second);
            }
        }
    }
    return bindings;
}

void VgfView::validateResource(const IResourceViewer &resourceViewer, uint32_t vgfMrtIndex) const {
    std::optional<vgflib::DescriptorType> expectedType = resourceTableDecoder->getDescriptorType(vgfMrtIndex);
    if (!expectedType.has_value()) {
        throw std::runtime_error("Descriptor type not found from VGF file");
    }

    switch (expectedType.value()) {
    case DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
        const auto &buffer = resourceViewer.getBuffer();

        // Check if buffer sizes match
        auto shape = resourceTableDecoder->getTensorShape(vgfMrtIndex);
        auto expectedBufferSize = bufferSize(shape);
        if (buffer.size() != expectedBufferSize) {
            throw std::runtime_error("Mismatch of buffer size declarations between JSON and VGF file");
        }
    } break;
    case DESCRIPTOR_TYPE_TENSOR_ARM: {
        const auto &tensor = resourceViewer.getTensor();
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
            throw std::runtime_error("Mismatch of tensor data type declarations "
                                     "between JSON and VGF file");
        }
    } break;
    default:
        throw std::runtime_error(
            "No resource validation should be performed for resources different from tensors and buffers");
        break;
    }
}

void VgfView::createIntermediateResources(IResourceCreator &creator) const {
    // Iterate over all VGF Resources, create intermediates
    size_t numResources = resourceTableDecoder->size();
    for (uint32_t resourceIndex = 0; resourceIndex < numResources; ++resourceIndex) {
        auto resourceCategory = resourceTableDecoder->getCategory(resourceIndex);
        if (resourceCategory == vgflib::ResourceCategory::INTERMEDIATE) {
            auto guidStr = createResourceGuidStr(resourceIndex, resourceCategory);
            auto type = resourceTableDecoder->getDescriptorType(resourceIndex);
            switch (type.value_or(DESCRIPTOR_TYPE_UNKNOWN)) {
            case (DESCRIPTOR_TYPE_UNIFORM_BUFFER): {
                auto shape = resourceTableDecoder->getTensorShape(resourceIndex);
                auto expectedBufferSize = bufferSize(shape);

                // Create Scenario Runner buffer resource
                BufferInfo info{guidStr, expectedBufferSize};
                creator.createBuffer(guidStr, info);
            } break;
            case (DESCRIPTOR_TYPE_TENSOR_ARM): {
                auto shape = resourceTableDecoder->getTensorShape(resourceIndex);
                auto format = resourceTableDecoder->getVkFormat(resourceIndex);

                auto info = TensorInfo{guidStr, std::vector<int64_t>(shape.begin(), shape.end()), vk::Format(format),
                                       -1, false};

                // Create Scenario Runner tensor
                creator.createTensor(guidStr, info);
            } break;
            default:
                throw std::runtime_error("Unknown resource type read from VGF file");
            }
        }
    }
}

} // namespace mlsdk::scenariorunner
