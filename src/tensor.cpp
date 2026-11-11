/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor.hpp"
#include "logging.hpp"
#include "resource_desc.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"

#include "vgf-utils/memory_map.hpp"

#include <cstring>
#include <utility>

namespace mlsdk::scenariorunner {
namespace {
constexpr vk::TensorTilingARM convertTiling(const Tiling tiling) {
    switch (tiling) {
    case Tiling::Linear:
        return vk::TensorTilingARM::eLinear;
    case Tiling::Optimal:
        return vk::TensorTilingARM::eOptimal;
    default:
        throw std::runtime_error("Unknown tiling");
    }
}

std::vector<int64_t> tensorScaleShape(const std::vector<int64_t> &shape, const BlockScalingInfo &scaling) {
    auto result = shape;
    if (scaling.blockShape.size() > shape.size()) {
        throw std::runtime_error("Block rank exceeds tensor rank");
    }
    const auto offset = shape.size() - scaling.blockShape.size();
    for (size_t i = 0; i < scaling.blockShape.size(); ++i) {
        const auto block = scaling.blockShape[i];
        if (block == 0 || result[offset + i] % block != 0) {
            throw std::runtime_error("Tensor dimensions must be divisible by positive block dimensions");
        }
        result[offset + i] /= block;
    }
    return result;
}

TensorData readTensorFile(const std::filesystem::path &path, vk::Format format) {
    const MemoryMap mapped(path.string());
    if (mapped.size() < 10) {
        throw std::runtime_error("Truncated tensor NumPy header");
    }
    const auto *bytes = static_cast<const unsigned char *>(mapped.ptr());
    const size_t prefixSize = bytes[6] == 1 ? 10 : 12;
    if (mapped.size() < prefixSize) {
        throw std::runtime_error("Truncated tensor NumPy header");
    }
    uint32_t headerLength = 0;
    if (bytes[6] == 1) {
        uint16_t length = 0;
        std::memcpy(&length, bytes + 8, sizeof(length));
        headerLength = length;
    } else {
        std::memcpy(&headerLength, bytes + 8, sizeof(headerLength));
    }
    if (headerLength > mapped.size() - prefixSize) {
        throw std::runtime_error("Truncated tensor NumPy header");
    }
    const auto parsed = vgfutils::numpy::parse(mapped);
    const auto expectedSize = elementSizeFromVkFormat(format) * totalElementsFromShape(parsed.shape);
    if (mapped.size() - prefixSize - headerLength != expectedSize || parsed.size() != expectedSize) {
        throw std::runtime_error("Tensor NumPy payload length does not match the tensor");
    }
    TensorData tensorData;
    tensorData.shape = parsed.shape;
    tensorData.data.resize(parsed.size());
    std::memcpy(tensorData.data.data(), parsed.ptr, tensorData.data.size());
    return tensorData;
}

} // namespace

Tensor::Tensor(TensorInfo tensorInfo)
    : _debugName(std::move(tensorInfo.debugName)), _shape(std::move(tensorInfo.shape)), _dataType(tensorInfo.format),
      _tiling(convertTiling(tensorInfo.tiling)), _memoryOffset(tensorInfo.memoryOffset),
      _descriptorBufferCaptureReplay(tensorInfo.descriptorBufferCaptureReplay) {
    rejectUnsupportedMXFPFormats(_dataType);
}

void Tensor::setup(const Context &ctx, std::shared_ptr<ResourceMemoryManager> memoryManager) {
    _memoryManager = std::move(memoryManager);

    // implicitly convert rank=[] to rank=[1]
    if (_shape.empty()) {
        _shape.push_back(1);
        _rankConverted = true;
    }

    // Create tensor
    vk::TensorUsageFlagsARM usageFlags = vk::TensorUsageFlagBitsARM::eShader |
                                         vk::TensorUsageFlagBitsARM::eTransferSrc |
                                         vk::TensorUsageFlagBitsARM::eTransferDst;
    if (ctx._optionals.data_graph) {
        usageFlags |= vk::TensorUsageFlagBitsARM::eDataGraph;
    }

    auto rank = static_cast<uint32_t>(_shape.size());

    if (_memoryManager->hasImageMetadata() && _tiling != vk::TensorTilingARM::eOptimal) {
        /*
          The extension to the spec does not support rank 4 tensors aliasing 2D images. Rank 4 tensor is associated with
          a 3D image. The image type check was added to avoid faults for 2D images due to this spec requirement:
              VkTensorDescriptionARM::pStrides[dimensionCount-4] must be equal to VkSubresourceLayout::depthPitch
              if VkTensorDescriptionARM::dimensionCount is greater than 3.

          For the 2D image dim0 stride is set as an image allocated memory size. Specifically not as a data size as
          it needs to account for a row pitch.
        */
        auto pushStride = [&](uint64_t stride) {
            if (stride > static_cast<uint64_t>(INT64_MAX)) {
                throw std::runtime_error("Value out of range for stride: " + std::to_string(stride));
            }
            _strides.push_back(static_cast<int64_t>(stride));
        };

        // setting pStrides[dimensionCount-4]
        if (rank > 3 && _memoryManager->getImageType() == vk::ImageType::e3D) {
            pushStride(_memoryManager->getSubResourceDepthPitch());
        } else if (rank > 3 && _memoryManager->getImageType() == vk::ImageType::e2D) {
            pushStride(_memoryManager->getMemSize());
        }
        // setting pStrides[dimensionCount-3]
        if (rank > 2) {
            pushStride(_memoryManager->getSubResourceRowPitch());
        }
        // setting pStrides[dimensionCount-2] and pStrides[dimensionCount-1]
        if (rank > 1) {
            const auto numComponents = numComponentsFromVkFormat(_memoryManager->getFormat());
            if (numComponents != _shape.back()) {
                throw std::runtime_error(
                    "Aliased tensor innermost dimension: " + std::to_string(_shape.back()) +
                    ", must match number of components of image: " + std::to_string(numComponents));
            }

            const uint64_t elementSize{elementSizeFromVkFormat(_dataType)};
            pushStride(elementSize * numComponents);
            pushStride(elementSize);
        }
    }

    int64_t *stridesPtr = _strides.data();
    if (_strides.empty()) {
        stridesPtr = nullptr;
    }

    vk::TensorDescriptionARM description(_tiling, _dataType, rank, _shape.data(), stridesPtr, usageFlags);

    vk::TensorCreateFlagsARM flags{};
    if (_descriptorBufferCaptureReplay) {
        flags |= vk::TensorCreateFlagBitsARM::eDescriptorBufferCaptureReplay;
    }

    vk::TensorCreateInfoARM createInfo(flags, &description, vk::SharingMode::eExclusive);
    _tensor = vk::raii::TensorARM(ctx.device(), createInfo);

    trySetVkRaiiObjectDebugName(ctx, _tensor, _debugName);

    vk::TensorMemoryRequirementsInfoARM memInfo(*_tensor);
    vk::MemoryRequirements2 memreqs = ctx.device().getTensorMemoryRequirementsARM(memInfo);

    _size = memreqs.memoryRequirements.size;
    _memoryManager->updateMemSize(memreqs.memoryRequirements.size + _memoryManager->getSubresourceOffset() +
                                  _memoryOffset);
    _memoryManager->updateMemType(memreqs.memoryRequirements.memoryTypeBits);
}

const vk::TensorARM &Tensor::tensor() const { return *_tensor; }

const vk::TensorViewARM &Tensor::tensorView() const { return *_tensorView; }

uint64_t Tensor::dataSize() const { return elementSizeFromVkFormat(_dataType) * totalElementsFromShape(_shape); }

vk::Format Tensor::dataType() const { return _dataType; }

const std::vector<int64_t> &Tensor::dimStrides() const { return _strides; }

const std::vector<int64_t> &Tensor::shape() const { return _shape; }

vk::TensorTilingARM Tensor::tiling() const { return _tiling; }

void Tensor::allocateMemory(const Context &ctx) {
    // Allocate memory
    if (!_memoryManager->isInitalized()) {
        _memoryManager->allocateDeviceMemory(ctx, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    // Bind tensor to memory
    const vk::BindTensorMemoryInfoARM bindInfo(*_tensor, *_memoryManager->getDeviceMemory(),
                                               _memoryManager->getSubresourceOffset() + _memoryOffset);
    ctx.device().bindTensorMemoryARM(vk::ArrayProxy<vk::BindTensorMemoryInfoARM>(bindInfo));

    // Create tensor view
    _tensorView = vk::raii::TensorViewARM(ctx.device(), {vk::TensorViewCreateFlagsARM(), *_tensor, _dataType});

    trySetVkRaiiObjectDebugName(ctx, _tensorView, _debugName + " view (default)");
}

void Tensor::fill(const void *data, size_t size) const {
    if (size < _size) {
        mlsdk::logging::warning("Tensor data size " + std::to_string(size) +
                                " is different from allocated memory size " + std::to_string(_size));
    } else if (size > _size) {
        throw std::runtime_error("Allocated Tensor memory is less than data size: " + std::to_string(_size) + " vs " +
                                 std::to_string(size));
    }
    const auto stagingOffset = _memoryManager->getSubresourceOffset() + _memoryOffset;
    void *pStagingMemory = _memoryManager->mapStagingBufferMemory(stagingOffset, _size);
    std::memcpy(pStagingMemory, data, size);
    _memoryManager->unmapStagingBufferMemory();
}

void Tensor::fillZero() const {
    const auto stagingOffset = _memoryManager->getSubresourceOffset() + _memoryOffset;
    void *pStagingMemory = _memoryManager->mapStagingBufferMemory(stagingOffset, _size);
    std::memset(pStagingMemory, 0, _size);
    _memoryManager->unmapStagingBufferMemory();
}

void Tensor::upload(const Context &ctx, const TensorDataView &data) const {
    // Validate shape; allow 0-d input when rank was converted to [1]
    bool shapesMatch = (data.shape == _shape) || (_rankConverted && data.shape.empty());
    if (!shapesMatch) {
        throw std::runtime_error("Tensor::upload: provided shape does not match tensor shape");
    }

    if (data.format.has_value() && data.format.value() != _dataType) {
        throw std::runtime_error("Tensor::upload: provided format does not match tensor format");
    }

    const uint64_t expectedSize = elementSizeFromVkFormat(_dataType) * totalElementsFromShape(_shape);
    if (data.size != expectedSize) {
        throw std::runtime_error("Tensor::upload: size does not match logical data size");
    }

    fill(data.data, data.size);
    _memoryManager->uploadData(ctx, _memoryManager->getSubresourceOffset() + _memoryOffset, _size);
}

TensorData Tensor::download(const Context &ctx) const {
    TensorData tensorData;
    tensorData.data = getTensorData(ctx);
    if (!_rankConverted) {
        tensorData.shape = _shape;
    }
    tensorData.format = _dataType;
    return tensorData;
}

std::vector<std::byte> Tensor::getTensorData(const Context &ctx) const {
    _memoryManager->downloadData(ctx, _memoryManager->getSubresourceOffset() + _memoryOffset, _size);

    const ScopeExit<void()> onScopeExitRun([&]() { _memoryManager->unmapStagingBufferMemory(); });

    // 1) map the GPU memory
    const auto *mapped = static_cast<std::byte *>(
        _memoryManager->mapStagingBufferMemory(_memoryManager->getSubresourceOffset() + _memoryOffset, _size));
    const auto dSize = dataSize();

    std::vector<std::byte> out;
    // 2) If padded/tiled (_size != dataSize) *and* a 4D tensor with strides,
    //    walk element-by-element using those strides to lay out the data correctly
    if (_size != dSize && _shape.size() == _strides.size() && _shape.size() == 4) {
        out.reserve(dSize);
        const int64_t elementSize{elementSizeFromVkFormat(dataType())};
        for (int64_t a = 0; a < _shape[0]; ++a) {
            for (int64_t b = 0; b < _shape[1]; ++b) {
                for (int64_t c = 0; c < _shape[2]; ++c) {
                    for (int64_t d = 0; d < _shape[3]; ++d) {
                        for (int64_t e = 0; e < elementSize; ++e) {
                            int64_t dataIdx =
                                (a * _strides[0]) + (b * _strides[1]) + (c * _strides[2]) + (d * _strides[3]) + e;
                            out.push_back(mapped[dataIdx]);
                        }
                    }
                }
            }
        }
    } else {
        if (_size != dSize) {
            mlsdk::logging::warning("Tensor data size " + std::to_string(dSize) +
                                    " is different from allocated memory size " + std::to_string(_size));
        }

        // 3) Otherwise data is contiguous: bulk copy
        out.resize(dSize);
        std::memcpy(out.data(), mapped, dSize);
    }

    return out;
}

void Tensor::store(const Context &ctx, const std::string &filename) const {
    const auto tensorData = download(ctx);
    writeTensorFiles(tensorData, filename);
}

const std::string &Tensor::debugName() const { return _debugName; }

TensorData loadTensorData(const TensorDesc &desc) {
    const auto format = getVkFormatFromString(desc.format);
    const std::optional<std::filesystem::path> values{desc.src};
    const std::optional<std::filesystem::path> scales{desc.blockScaling ? desc.blockScaling->src : std::nullopt};
    const auto scaling = desc.blockScaling ? std::optional{desc.blockScaling->info} : std::nullopt;
    if (scales.has_value() != (scaling.has_value() && values.has_value())) {
        throw std::runtime_error("Block-scaled tensors require both value and scale input paths");
    }
    if (values && (values->empty() ||
                   (scales && (scales->empty() || values->lexically_normal() == scales->lexically_normal())))) {
        throw std::runtime_error("Tensor input paths must be nonempty and distinct");
    }
    if (scaling && scaling->format != E8M0_FORMAT) {
        throw std::runtime_error("Tensor scale format must be E8M0");
    }
    TensorData result;
    if (values) {
        result = readTensorFile(*values, format);
        if ((!desc.dims.empty() || scaling) && result.shape != desc.dims) {
            throw std::runtime_error("Tensor value NumPy shape does not match the tensor");
        }
    } else {
        result.shape = desc.dims;
        const auto expectedSize = elementSizeFromVkFormat(format) * totalElementsFromShape(desc.dims);
        result.data.resize(expectedSize, std::byte{0});
    }
    result.format = format;
    result.blockScaling = scaling;
    if (scaling) {
        if (format != vk::Format::eR8SfloatFpencodingFloat8E4M3ARM &&
            format != vk::Format::eR8SfloatFpencodingFloat8E5M2ARM) {
            throw std::runtime_error("Block-scaled file I/O supports FP8 values");
        }
        const auto scaleShape = tensorScaleShape(desc.dims, *scaling);
        const auto scaleSize = elementSizeFromVkFormat(scaling->format) * totalElementsFromShape(scaleShape);
        if (scales) {
            auto scaleData = readTensorFile(*scales, scaling->format);
            if (scaleData.shape != scaleShape) {
                throw std::runtime_error("Tensor scale NumPy shape does not match its block shape");
            }
            result.scales = std::move(scaleData.data);
        } else {
            result.scales.resize(scaleSize, std::byte{0});
        }
    }
    return result;
}

void writeTensorFiles(const TensorData &data, const std::filesystem::path &values,
                      const std::optional<std::filesystem::path> &scales) {
    if (data.blockScaling.has_value() != scales.has_value()) {
        throw std::runtime_error("Block-scaled tensors require both value and scale output paths");
    }
    if (values.empty() || (scales && (scales->empty() || values.lexically_normal() == scales->lexically_normal()))) {
        throw std::runtime_error("Tensor output paths must be nonempty and distinct");
    }
    const auto dtype = getDTypeFromVkFormat(data.format.value());
    const vgfutils::numpy::DataPtr valuePayload{reinterpret_cast<const char *>(data.data.data()), data.shape, dtype};
    if (valuePayload.size() != data.data.size()) {
        throw std::runtime_error("Tensor value payload length does not match its shape and format");
    }
    std::optional<vgfutils::numpy::DataPtr> scalePayload;
    if (data.blockScaling) {
        if (data.blockScaling->format != E8M0_FORMAT) {
            throw std::runtime_error("Tensor scale format must be E8M0");
        }
        if (data.format != vk::Format::eR8SfloatFpencodingFloat8E4M3ARM &&
            data.format != vk::Format::eR8SfloatFpencodingFloat8E5M2ARM) {
            throw std::runtime_error("Block-scaled file I/O supports FP8 values");
        }
        scalePayload.emplace(reinterpret_cast<const char *>(data.scales.data()),
                             tensorScaleShape(data.shape, *data.blockScaling),
                             getDTypeFromVkFormat(data.blockScaling->format));
        if (scalePayload->size() != data.scales.size()) {
            throw std::runtime_error("Tensor scale payload length does not match its block shape");
        }
    }
    vgfutils::numpy::write(values.string(), valuePayload);
    if (scalePayload) {
        vgfutils::numpy::write(scales->string(), *scalePayload);
    }
}

} // namespace mlsdk::scenariorunner
