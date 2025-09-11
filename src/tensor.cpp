/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor.hpp"
#include "logging.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"

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

} // namespace

Tensor::Tensor(const TensorInfo &tensorInfo, std::shared_ptr<ResourceMemoryManager> memoryManager)
    : _debugName(tensorInfo.debugName), _shape(tensorInfo.shape), _dataType(tensorInfo.format),
      _memoryManager(std::move(memoryManager)), _tiling(convertTiling(tensorInfo.tiling)),
      _memoryOffset(tensorInfo.memoryOffset), _isAliasedWithImage(tensorInfo.isAliasedWithImage) {}

void Tensor::setup(const Context &ctx) {

    // implicitly convert rank=[] to rank=[1]
    if (_shape.empty()) {
        _shape.push_back(1);
        _rankConverted = true;
    }

    // Create tensor
    vk::TensorUsageFlagsARM usageFlags =
        vk::TensorUsageFlagBitsARM::eShader | vk::TensorUsageFlagBitsARM::eTransferSrc |
        vk::TensorUsageFlagBitsARM::eTransferDst | vk::TensorUsageFlagBitsARM::eDataGraph;

    uint32_t rank = static_cast<uint32_t>(_shape.size());

    if (_isAliasedWithImage && _tiling != vk::TensorTilingARM::eOptimal) {
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
            if (numComponentsFromVkFormat(_memoryManager->getFormat()) != _shape.back()) {
                throw std::runtime_error("Aliased tensor innermost dimension: " + std::to_string(_shape.back()) +
                                         ", must match number of components of image: " +
                                         std::to_string(numComponentsFromVkFormat(_memoryManager->getFormat())));
            }

            pushStride(elementSizeFromVkFormat(_dataType) * numComponentsFromVkFormat(_memoryManager->getFormat()));
            pushStride(elementSizeFromVkFormat(_dataType));
        }
    }

    int64_t *strides_ptr = _strides.data();
    if (_strides.empty()) {
        strides_ptr = nullptr;
    }

    vk::TensorDescriptionARM description(_tiling, _dataType, rank, _shape.data(), strides_ptr, usageFlags);

    vk::TensorCreateInfoARM createInfo(vk::TensorCreateFlagsARM(), &description, vk::SharingMode::eExclusive);
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

uint64_t Tensor::memSize() const { return _memoryManager->getMemSize(); }

vk::Format Tensor::dataType() const { return _dataType; }

const std::vector<int64_t> &Tensor::dimStrides() const { return _strides; }

const std::vector<int64_t> &Tensor::shape() const { return _shape; }

vk::TensorTilingARM Tensor::tiling() const { return _tiling; }

void *Tensor::map() const {
    if (!_memoryManager->isInitalized()) {
        throw std::runtime_error("Uninitialized MemoryManager for Tensor");
    }
    return _memoryManager->getDeviceMemory().mapMemory(_memoryOffset, _size);
}

void Tensor::unmap() const {
    if (!_memoryManager->isInitalized()) {
        throw std::runtime_error("Uninitialized MemoryManager for Tensor");
    }
    _memoryManager->getDeviceMemory().unmapMemory();
}

void Tensor::allocateMemory(const Context &ctx) {
    // Allocate memory
    if (!_memoryManager->isInitalized()) {
        _memoryManager->allocateDeviceMemory(ctx, vk::MemoryPropertyFlagBits::eHostVisible |
                                                      vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    // Bind tensor to memory
    const vk::BindTensorMemoryInfoARM bindInfo(*_tensor, *_memoryManager->getDeviceMemory(),
                                               _memoryManager->getSubresourceOffset() + _memoryOffset);
    ctx.device().bindTensorMemoryARM(vk::ArrayProxy<vk::BindTensorMemoryInfoARM>(bindInfo));

    // Create tensor view
    _tensorView = vk::raii::TensorViewARM(ctx.device(), {vk::TensorViewCreateFlagsARM(), *_tensor, _dataType});

    trySetVkRaiiObjectDebugName(ctx, _tensorView, _debugName + " view (default)");
}

void Tensor::fillFromDescription(const TensorDesc &desc) const {
    if (desc.src) {
        MemoryMap mapped(desc.src.value());
        auto dataPtr = vgfutils::numpy::parse(mapped);
        uint64_t elementSizeFromDesc = elementSizeFromVkFormat(getVkFormatFromString(desc.format));
        uint64_t expectedSize = elementSizeFromDesc * totalElementsFromShape(desc.dims);
        if (expectedSize != dataPtr.size()) {
            throw std::runtime_error("Tensor and data have different size mismatch");
        }
        fill(dataPtr.ptr, dataPtr.size());
    } else {
        fillZero();
    }
}

void Tensor::fill(const void *data, size_t size) const {
    if (size < memSize()) {
        mlsdk::logging::warning("Tensor data size " + std::to_string(size) +
                                " is different from allocated memory size " + std::to_string(memSize()));
    } else if (size > memSize()) {
        const std::string msg = "Allocated Tensor memory is less than data size: " + std::to_string(memSize()) +
                                " vs " + std::to_string(size);
        throw std::runtime_error(msg);
    }
    void *pDeviceMemory = map();
    std::memcpy(pDeviceMemory, data, size);
    unmap();
}

void Tensor::fillZero() const {
    void *pDeviceMemory = map();
    std::memset(pDeviceMemory, 0, memSize());
    unmap();
}

std::vector<char> Tensor::getTensorData() const {
    const ScopeExit<void()> onScopeExitRun([&]() { unmap(); });

    // 1) map the GPU memory
    const auto *mapped = static_cast<char *>(map());
    const auto dSize = dataSize();

    std::vector<char> out;
    // 2) If padded/tiled (memSize != dataSize) *and* a 4D tensor with strides,
    //    walk element-by-element using those strides to lay out the data correctly
    if (memSize() != dSize && _shape.size() == _strides.size() && _shape.size() == 4) {
        out.reserve(dSize);
        const int64_t elementSize{elementSizeFromVkFormat(dataType())};
        for (int64_t a = 0; a < _shape[0]; ++a) {
            for (int64_t b = 0; b < _shape[1]; ++b) {
                for (int64_t c = 0; c < _shape[2]; ++c) {
                    for (int64_t d = 0; d < _shape[3]; ++d) {
                        for (int64_t e = 0; e < elementSize; ++e) {
                            int64_t dataIdx = a * _strides[0] + b * _strides[1] + c * _strides[2] + d * _strides[3] + e;
                            out.push_back(mapped[dataIdx]);
                        }
                    }
                }
            }
        }
    } else {
        if (memSize() != dataSize()) {
            mlsdk::logging::warning("Tensor data size " + std::to_string(dataSize()) +
                                    " is different from allocated memory size " + std::to_string(memSize()));
        }

        // 3) Otherwise data is contiguous: bulk copy
        out.resize(dSize);
        std::memcpy(out.data(), mapped, dSize);
    }

    return out;
}

void Tensor::store(const std::string &filename) const {
    const auto data = getTensorData();
    const vgfutils::numpy::DataPtr dataPtr(data.data(), _rankConverted ? std::vector<int64_t>(0) : _shape,
                                           getDTypeFromVkFormat(dataType()));
    vgfutils::numpy::write(filename, dataPtr);
}

const std::string &Tensor::debugName() const { return _debugName; }
} // namespace mlsdk::scenariorunner
