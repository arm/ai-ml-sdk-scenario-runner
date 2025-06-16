/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor.hpp"
#include "logging.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"

namespace mlsdk::scenariorunner {

Tensor::Tensor(Context &ctx, const std::string &debugName, vk::Format dataType, const std::vector<int64_t> &shape,
               bool isAliased, vk::TensorTilingARM tiling, std::shared_ptr<ResourceMemoryManager> memoryManager,
               bool isConstant)
    : _debugName(debugName), _shape(shape), _dataType(dataType), _memoryManager(memoryManager), _tiling(tiling),
      _isConstant(isConstant) {

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

    if (isAliased && _tiling != vk::TensorTilingARM::eOptimal) {

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
        if (rank > 3 && memoryManager->getImageType() == vk::ImageType::e3D) {
            pushStride(memoryManager->getSubResourceDepthPitch());
        } else if (rank > 3 && memoryManager->getImageType() == vk::ImageType::e2D) {
            pushStride(_memoryManager->getMemSize());
        }
        // setting pStrides[dimensionCount-3]
        if (rank > 2) {
            pushStride(memoryManager->getSubResourceRowPitch());
        }
        // setting pStrides[dimensionCount-2] and pStrides[dimensionCount-1]
        if (rank > 1) {
            if (numComponentsFromVkFormat(memoryManager->getFormat()) != _shape.back()) {
                throw std::runtime_error("Aliased tensor innermost dimension: " + std::to_string(_shape.back()) +
                                         ", must match number of components of image: " +
                                         std::to_string(numComponentsFromVkFormat(memoryManager->getFormat())));
            }

            pushStride(elementSizeFromVkFormat(_dataType) * numComponentsFromVkFormat(memoryManager->getFormat()));
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

    trySetVkRaiiObjectDebugName(ctx, _tensor, debugName);

    vk::TensorMemoryRequirementsInfoARM memInfo(*_tensor);
    vk::MemoryRequirements2 memreqs = ctx.device().getTensorMemoryRequirementsARM(memInfo);

    _memoryManager->updateMemSize(memreqs.memoryRequirements.size + memoryManager->getSubresourceOffset());
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

void *Tensor::map() {
    if (!_memoryManager->isInitalized()) {
        throw std::runtime_error("Uninitialized MemoryManager for Tensor");
    }
    return _memoryManager->getDeviceMemory().mapMemory(0, memSize());
}

void Tensor::unmap() {
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
                                               _memoryManager->getSubresourceOffset());
    ctx.device().bindTensorMemoryARM(vk::ArrayProxy<vk::BindTensorMemoryInfoARM>(bindInfo));

    // Create tensor view
    _tensorView = vk::raii::TensorViewARM(ctx.device(), {vk::TensorViewCreateFlagsARM(), *_tensor, _dataType});

    trySetVkRaiiObjectDebugName(ctx, _tensorView, _debugName + " view (default)");
}

std::shared_ptr<ResourceMemoryManager> Tensor::getMemoryManager() { return _memoryManager; }

void Tensor::fillFromDescription(const TensorDesc &desc) {
    if (desc.aliasTarget.resourceRef.isValid()) {
        // Tensors dont overwrite shared memory
        throw std::runtime_error("Cannot fill aliased tensor with data");
    }

    if (desc.src) {
        MemoryMap mapped(desc.src.value());
        mlsdk::numpy::data_ptr dataPtr;
        mlsdk::numpy::parse(mapped, dataPtr);

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

void Tensor::fill(const void *data, size_t size) {
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

void Tensor::fillZero() {
    void *pDeviceMemory = map();
    std::memset(pDeviceMemory, 0, memSize());
    unmap();
}

void Tensor::store(Context &, const std::string &filename) {
    ScopeExit<void()> on_scope_exit_run([&] { unmap(); });

    const char *mapped = reinterpret_cast<const char *>(map());

    if (memSize() != dataSize() && _shape.size() == _strides.size() && _shape.size() == 4) {
        mlsdk::numpy::write(filename, {_shape.begin(), _shape.end()}, getDTypeFromVkFormat(dataType()),
                            [&](std::ostream &out) {
                                int64_t writtenBytes{0};
                                int64_t elementSize{elementSizeFromVkFormat(dataType())};
                                for (int64_t a = 0; a < _shape[0]; ++a) {
                                    for (int64_t b = 0; b < _shape[1]; ++b) {
                                        for (int64_t c = 0; c < _shape[2]; ++c) {
                                            for (int64_t d = 0; d < _shape[3]; ++d) {
                                                for (int64_t e = 0; e < elementSize; ++e) {
                                                    int64_t dataIdx = a * _strides[0] + b * _strides[1] +
                                                                      c * _strides[2] + d * _strides[3] + e;
                                                    out.put(mapped[dataIdx]);
                                                    writtenBytes++;
                                                }
                                            }
                                        }
                                    }
                                }
                                return writtenBytes;
                            });

        return;
    }

    if (memSize() != dataSize()) {
        mlsdk::logging::warning("Tensor data size " + std::to_string(dataSize()) +
                                " is different from allocated memory size " + std::to_string(memSize()));
    }

    mlsdk::numpy::data_ptr data(reinterpret_cast<const char *>(mapped),
                                _rankConverted ? std::vector<uint64_t>(0)
                                               : std::vector<uint64_t>{_shape.begin(), _shape.end()},
                                getDTypeFromVkFormat(dataType()));
    mlsdk::numpy::write(filename, data);
}

vk::TensorTilingARM Tensor::convertTiling(const Tiling tiling) {
    switch (tiling) {
    case Tiling::Linear:
        return vk::TensorTilingARM::eLinear;
    case Tiling::Optimal:
        return vk::TensorTilingARM::eOptimal;
    default:
        throw std::runtime_error("Unknown tiling");
    }
}

const std::string &Tensor::debugName() const { return _debugName; }
} // namespace mlsdk::scenariorunner
