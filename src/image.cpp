/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dds_reader.hpp"
#include "logging.hpp"
#include "resource_desc.hpp"
#include "utils.hpp"
#include "vulkan_debug_utils.hpp"
#include "vulkan_memory_manager.hpp"

#include <cmath>

namespace mlsdk::scenariorunner {
namespace {
constexpr vk::Filter convertFilter(const FilterMode filter) {
    switch (filter) {
    case FilterMode::Linear:
        return vk::Filter::eLinear;
    case FilterMode::Nearest:
        return vk::Filter::eNearest;
    default:
        throw std::runtime_error("Unknown filter mode");
    }
}

constexpr vk::SamplerMipmapMode convertSamplerMipmapMode(const FilterMode mode) {
    switch (mode) {
    case FilterMode::Linear:
        return vk::SamplerMipmapMode::eLinear;
    case FilterMode::Nearest:
        return vk::SamplerMipmapMode::eNearest;
    default:
        throw std::runtime_error("Unknown sampler mipmap mode");
    }
}

constexpr vk::SamplerAddressMode convertSamplerAddressMode(const AddressMode mode) {
    switch (mode) {
    case AddressMode::ClampBorder:
        return vk::SamplerAddressMode::eClampToBorder;
    case AddressMode::ClampEdge:
        return vk::SamplerAddressMode::eClampToEdge;
    case AddressMode::Repeat:
        return vk::SamplerAddressMode::eRepeat;
    case AddressMode::MirroredRepeat:
        return vk::SamplerAddressMode::eMirroredRepeat;
    default:
        throw std::runtime_error("Unknown sampler address mode");
    }
}

constexpr vk::BorderColor convertBorderColor(const BorderColor color) {
    switch (color) {
    case BorderColor::FloatTransparentBlack:
        return vk::BorderColor::eFloatTransparentBlack;
    case BorderColor::FloatOpaqueBlack:
        return vk::BorderColor::eFloatOpaqueBlack;
    case BorderColor::FloatOpaqueWhite:
        return vk::BorderColor::eFloatOpaqueWhite;
    case BorderColor::IntTransparentBlack:
        return vk::BorderColor::eIntTransparentBlack;
    case BorderColor::IntOpaqueBlack:
        return vk::BorderColor::eIntOpaqueBlack;
    case BorderColor::IntOpaqueWhite:
        return vk::BorderColor::eIntOpaqueWhite;
    case BorderColor::FloatCustomEXT:
        return vk::BorderColor::eFloatCustomEXT;
    case BorderColor::IntCustomEXT:
        return vk::BorderColor::eIntCustomEXT;
    default:
        throw std::runtime_error("Invalid border color");
    }
}

constexpr vk::ImageTiling convertTiling(const Tiling tiling) {
    switch (tiling) {
    case Tiling::Linear:
        return vk::ImageTiling::eLinear;
    case Tiling::Optimal:
        return vk::ImageTiling::eOptimal;
    default:
        throw std::runtime_error("Unknown tiling");
    }
}

} // namespace

Image::Image(const ImageInfo &imageInfo, std::shared_ptr<ResourceMemoryManager> memoryManager)
    : _imageInfo(imageInfo), _memoryManager(std::move(memoryManager)) {}

void Image::setup(const Context &ctx) {
    // Create image

    if (_imageInfo.mips == 0) {
        throw std::runtime_error("Number of mips cannot be 0");
    }
    if (_imageInfo.mips >
        static_cast<uint32_t>(std::floor(std::log2(
            std::max(static_cast<uint32_t>(_imageInfo.shape[1]), static_cast<uint32_t>(_imageInfo.shape[2]))))) +
            1) {
        throw std::runtime_error("Number of mips exceeds maximum number allowed for the image size");
    }
    if (_imageInfo.isAliased && _imageInfo.mips > 1) {
        throw std::runtime_error("A mipped image cannot be aliased");
    }

    const vk::Extent3D extent(static_cast<uint32_t>(_imageInfo.shape[1]), static_cast<uint32_t>(_imageInfo.shape[2]),
                              static_cast<uint32_t>(_imageInfo.shape[3]));
    vk::ImageUsageFlags usageFlags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

    vk::FormatFeatureFlags requiredFormatFlags;
    if (_imageInfo.isInput) {
        requiredFormatFlags |= vk::FormatFeatureFlagBits::eTransferDst;
    }
    if (_imageInfo.isSampled) {
        usageFlags |= vk::ImageUsageFlagBits::eSampled;
        requiredFormatFlags |= vk::FormatFeatureFlagBits::eSampledImage;
    }
    if (_imageInfo.isStorage) {
        usageFlags |= vk::ImageUsageFlagBits::eStorage;
        requiredFormatFlags |= vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eTransferSrc;
    }
    if (_imageInfo.mips > 1) {
        requiredFormatFlags |= vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst;
    }

    // Force D32S8 to D32 because of limited tiling support for stencil formats.
    _dataType = (_imageInfo.format == vk::Format::eD32SfloatS8Uint) ? vk::Format::eD32Sfloat : _imageInfo.format;

    auto featProps = ctx.physicalDevice().getFormatProperties(_dataType);

    if (_imageInfo.tiling.has_value()) {
        // Set tiling based on JSON file if set and then validate
        _tiling = convertTiling(_imageInfo.tiling.value());
        if (_tiling == vk::ImageTiling::eLinear &&
            ((featProps.linearTilingFeatures & requiredFormatFlags) != requiredFormatFlags)) {
            throw std::runtime_error("Tiling type: LINEAR is not supported for this format type");
        }
        if (_tiling == vk::ImageTiling::eOptimal) {
            if ((featProps.optimalTilingFeatures & requiredFormatFlags) != requiredFormatFlags) {
                throw std::runtime_error("Tiling type: OPTIMAL is not supported for this formatType");
            } else if (_imageInfo.isAliased) {
                mlsdk::logging::info("Allowing OPTIMAL tiling with aliasing for image");
            }
        }
    } else if ((featProps.linearTilingFeatures & requiredFormatFlags) == requiredFormatFlags &&
               _imageInfo.mips <= getFormatMaxMipLevels(ctx, vk::ImageTiling::eLinear, usageFlags)) {
        _tiling = vk::ImageTiling::eLinear;
    } else if ((featProps.optimalTilingFeatures & requiredFormatFlags) == requiredFormatFlags) {
        _tiling = vk::ImageTiling::eOptimal;
    } else {
        throw std::runtime_error("No supported tiling for this data type");
    }

    if (_imageInfo.mips > getFormatMaxMipLevels(ctx, _tiling, usageFlags)) {
        throw std::runtime_error("The mip level provided is not supported for " + _imageInfo.debugName);
    }

    _initialLayout = vk::ImageLayout::eUndefined;

    if (_imageInfo.isAliased && _tiling != vk::ImageTiling::eLinear) {
        usageFlags |= vk::ImageUsageFlagBits::eTensorAliasingARM;
    }

    const vk::ImageCreateInfo imageCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, _dataType, extent,
                                              /*mipLevels=*/_imageInfo.mips,
                                              /*arrayLayers=*/1, vk::SampleCountFlagBits::e1, _tiling, usageFlags,
                                              vk::SharingMode::eExclusive, /*queueFamilyIndices=*/{}, _initialLayout);
    _image = vk::raii::Image(ctx.device(), imageCreateInfo);

    trySetVkRaiiObjectDebugName(ctx, _image, _imageInfo.debugName);

    // Create image sampler
    const auto borderAddressMode = convertSamplerAddressMode(_imageInfo.samplerSettings.borderAddressMode);
    const auto mipFilter = convertFilter(_imageInfo.samplerSettings.minFilter);
    const auto magFilter = convertFilter(_imageInfo.samplerSettings.magFilter);
    const auto mipMapMode = convertSamplerMipmapMode(_imageInfo.samplerSettings.mipFilter);
    const auto borderColor = convertBorderColor(_imageInfo.samplerSettings.borderColor);
    vk::SamplerCreateInfo samplerCreateInfo(vk::SamplerCreateFlags(), mipFilter, magFilter, mipMapMode,
                                            borderAddressMode, borderAddressMode, borderAddressMode,
                                            /*mipLodBias=*/0.0f, /*anisotropyEnable=*/false, /*maxAnisotropy=*/1.0f,
                                            /*compareEnable=*/false, vk::CompareOp::eNever, /*minLod=*/0.0f,
                                            /*maxLod=*/static_cast<float>(_imageInfo.mips - 1), borderColor);

    vk::SamplerCustomBorderColorCreateInfoEXT customBorderColorCreateInfo;
    if ((borderColor == vk::BorderColor::eFloatCustomEXT) || (borderColor == vk::BorderColor::eIntCustomEXT)) {

        if (!ctx._optionals.custom_border_color) {
            throw std::runtime_error("Error sampler custom border color extension is unsupported "
                                     "on this device/driver");
        }

        vk::ClearColorValue customClearColorValue =
            borderColor == vk::BorderColor::eFloatCustomEXT
                ? vk::ClearColorValue(std::get<std::array<float, 4>>(_imageInfo.samplerSettings.customBorderColor))
                : vk::ClearColorValue(std::get<std::array<int, 4>>(_imageInfo.samplerSettings.customBorderColor));

        customBorderColorCreateInfo.setCustomBorderColor(customClearColorValue);
        customBorderColorCreateInfo.setFormat(_dataType);
        samplerCreateInfo.setPNext(&customBorderColorCreateInfo);
    }
    _sampler = vk::raii::Sampler(ctx.device(), samplerCreateInfo);

    vk::MemoryRequirements memoryRequirements = _image.getMemoryRequirements();
    _memoryManager->updateMemSize(memoryRequirements.size + _imageInfo.memoryOffset);
    _memoryManager->updateMemType(memoryRequirements.memoryTypeBits);

    vk::ImageSubresource targetSubresource(getImageAspectMaskForVkFormat(_dataType));
    if (_imageInfo.mips == 1 && _tiling == vk::ImageTiling::eLinear) {
        vk::SubresourceLayout targetSubresourceLayout = _image.getSubresourceLayout(targetSubresource);

        _memoryManager->updateSubResourceOffset(targetSubresourceLayout.offset);
        _memoryManager->updateSubResourceRowPitch(targetSubresourceLayout.rowPitch);
        _memoryManager->updateSubResourceDepthPitch(targetSubresourceLayout.depthPitch);
        _memoryManager->updateSubResourceArrayPitch(targetSubresourceLayout.arrayPitch);
    }

    _memoryManager->updateFormat(_dataType);
    _memoryManager->updateImageType(vk::ImageType::e2D);

    // Create the staging buffer
    vk::BufferCreateInfo bufferCreateInfo{vk::BufferCreateFlags(),
                                          dataSize(),
                                          vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                          vk::SharingMode::eExclusive,
                                          ctx.familyQueueIdx(),
                                          nullptr};

    _stagingBuffer = vk::raii::Buffer(ctx.device(), bufferCreateInfo);
    const vk::MemoryRequirements memReqs = _stagingBuffer.getMemoryRequirements();
    const auto flags = vk::MemoryPropertyFlagBits::eHostVisible;
    const uint32_t memTypeIndex = findMemoryIdx(ctx, memReqs.memoryTypeBits, flags);
    if (memTypeIndex == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Cannot find a memory type with the required properties");
    }

    const vk::MemoryAllocateInfo memAllocInfo(memReqs.size, memTypeIndex);
    _stagingBufferDeviceMemory = vk::raii::DeviceMemory(ctx.device(), memAllocInfo);
    _stagingBuffer.bindMemory(*_stagingBufferDeviceMemory, 0);
}

uint32_t Image::getFormatMaxMipLevels(const Context &ctx, vk::ImageTiling tiling, vk::ImageUsageFlags usageFlags) {
    try {
        vk::ImageFormatProperties formatProps;
        formatProps =
            ctx.physicalDevice().getImageFormatProperties(_dataType, vk::ImageType::e2D, tiling, usageFlags, {});

        return formatProps.maxMipLevels;
    } catch (const vk::FormatNotSupportedError &) {
        // Format is not supported, return 0 to disallow image creation
        return 0;
    }
}

vk::Image Image::image() const { return *_image; }

vk::ImageView Image::imageView() const { return *_imageView; }

vk::ImageView Image::imageView(uint32_t lod) const {
    if (lod > _imageInfo.mips - 1) {
        std::stringstream errorMessage;
        errorMessage << "Requested level of details for the Image is greater than configured mipmaps. ";
        errorMessage << "MipMaps configured: " << _imageInfo.mips << ", lod index requested: " << lod;
        throw std::runtime_error(errorMessage.str());
    }

    return *_imageViewMips[lod];
}

vk::Sampler Image::sampler() const { return *_sampler; }

uint64_t Image::memSize() const { return _memoryManager->getMemSize(); }

uint64_t Image::dataSize() const {
    uint64_t size = elementSizeFromVkFormat(_dataType) * totalElementsFromShape(_imageInfo.shape);
    return size;
}

vk::Format Image::dataType() const { return _dataType; }

const std::vector<int64_t> &Image::shape() const { return _imageInfo.shape; }

vk::ImageTiling Image::tiling() const { return _tiling; }

void Image::addTransitionLayoutCommand(vk::raii::CommandBuffer &cmdBuf, vk::ImageLayout expectedLayout) {
    if (_targetLayout == expectedLayout)
        return; // No transition needed

    vk::PipelineStageFlagBits2 srcStage = vk::PipelineStageFlagBits2::eComputeShader;
    vk::AccessFlagBits2 srcAccess = vk::AccessFlagBits2::eShaderWrite;
    vk::PipelineStageFlags2 dstStage =
        vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
    vk::AccessFlagBits2 dstAccess = vk::AccessFlagBits2::eShaderRead;

    vk::MemoryBarrier2 memoryBarrier{srcStage, srcAccess, dstStage, dstAccess};
    vk::ImageMemoryBarrier2 imageBarrier{srcStage,
                                         srcAccess,
                                         dstStage,
                                         dstAccess,
                                         _targetLayout,
                                         expectedLayout,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         *_image,
                                         vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

    vk::DependencyInfo depInfo{
        {},            // memoryBarriers (global)
        memoryBarrier, // memoryBarrier2
        {},            // bufferBarriers
        imageBarrier   // imageBarrier2
    };

    cmdBuf.pipelineBarrier2(depInfo);

    _targetLayout = expectedLayout;
}

void Image::transitionLayout(const Context &ctx, vk::ImageLayout expectedLayout) {
    const vk::CommandPoolCreateInfo cmdPoolCreateInfo({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                                      ctx.familyQueueIdx());
    auto cmdPool = ctx.device().createCommandPool(cmdPoolCreateInfo);
    const vk::CommandBufferAllocateInfo cmdBufferAllocInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::raii::CommandBuffer cmdBuffer = std::move(ctx.device().allocateCommandBuffers(cmdBufferAllocInfo).front());
    const vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuffer.begin(CmdBufferBeginInfo);
    addTransitionLayoutCommand(cmdBuffer, expectedLayout);
    cmdBuffer.end();
    vk::SubmitInfo submitInfo({}, {}, *cmdBuffer);
    auto queue = ctx.device().getQueue(ctx.familyQueueIdx(), 0);
    auto fence = ctx.device().createFence({});
    queue.submit(submitInfo, *fence);
    const uint64_t timeout = static_cast<uint64_t>(-1);
    auto res = ctx.device().waitForFences({*fence}, true, timeout);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Error while waiting for fence.");
    }
}

void Image::allocateMemory(const Context &ctx) {
    // Allocate memory
    if (!_memoryManager->isInitalized()) {
        vk::MemoryPropertyFlags memoryFlags;
        if (_imageInfo.isAliased) {
            memoryFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        } else {
            memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        }
        _memoryManager->allocateDeviceMemory(ctx, memoryFlags);
    }

    // Bind image to memory
    const vk::BindImageMemoryInfo bindInfo(*_image, *_memoryManager->getDeviceMemory(), _imageInfo.memoryOffset);
    ctx.device().bindImageMemory2(vk::ArrayProxy<vk::BindImageMemoryInfo>(bindInfo));

    const vk::ImageAspectFlags aspectMask = getImageAspectMaskForVkFormat(_dataType);
    const vk::ImageSubresourceRange subRange(aspectMask, /* baseMipLevel_ */ 0,
                                             /* levelCount_ */ _imageInfo.mips, /* baseArrayLayer_ */ 0,
                                             /* layerCount_ */ 1);
    const vk::ImageViewCreateInfo imageViewCreateInfo(vk::ImageViewCreateFlags(), *_image, vk::ImageViewType::e2D,
                                                      _dataType, vk::ComponentMapping(), subRange);

    _imageView = vk::raii::ImageView(ctx.device(), imageViewCreateInfo);

    trySetVkRaiiObjectDebugName(ctx, _imageView, _imageInfo.debugName + " view (default)");

    // Create image view for each lod
    if (_imageInfo.mips > 1) {
        for (uint32_t m = 0; m < _imageInfo.mips; m++) {
            const vk::ImageSubresourceRange mipMapSubRange(aspectMask,
                                                           /* baseMipLevel_ */ m,
                                                           /* levelCount_ */ 1, /* baseArrayLayer_ */ 0,
                                                           /* layerCount_ */ 1);

            const vk::ImageViewCreateInfo mipMapViewCreateInfo(vk::ImageViewCreateFlags(), *_image,
                                                               vk::ImageViewType::e2D, _dataType,
                                                               vk::ComponentMapping(), mipMapSubRange);

            auto imageViewMip = vk::raii::ImageView(ctx.device(), mipMapViewCreateInfo);
            trySetVkRaiiObjectDebugName(ctx, _imageView,
                                        _imageInfo.debugName + " view (mip " + std::to_string(m) + ")");
            _imageViewMips.emplace_back(std::move(imageViewMip));
        }
    }
}

void Image::resetLayout() { _targetLayout = vk::ImageLayout::eUndefined; }

void Image::fillFromDescription(const Context &ctx, const ImageDesc &desc) {
    std::vector<uint8_t> data;
    vk::Format fileFormat = vk::Format::eUndefined;
    const vk::ImageAspectFlags aspectMask = getImageAspectMaskForVkFormat(_dataType);

    // Determine image data, from file or zeroed
    if (desc.src) {
        loadDataFromDDS(desc.src.value(), data, fileFormat);
    } else {
        data.resize(dataSize());
        std::fill_n(data.begin(), dataSize(), 0);
    }
    _targetLayout = vk::ImageLayout::eGeneral;

    // D32S8 stencil discard case
    if ((_dataType == vk::Format::eR32Sfloat || _dataType == vk::Format::eD32Sfloat) &&
        fileFormat == vk::Format::eD32SfloatS8Uint) {
        // Depth stencil discarding
        std::vector<uint8_t> bodge_data(dataSize());
        bool has_stencil_data = false;
        for (uint64_t i = 0; i < totalElementsFromShape(shape()); ++i) {
            uint64_t depth_idx = i * elementSizeFromVkFormat(fileFormat);
            uint64_t bodge_idx = i * elementSizeFromVkFormat(_dataType);
            if (data[depth_idx + 4]) {
                has_stencil_data = true;
            }
            bodge_data[bodge_idx + 0] = data[depth_idx + 0];
            bodge_data[bodge_idx + 1] = data[depth_idx + 1];
            bodge_data[bodge_idx + 2] = data[depth_idx + 2];
            bodge_data[bodge_idx + 3] = data[depth_idx + 3];
        }
        if (has_stencil_data) {
            mlsdk::logging::warning("Ignoring stencil data");
        }
        data = std::move(bodge_data);
    }

    if (data.size() != dataSize()) {
        throw std::runtime_error("Expected DDS image input size is " + std::to_string(dataSize()) + ", but got " +
                                 std::to_string(data.size()) + " instead");
    }

    // Create Image barrier
    auto accessFlag = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                      vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eHostWrite;
    auto memoryBarrier = vk::MemoryBarrier2(vk::PipelineStageFlagBits2::eAllCommands, accessFlag,
                                            vk::PipelineStageFlagBits2::eAllCommands, accessFlag);
    auto imageBarrier = vk::ImageMemoryBarrier2();
    imageBarrier.srcAccessMask = vk::AccessFlagBits2::eNone;
    imageBarrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
    imageBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    imageBarrier.dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer;
    imageBarrier.oldLayout = _initialLayout;
    imageBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = _image;
    imageBarrier.subresourceRange.aspectMask = aspectMask;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = _imageInfo.mips;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    void *pBufferDeviceMemory = _stagingBufferDeviceMemory.mapMemory(0, data.size());
    std::memcpy(pBufferDeviceMemory, data.data(), data.size());
    _stagingBufferDeviceMemory.unmapMemory();

    // Setup of casting operation
    const vk::CommandPoolCreateInfo cmdPoolCreateInfo({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                                      ctx.familyQueueIdx());
    auto cmdPool = ctx.device().createCommandPool(cmdPoolCreateInfo);
    const vk::CommandBufferAllocateInfo cmdBufferAllocInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::raii::CommandBuffer cmdBuffer = std::move(ctx.device().allocateCommandBuffers(cmdBufferAllocInfo).front());
    const vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuffer.begin(CmdBufferBeginInfo);

    vk::BufferImageCopy region{};
    const vk::Extent3D extent(static_cast<uint32_t>(_imageInfo.shape[1]), static_cast<uint32_t>(_imageInfo.shape[2]),
                              static_cast<uint32_t>(_imageInfo.shape[3]));
    const vk::Offset3D offset(0, 0, 0);
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = aspectMask;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = offset;
    region.imageExtent = extent;

    cmdBuffer.pipelineBarrier2(vk::DependencyInfo((vk::DependencyFlags)0, memoryBarrier, {}, imageBarrier));
    cmdBuffer.copyBufferToImage(_stagingBuffer, _image, vk::ImageLayout::eTransferDstOptimal, region);

    // Create blit command
    vk::ImageBlit2 blit{};
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.srcSubresource.aspectMask = aspectMask;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;
    blit.dstSubresource.aspectMask = aspectMask;

    int32_t mipWidth = static_cast<int32_t>(_imageInfo.shape[1]);
    int32_t mipHeight = static_cast<int32_t>(_imageInfo.shape[2]);
    for (uint32_t i = 1; i < _imageInfo.mips; i++) {
        // Create barrier before the first blit and between blits
        imageBarrier.subresourceRange.baseMipLevel = i - 1;
        imageBarrier.subresourceRange.levelCount = 1;
        imageBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        imageBarrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        imageBarrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        imageBarrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
        imageBarrier.srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer;
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo((vk::DependencyFlags)0, memoryBarrier, {}, imageBarrier));

        blit.srcOffsets[0] = vk::Offset3D(0, 0, 0);
        blit.srcOffsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
        blit.srcSubresource.mipLevel = i - 1;
        blit.dstOffsets[0] = vk::Offset3D(0, 0, 0);
        blit.dstOffsets[1] = vk::Offset3D(std::max(mipWidth / 2, 1), std::max(mipHeight / 2, 1), 1);
        blit.dstSubresource.mipLevel = i;
        cmdBuffer.blitImage2(vk::BlitImageInfo2(_image, vk::ImageLayout::eTransferSrcOptimal, _image,
                                                vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear));

        // Update mip dimensions
        mipHeight = mipHeight / 2;
        mipWidth = mipWidth / 2;
    }

    // Restore the last miplevel back to transfer source
    imageBarrier.subresourceRange.baseMipLevel = _imageInfo.mips - 1;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    imageBarrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    imageBarrier.srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer;
    imageBarrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    imageBarrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
    cmdBuffer.pipelineBarrier2(vk::DependencyInfo((vk::DependencyFlags)0, memoryBarrier, {}, imageBarrier));

    // Transition into target image format
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = _imageInfo.mips;
    imageBarrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    imageBarrier.newLayout = _targetLayout;
    imageBarrier.srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer;
    imageBarrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    imageBarrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
    cmdBuffer.pipelineBarrier2(vk::DependencyInfo((vk::DependencyFlags)0, memoryBarrier, {}, imageBarrier));

    cmdBuffer.end();
    vk::SubmitInfo submitInfo({}, {}, *cmdBuffer);
    auto queue = ctx.device().getQueue(ctx.familyQueueIdx(), 0);
    auto fence = ctx.device().createFence({});
    queue.submit(submitInfo, *fence);
    const uint64_t timeout = static_cast<uint64_t>(-1);
    auto res = ctx.device().waitForFences({*fence}, true, timeout);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Error while waiting for fence.");
    }
}

std::vector<char> Image::getImageData(Context &ctx) {
    // Use staging buffer to get image data
    const vk::CommandPoolCreateInfo cmdPoolCreateInfo({vk::CommandPoolCreateFlagBits::eResetCommandBuffer},
                                                      ctx.familyQueueIdx());
    auto cmdPool = ctx.device().createCommandPool(cmdPoolCreateInfo);
    const vk::CommandBufferAllocateInfo cmdBufferAllocInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::raii::CommandBuffer cmdBuffer = std::move(ctx.device().allocateCommandBuffers(cmdBufferAllocInfo).front());
    const vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuffer.begin(CmdBufferBeginInfo);
    vk::BufferImageCopy region{};
    const vk::Extent3D extent(static_cast<uint32_t>(_imageInfo.shape[1]), static_cast<uint32_t>(_imageInfo.shape[2]),
                              static_cast<uint32_t>(_imageInfo.shape[3]));
    const vk::Offset3D offset(0, 0, 0);
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = getImageAspectMaskForVkFormat(_imageInfo.format);
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = offset;
    region.imageExtent = extent;

    addTransitionLayoutCommand(cmdBuffer, vk::ImageLayout::eGeneral);
    cmdBuffer.copyImageToBuffer(_image, vk::ImageLayout::eGeneral, _stagingBuffer, {region});
    cmdBuffer.end();

    vk::SubmitInfo submitInfo({}, {}, *cmdBuffer);
    auto queue = ctx.device().getQueue(ctx.familyQueueIdx(), 0);
    auto fence = ctx.device().createFence({});
    queue.submit(submitInfo, *fence);
    const uint64_t timeout = static_cast<uint64_t>(-1);
    auto res = ctx.device().waitForFences({*fence}, true, timeout);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Error while waiting for fence.");
    }

    std::vector<char> data(dataSize());
    void *pBufferDeviceMemory = _stagingBufferDeviceMemory.mapMemory(0, data.size());
    std::memcpy(data.data(), pBufferDeviceMemory, data.size());
    _stagingBufferDeviceMemory.unmapMemory();
    return data;
}

void Image::store(Context &ctx, const std::string &filename) {
    auto data = getImageData(ctx);
    saveDataToDDS(filename, *this, data);
}

bool Image::isSampled() const { return _imageInfo.isSampled; }

const std::string &Image::debugName() const { return _imageInfo.debugName; }

vk::ImageLayout Image::getImageLayout() const { return _targetLayout; }
} // namespace mlsdk::scenariorunner
