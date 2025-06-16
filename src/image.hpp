/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "context.hpp"
#include "resource.hpp"
#include "resource_desc.hpp"
#include "types.hpp"

namespace mlsdk::scenariorunner {

class ResourceMemoryManager;

class Image : public Resource {
  public:
    /// \brief Constructor
    ///
    /// \param ctx Contextual information about the Vulkan®  instance
    /// \param imageInfo ImageInfo struct
    /// \param memoryManager Memory manager for this resource
    Image(Context &ctx, const ImageInfo &imageInfo, std::shared_ptr<ResourceMemoryManager> memoryManager);
    Image() = default;

    /// \brief Image accessor
    /// \return The underlying Vulkan® image
    vk::Image image() const;

    /// \brief Image view accessor
    /// \return The underlying Vulkan® image view
    vk::ImageView imageView() const;

    vk::ImageView imageView(uint32_t lod) const;

    /// \brief Image sampler accessor
    /// \return The underlying Vulkan® image sampler
    vk::Sampler sampler() const;

    /// \brief Get image packed data size in bytes
    /// \return Size of image packed data in bytes
    uint64_t dataSize() const;

    /// \brief Get image allocated memory size in bytes
    /// \return Size of image memory in bytes
    uint64_t memSize() const;

    /// \brief Get image data type
    /// \return Data type of image
    vk::Format dataType() const;

    /// \brief Get image shape
    /// \return Vector containing the shape of image
    const std::vector<int64_t> &shape() const;
    void transitionLayout(vk::raii::CommandBuffer &cmdBuf, vk::ImageLayout expectedLayout);
    void allocateMemory(Context &ctx);
    void resetLayout();

    std::shared_ptr<ResourceMemoryManager> getMemoryManager();

    void fillFromDescription(Context &ctx, const ImageDesc &desc);

    // Sampler settings helper functions
    static vk::Filter convertFilter(const FilterMode filter);

    static vk::SamplerMipmapMode convertSamplerMipmapMode(const FilterMode mode);

    static vk::SamplerAddressMode convertSamplerAddressMode(const AddressMode mode);

    static vk::BorderColor convertBorderColor(const BorderColor color);

    static vk::ImageTiling convertTiling(const Tiling tiling);

    void store(Context &ctx, const std::string &filename) override;

    bool isSampled() const;

    vk::ImageLayout getImageLayout() const;

    const std::string &debugName() const;
    const ImageInfo &getInfo() const { return _imageInfo; }

  private:
    std::vector<char> getImageData(Context &ctx);

    vk::raii::Image _image{nullptr};
    vk::raii::Buffer _stagingBuffer{nullptr};
    vk::raii::DeviceMemory _stagingBufferDeviceMemory{nullptr};
    vk::raii::ImageView _imageView{nullptr};
    vk::raii::Sampler _sampler{nullptr};
    vk::Format _dataType{};
    const ImageInfo _imageInfo{};
    std::shared_ptr<ResourceMemoryManager> _memoryManager;
    uint32_t _mips{};
    std::vector<vk::raii::ImageView> _imageViewMips{};
    vk::ImageLayout _initialLayout{};
    vk::ImageLayout _targetLayout{};
};

} // namespace mlsdk::scenariorunner
