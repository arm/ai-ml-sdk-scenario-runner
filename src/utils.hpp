/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "context.hpp"
#include "numpy.hpp"
#include "resource_desc.hpp"

#include "spirv-tools/libspirv.hpp"
#include "vgf/decoder.h"
#include "vulkan/vulkan_raii.hpp"

namespace mlsdk::scenariorunner {

/** Data format components (channels) count.
 *
 *  @param[in] format vk::format enum value.
 *
 *  @return number of components.
 *
 */
uint32_t numComponentsFromVkFormat(vk::Format format);

/** Size in bytes of a given vk::format data type.
 *
 *  @param[in] format vk::format enum value.
 *
 *  @return size of the format in bytes.
 *
 */
uint32_t elementSizeFromVkFormat(vk::Format format);

/** Get vk::Format enum element from its string description.
 *
 *  @param[in] format String description of the VkFormat.
 *
 *  @return vk::Format enum element
 *
 */
vk::Format getVkFormatFromString(const std::string &format);

/** Get vk::ImageAspectFlags value for a given vk::Format data type.
 *
 *  @param[in] format vk::Format enum value
 *
 *  @return aspect mask for the format
 *
 */
vk::ImageAspectFlags getImageAspectMaskForVkFormat(vk::Format format);

/** Get vk::Format enum element from the parser type.
 *
 *  @param[in] format parser type for encoding VkFormat.
 *
 *  @return vk::Format enum element
 *
 */
vk::Format getVkFormatFromParser(const mlsdk_vk_format &format);

/** Get NumPy dtype from vk::Format description.
 *
 *  @param[in] format vk::Format enum element.
 *
 *  @return NumPy dtype
 *
 */
const mlsdk::numpy::dtype getDTypeFromVkFormat(vk::Format format);

/** Calculates the total number of elements of a vector that represents a shape.
 *
 * @param[in] shape Vector representing the shape of a tensor.
 *
 * @return Total number of elements from the given shape.
 *
 */
uint64_t totalElementsFromShape(const std::vector<int64_t> &shape);

/** Find the memory index in a device that fulfills the required
 * properties
 *
 * @param[in] ctx Internal runner context
 * @param[in] memTypeBits Memory type flag bits
 * @param[in] required   Required memory properties
 * @return The index of the memory type if succeeds else UINT32_MAX on failure
 */
uint32_t findMemoryIdx(const Context &ctx, uint32_t memTypeBits, vk::MemoryPropertyFlags required);

/** Allocate device memory
 *
 * @param[in] ctx Internal runner context
 * @param[in] size Size in bytes
 * @param[in] memoryPropertyFlags Required memory property flags
 * @param[in] memoryTypeBits Bit mask of allowed memory types
 * @return The index of the memory type if succeeds else UINT32_MAX on failure
 */
vk::raii::DeviceMemory allocateDeviceMemory(const Context &ctx, const vk::DeviceSize size,
                                            const vk::MemoryPropertyFlags memoryPropertyFlags,
                                            const uint32_t memoryTypeBits);

/** Read the shader code from file. In case of GLSL shader, the code will be compiled into SPIR-V before being
 * returned.
 *
 * @param[in] shaderDesc Shader description meta-data. Contains the filename, file type, etc.
 * @return Bytes read from the file
 */
std::vector<uint32_t> readShaderCode(const ShaderDesc &shaderDesc);

/** Consumer function for messages communicated from the SPIRV-Tools library
 *
 *  @param[in] level    Message level
 *  @param[in] position Message position
 *  @param[in] source   Message source
 *  @param[in] message  Message string
 */
void SPIRVMessageConsumer(spv_message_level_t level, const char *source, const spv_position_t &position,
                          const char *message);

/** A naive scope exit runner
 *
 */
template <typename F> class ScopeExit {
  public:
    explicit ScopeExit(const std::function<F> &f) : _f(f) {}
    ~ScopeExit() noexcept { _f(); }

  private:
    std::function<F> _f;
};

} // namespace mlsdk::scenariorunner
