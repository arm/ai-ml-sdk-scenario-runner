/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "optical_flow_utils.hpp"
#include "commands.hpp"
#include "data_manager.hpp"
#include "image.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mlsdk::scenariorunner {
namespace {
struct OpticalFlowSelectedExtent {
    uint32_t width{0};
    uint32_t height{0};

    bool operator==(const OpticalFlowSelectedExtent &other) const {
        return width == other.width && height == other.height;
    }
};

OpticalFlowSelectedExtent getOpticalFlowSelectedExtent(const DataManager &dataManager, const TypedBinding &binding,
                                                       const char *bindingRole) {
    if (!dataManager.hasImage(binding.resourceRef)) {
        throw std::runtime_error("Optical flow " + std::string(bindingRole) + " binding must reference an image");
    }

    const auto &image = dataManager.getImage(binding.resourceRef);
    const auto &shape = image.shape();
    if (shape.size() < 3) {
        throw std::runtime_error("Optical flow " + std::string(bindingRole) + " image must have at least 3 dimensions");
    }

    const uint32_t lod = binding.lod.value_or(0);
    const uint32_t mips = std::max(image.getInfo().mips, 1u);
    if (lod >= mips) {
        throw std::runtime_error("Optical flow " + std::string(bindingRole) + " mip level exceeds available mips");
    }

    const uint32_t width = std::max(1u, static_cast<uint32_t>(shape[1]) >> lod);
    const uint32_t height = std::max(1u, static_cast<uint32_t>(shape[2]) >> lod);
    return {width, height};
}
} // namespace

void verifyOpticalFlowConfig(const DataManager &dataManager, const TypedBinding &searchImageBinding,
                             const TypedBinding &templateImageBinding, const TypedBinding &outputImageBinding,
                             const std::optional<TypedBinding> &hintMotionVectorsBinding,
                             const std::optional<TypedBinding> &outputCostBinding, uint32_t width, uint32_t height,
                             OpticalFlowGridSize gridSize) {
    const auto &searchImage = dataManager.getImage(searchImageBinding.resourceRef);
    const auto &templateImage = dataManager.getImage(templateImageBinding.resourceRef);

    const auto searchExtent = getOpticalFlowSelectedExtent(dataManager, searchImageBinding, "search");
    const auto templateExtent = getOpticalFlowSelectedExtent(dataManager, templateImageBinding, "template");
    const auto outputExtent = getOpticalFlowSelectedExtent(dataManager, outputImageBinding, "output");

    if (width != searchExtent.width || height != searchExtent.height) {
        throw std::runtime_error(
            "Optical flow search image dimensions do not match specified input width/height at the selected mip level");
    }
    if (!(searchExtent == templateExtent)) {
        throw std::runtime_error(
            "Optical flow search and template images must have the same dimensions at the selected mip levels");
    }
    if (searchImage.dataType() != templateImage.dataType()) {
        throw std::runtime_error("Optical flow search and template images must have the same data type");
    }

    if (outputCostBinding.has_value()) {
        const auto costExtent = getOpticalFlowSelectedExtent(dataManager, outputCostBinding.value(), "output_cost");
        if (!(costExtent == outputExtent)) {
            throw std::runtime_error("Optical flow output cost image must have the same dimensions as the output flow "
                                     "vector image at the selected mip levels");
        }
    }
    if (hintMotionVectorsBinding.has_value()) {
        const auto hintExtent =
            getOpticalFlowSelectedExtent(dataManager, hintMotionVectorsBinding.value(), "hint_motion_vectors");
        if (!(hintExtent == outputExtent)) {
            throw std::runtime_error("Optical flow hint motion vector image must have the same dimensions as the "
                                     "output flow vector image at the selected mip levels");
        }
    }

    switch (gridSize) {
    case OpticalFlowGridSize::e1x1:
        if (!(outputExtent == searchExtent)) {
            throw std::runtime_error("Optical flow output flow vector image must have the same dimensions as the "
                                     "input for 1x1 grid size at the selected mip levels");
        }
        break;
    case OpticalFlowGridSize::e2x2:
        if (outputExtent.width != (searchExtent.width + 1) / 2 ||
            outputExtent.height != (searchExtent.height + 1) / 2) {
            throw std::runtime_error("Optical flow output flow vector image dimensions are incompatible with input "
                                     "dimensions for 2x2 grid size at the selected mip levels");
        }
        break;
    case OpticalFlowGridSize::e4x4:
        if (outputExtent.width != (searchExtent.width + 3) / 4 ||
            outputExtent.height != (searchExtent.height + 3) / 4) {
            throw std::runtime_error("Optical flow output flow vector image dimensions are incompatible with input "
                                     "dimensions for 4x4 grid size at the selected mip levels");
        }
        break;
    case OpticalFlowGridSize::e8x8:
        if (outputExtent.width != (searchExtent.width + 7) / 8 ||
            outputExtent.height != (searchExtent.height + 7) / 8) {
            throw std::runtime_error("Optical flow output flow vector image dimensions are incompatible with input "
                                     "dimensions for 8x8 grid size at the selected mip levels");
        }
        break;
    default:
        throw std::runtime_error("Unsupported optical flow grid size");
    }
}

} // namespace mlsdk::scenariorunner
