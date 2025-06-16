/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "context.hpp"
#include "vulkan/vulkan_raii.hpp"
#include "vulkan/vulkan_structs.hpp"

#include <string>

namespace mlsdk::scenariorunner {

template <class VkRaiiObject>
void trySetVkRaiiObjectDebugName(const Context &ctx, const VkRaiiObject &object, const std::string &debugName) {
    if (ctx.gpuDebugMarkersEnabled()) {
        ctx.device().setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
            VkRaiiObject::objectType, reinterpret_cast<uint64_t>(static_cast<typename VkRaiiObject::CType>(*object)),
            debugName.c_str()});
    }
}

} // namespace mlsdk::scenariorunner
