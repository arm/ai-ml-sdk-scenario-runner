/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "device_selection.hpp"
#include "scenario_build_data.hpp"
#include "scenario_runner/scenario_options.hpp"

#include <gtest/gtest.h>

namespace mlsdk::scenariorunner {
namespace {

vk::QueueFlags getRequiredQueueFlags(const detail::ScenarioBuildData &buildData) {
    return detail::getDeviceRequirements(buildData, ScenarioOptions{}).queueFlags;
}

TEST(QueueSelection, TransferOnlyWorkloadAcceptsExplicitOrImplicitTransferSupport) {
    const vk::QueueFlags noWorkload;

    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, noWorkload));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eGraphics, noWorkload));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eCompute, noWorkload));
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eVideoDecodeKHR, noWorkload));
}

TEST(QueueSelection, ExplicitTransferRequirementAcceptsImplicitTransferSupport) {
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eTransfer));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eGraphics, vk::QueueFlagBits::eTransfer));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eTransfer));
}

TEST(QueueSelection, WorkloadCapabilityIsRequiredInAdditionToTransferSupport) {
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eCompute));
}

TEST(QueueSelection, PipelineBarrierStagesDetermineWorkloadRequirements) {
    detail::ScenarioBuildData buildData;
    MemoryBarrierInfo barrierInfo;
    barrierInfo.srcStages = {PipelineStage::Graphics};
    barrierInfo.dstStages = {PipelineStage::Graph};

    PipelineBarrierData command;
    command.memoryBarriers.push_back(buildData.resources.addMemoryBarrier(std::move(barrierInfo)));
    buildData.commands.emplace_back(std::move(command));

    const auto requiredFlags = getRequiredQueueFlags(buildData);
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eCompute, requiredFlags));
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eGraphics, requiredFlags));
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eTransfer,
                                                   requiredFlags));
    EXPECT_TRUE(detail::supportsQueueRequirements(
        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eTransfer, requiredFlags));
}

TEST(QueueSelection, ComputePipelineBarrierRequiresComputeCapability) {
    detail::ScenarioBuildData buildData;
    MemoryBarrierInfo barrierInfo;
    barrierInfo.srcAccess = MemoryAccess::ComputeShaderWrite;
    barrierInfo.dstAccess = MemoryAccess::ComputeShaderRead;

    PipelineBarrierData command;
    command.memoryBarriers.push_back(buildData.resources.addMemoryBarrier(std::move(barrierInfo)));
    buildData.commands.emplace_back(std::move(command));

    const auto requiredFlags = getRequiredQueueFlags(buildData);
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, requiredFlags));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eCompute, requiredFlags));
}

TEST(QueueSelection, DataGraphAccessWithAllStagesRequiresDataGraphCapability) {
    detail::ScenarioBuildData buildData;
    MemoryBarrierInfo barrierInfo;
    barrierInfo.srcAccess = MemoryAccess::GraphWrite;
    barrierInfo.dstAccess = MemoryAccess::GraphRead;

    PipelineBarrierData command;
    command.memoryBarriers.push_back(buildData.resources.addMemoryBarrier(std::move(barrierInfo)));
    buildData.commands.emplace_back(std::move(command));

    const auto requiredFlags = getRequiredQueueFlags(buildData);
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, requiredFlags));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eTransfer,
                                                  requiredFlags));
}

TEST(QueueSelection, GenericMemoryAccessWithAllStagesAllowsTransferCapability) {
    detail::ScenarioBuildData buildData;
    MemoryBarrierInfo barrierInfo;
    barrierInfo.srcAccess = MemoryAccess::MemoryWrite;
    barrierInfo.dstAccess = MemoryAccess::MemoryRead;

    PipelineBarrierData command;
    command.memoryBarriers.push_back(buildData.resources.addMemoryBarrier(std::move(barrierInfo)));
    buildData.commands.emplace_back(std::move(command));

    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, getRequiredQueueFlags(buildData)));
}

TEST(QueueSelection, DataGraphQueueAlsoNeedsTransferCapability) {
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eDataGraphARM, vk::QueueFlagBits::eDataGraphARM));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eDataGraphARM | vk::QueueFlagBits::eTransfer,
                                                  vk::QueueFlagBits::eDataGraphARM));
}

TEST(QueueSelection, MultiMipImageRequiresGraphicsCapability) {
    detail::ScenarioBuildData buildData;
    ImageInfo imageInfo{};
    imageInfo.mips = 2;
    buildData.resources.addImage(std::move(imageInfo));

    const auto requiredFlags = getRequiredQueueFlags(buildData);
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, requiredFlags));
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eCompute, requiredFlags));
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eGraphics, requiredFlags));
}

TEST(QueueSelection, AliasedImageLayoutTransitionRequiresGraphicsAndCompute) {
    detail::ScenarioBuildData buildData;
    ImageInfo imageInfo{};
    imageInfo.debugName = "image";
    imageInfo.shape = {1, 2, 2, 1};
    imageInfo.format = imageInfo.targetFormat = vk::Format::eR8Uint;
    imageInfo.isStorage = true;
    imageInfo.mips = 1;
    const auto image = buildData.resources.addImage(imageInfo);
    buildData.initializations.emplace_back(detail::ImageInitialization{image, std::nullopt, "image"});

    // An unaliased image is zero-initialized by upload, which only needs transfer support.
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, getRequiredQueueFlags(buildData)));

    const auto buffer = buildData.resources.addBuffer(BufferInfo{"buffer", 4, 0});
    const auto group = buildData.groupManager.createMemoryGroup();
    buildData.groupManager.addResourceToGroup(group, image);
    buildData.groupManager.addResourceToGroup(group, buffer);

    // Without initial data, an aliased image instead records a shader-stage layout transition.
    const auto requiredFlags = getRequiredQueueFlags(buildData);
    EXPECT_FALSE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, requiredFlags));
    EXPECT_FALSE(
        detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eCompute, requiredFlags));
    EXPECT_FALSE(
        detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eGraphics, requiredFlags));
    EXPECT_TRUE(
        detail::supportsQueueRequirements(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute, requiredFlags));

    // Supplying initial data takes the upload path even when the image is aliased.
    auto &initialization = std::get<detail::ImageInitialization>(buildData.initializations.front());
    initialization.data = ImageData{std::vector<std::byte>(4), imageInfo.shape, imageInfo.format, 1};
    EXPECT_TRUE(detail::supportsQueueRequirements(vk::QueueFlagBits::eTransfer, getRequiredQueueFlags(buildData)));
}

} // namespace
} // namespace mlsdk::scenariorunner
