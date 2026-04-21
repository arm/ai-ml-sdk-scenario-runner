/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "guid.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlsdk::scenariorunner {

enum class CommandType {
    Unknown,
    DispatchCompute,
    DispatchDataGraph,
    DispatchOpticalFlow,
    DispatchSpirvGraph,
    DispatchFragment,
    DispatchBarrier,
    MarkBoundary
};

enum class DescriptorType { Unknown, Auto, StorageImage };

enum class OpticalFlowGridSize : uint32_t {
    Invalid = 0xFFFFFFFFu,
    e1x1 = 0,
    e2x2 = 1,
    e4x4 = 2,
    e8x8 = 3,
};

enum class OpticalFlowPerformanceLevel : uint32_t {
    Invalid = 0xFFFFFFFFu,
    Unknown = 0,
    Slow = 1,
    Medium = 2,
    Fast = 3,
};

enum class OpticalFlowExecutionFlag : uint32_t {
    Invalid = 0,
    DisableTemporalHints = 0x1u,
    InputUnchanged = 0x2u,
    ReferenceUnchanged = 0x4u,
    InputIsPreviousReference = 0x8u,
    ReferenceIsPreviousInput = 0x10u,
};

/**
 * @brief Commands are executed by the ScenarioRunner. CommandDesc describes a Command.
 *
 */
struct CommandDesc {
    CommandDesc() = default;
    explicit CommandDesc(CommandType commandType);
    virtual ~CommandDesc() = default;

    CommandType commandType = CommandType::Unknown;
};

/**
 * @brief A Binding maps a resource reference to a Vulkan Descriptor Set and ID. BindingDesc describes a Binding.
 *
 */
struct BindingDesc {
    uint32_t set{};
    uint32_t id{};
    Guid resourceRef;
    std::optional<uint32_t> lod;
    DescriptorType descriptorType = DescriptorType::Auto;
};

/**
 * @brief The DispatchCompute command dispatches a compute shader to execute. DispatchComputeDesc describes a
 * DispatchCompute Command.
 *
 */
struct DispatchComputeDesc : CommandDesc {
    DispatchComputeDesc();

    std::string debugName;
    std::vector<BindingDesc> bindings;
    std::vector<uint32_t> rangeND;
    Guid shaderRef;
    bool implicitBarrier{true};
    std::optional<Guid> pushDataRef;
};

/**
 * @brief The DispatchFragment command dispatches a fragment shader to execute. DispatchFragmentDesc describes a
 * DispatchFragment Command.
 *
 */
struct FragmentAttachmentDesc {
    Guid resourceRef;
    std::optional<uint32_t> lod;
};

struct DispatchFragmentDesc : CommandDesc {
    DispatchFragmentDesc();

    std::string debugName;
    std::vector<BindingDesc> bindings;
    Guid vertexShaderRef;
    Guid fragmentShaderRef;
    std::vector<FragmentAttachmentDesc> colorAttachments;
    std::optional<std::array<uint32_t, 2>> renderExtent;
    bool implicitBarrier{true};
    std::optional<Guid> pushDataRef;
};

/**
 * @brief Maps the raw data resource containing push constants data
 * to the shader node in the graph which consumes these
 *
 */
struct PushConstantMap {
    Guid pushDataRef;
    Guid shaderTarget;
};

/**
 * @brief Describes a placeholder shader node in the graph that will be substituted with an actual shader
 * implementation
 *
 */
struct ShaderSubstitution {
    Guid shaderRef;
    std::string target;
};

/**
 * @brief The DispatchDataGraph command dispatches a data graph shader to execute. DispatchDataGraphDesc describes a
 * DispatchDataGraph.
 *
 */
struct DispatchDataGraphDesc : CommandDesc {
    DispatchDataGraphDesc();

    Guid dataGraphRef;
    std::string debugName;
    std::vector<BindingDesc> bindings;
    std::vector<PushConstantMap> pushConstants;
    std::vector<ShaderSubstitution> shaderSubstitutions;
    bool implicitBarrier{true};
};

/**
 * @brief The DispatchSpirvGraph command dispatches a SPIR-V-only data graph to execute. DispatchSpirvGraphDesc
 * describes a DispatchSpirvGraph.
 */
struct DispatchSpirvGraphDesc : CommandDesc {
    DispatchSpirvGraphDesc();

    Guid dataGraphRef;
    std::string debugName;
    std::vector<BindingDesc> bindings;
    std::vector<Guid> graphConstants;
    bool implicitBarrier{true};
    std::string entry{"main"};
};

/**
 * @brief The DispatchOpticalFlow command dispatches an optical flow data graph pipeline to execute.
 * DispatchOpticalFlowDesc describes a DispatchOpticalFlow command.
 *
 */
struct DispatchOpticalFlowDesc : CommandDesc {
    DispatchOpticalFlowDesc();

    std::string debugName;
    BindingDesc searchImage;
    BindingDesc templateImage;
    BindingDesc outputImage;
    std::optional<BindingDesc> hintMotionVectors;
    std::optional<BindingDesc> outputCost;

    uint32_t width{0};
    uint32_t height{0};
    OpticalFlowGridSize gridSize{OpticalFlowGridSize::e1x1};
    OpticalFlowPerformanceLevel performanceLevel{OpticalFlowPerformanceLevel::Medium};
    uint32_t executionFlags{0};
    uint32_t meanFlowL1NormHint{0};

    bool implicitBarrier{true};
};

/**
 * @brief The DispatchBarrier command dispatches a barrier to execute. DispatchBarrierDesc describes a
 * DispatchBarrier Command.
 *
 */
struct DispatchBarrierDesc : CommandDesc {
    DispatchBarrierDesc();

    std::vector<std::string> memoryBarriersRef;
    std::vector<std::string> imageBarriersRef;
    std::vector<std::string> tensorBarriersRef;
    std::vector<std::string> bufferBarriersRef;
};

struct MarkBoundaryDesc : CommandDesc {
    MarkBoundaryDesc();

    std::vector<std::string> resources;
};

} // namespace mlsdk::scenariorunner
