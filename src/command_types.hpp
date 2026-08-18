/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "commands.hpp"
#include "compute.hpp"
#include "types.hpp"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlsdk::scenariorunner {

/// \brief Compute data with typed bindings
struct DispatchComputeData {
    explicit DispatchComputeData(ShaderId shader) : shader(shader) {}

    std::string debugName;
    std::vector<TypedBinding> bindings;
    ComputeDispatch computeDispatch{};
    ShaderId shader;
    bool implicitBarrier{true};
    std::optional<RawDataId> pushData;
};

/// \brief Fragment (graphics) data with typed bindings
struct DispatchFragmentData {
    DispatchFragmentData(ShaderId vertexShader, ShaderId fragmentShader)
        : vertexShader(vertexShader), fragmentShader(fragmentShader) {}

    std::string debugName;
    std::vector<TypedBinding> bindings;
    ShaderId vertexShader;
    ShaderId fragmentShader;
    struct Attachment {
        ImageId resource;
        std::optional<uint32_t> lod;
    };
    std::vector<Attachment> colorAttachments;
    std::optional<vk::Extent2D> renderExtent;
    bool implicitBarrier{true};
    std::optional<RawDataId> pushData;
};

struct ResolvedPushConstantMap {
    RawDataId pushData;
    std::string shaderTarget;
};

struct ResolvedShaderSubstitution {
    ShaderId shader;
    std::string target;
};

/// \brief Compute data graph with typed bindings
struct DispatchDataGraphData {
    explicit DispatchDataGraphData(DataGraphId dataGraph) : dataGraph(dataGraph) {}

    DataGraphId dataGraph;
    std::string debugName;
    std::vector<TypedBinding> bindings;
    std::vector<ResolvedPushConstantMap> pushConstants;
    std::vector<ResolvedShaderSubstitution> shaderSubstitutions;
    bool implicitBarrier{true};
};

/// \brief SPIR-V-only data graph with typed bindings and constants
struct DispatchSpirvGraphData {
    explicit DispatchSpirvGraphData(ShaderId graphShader) : graphShader(graphShader) {}

    ShaderId graphShader;
    std::string debugName;
    std::vector<TypedBinding> bindings;
    std::vector<GraphConstantResourceId> graphConstants;
    bool implicitBarrier{true};
};

/// \brief Optical flow data graph with typed bindings
struct DispatchOpticalFlowData {
    DispatchOpticalFlowData(TypedBinding search, TypedBinding reference, TypedBinding output)
        : searchImage(search), templateImage(reference), outputImage(output) {}

    std::string debugName;
    TypedBinding searchImage;
    TypedBinding templateImage;
    TypedBinding outputImage;
    std::optional<TypedBinding> hintMotionVectors;
    std::optional<TypedBinding> outputCost;
    uint32_t width{0};
    uint32_t height{0};
    OpticalFlowPerformanceLevel performanceLevel{OpticalFlowPerformanceLevel::Medium};
    uint32_t executionFlags{0};
    OpticalFlowGridSize gridSize{OpticalFlowGridSize::e1x1};
    uint32_t meanFlowL1NormHint{0};

    bool implicitBarrier{true};
};

/// \brief Typed barriers
struct DispatchBarrierData {
    std::vector<MemoryBarrierId> memoryBarriers;
    std::vector<ImageBarrierId> imageBarriers;
    std::vector<TensorBarrierId> tensorBarriers;
    std::vector<BufferBarrierId> bufferBarriers;
};

/// \brief Typed resources
struct MarkBoundaryData {
    std::vector<BufferId> buffers;
    std::vector<ImageId> images;
    std::vector<TensorId> tensors;
};

namespace detail {
using ScenarioCommand =
    std::variant<DispatchComputeData, DispatchFragmentData, DispatchDataGraphData, DispatchSpirvGraphData,
                 DispatchOpticalFlowData, DispatchBarrierData, MarkBoundaryData>;
} // namespace detail

} // namespace mlsdk::scenariorunner
