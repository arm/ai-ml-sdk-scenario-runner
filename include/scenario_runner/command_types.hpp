/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {

/// \brief Optical flow output grid size
enum class OpticalFlowGridSize : uint32_t {
    Invalid = 0xFFFFFFFFu, ///< Invalid grid size
    e1x1 = 0,              ///< One-by-one-pixel grid
    e2x2 = 1,              ///< Two-by-two-pixel grid
    e4x4 = 2,              ///< Four-by-four-pixel grid
    e8x8 = 3,              ///< Eight-by-eight-pixel grid
};

/// \brief Requested optical flow performance level
enum class OpticalFlowPerformanceLevel : uint32_t {
    Invalid = 0xFFFFFFFFu, ///< Invalid performance level
    Unknown = 0,           ///< Performance level has not been specified
    Slow = 1,              ///< Slow performance preset
    Medium = 2,            ///< Medium performance preset
    Fast = 3,              ///< Fast performance preset
};

/// \brief Workgroup count and profiling label for a compute dispatch
struct ComputeDispatch {
    /// @brief Number of workgroups dispatched in the X dimension.
    uint32_t gwcx{1};
    /// @brief Number of workgroups dispatched in the Y dimension.
    uint32_t gwcy{1};
    /// @brief Number of workgroups dispatched in the Z dimension.
    uint32_t gwcz{1};
    /// @brief Label used for profiling output.
    std::string profileName;
};

/// \brief Command that dispatches a compute shader
struct DispatchComputeData {
    /// @brief Construct a command for a compute shader.
    /// @param shader Registered compute shader to dispatch.
    explicit DispatchComputeData(ShaderId shader) : shader(shader) {}

    /// @brief Human-readable name used in diagnostics.
    std::string debugName;
    /// @brief Descriptor bindings supplied to the shader.
    std::vector<TypedBinding> bindings;
    /// @brief Workgroup counts and profiling label.
    ComputeDispatch computeDispatch{};
    /// @brief Compute shader to dispatch.
    ShaderId shader;
    /// @brief Insert implicit synchronization around the dispatch.
    bool implicitBarrier{true};
    /// @brief Optional raw push-constant data.
    std::optional<RawDataId> pushData;
};

/// \brief Command that dispatches a graphics pipeline
struct DispatchFragmentData {
    /// @brief Construct a graphics command from its shader stages.
    /// @param vertexShader Registered vertex shader.
    /// @param fragmentShader Registered fragment shader.
    DispatchFragmentData(ShaderId vertexShader, ShaderId fragmentShader)
        : vertexShader(vertexShader), fragmentShader(fragmentShader) {}

    /// @brief Human-readable name used in diagnostics.
    std::string debugName;
    /// @brief Descriptor bindings supplied to the shaders.
    std::vector<TypedBinding> bindings;
    /// @brief Vertex shader to dispatch.
    ShaderId vertexShader;
    /// @brief Fragment shader to dispatch.
    ShaderId fragmentShader;
    /// @brief Image and optional mip level used as a color attachment.
    struct Attachment {
        /// @brief Image used as the attachment.
        ImageId resource;
        /// @brief Image mip level, or no value for the base level.
        std::optional<uint32_t> lod;
    };
    /// @brief Color attachments written by the graphics pipeline.
    std::vector<Attachment> colorAttachments;
    /// @brief Render area, or no value to derive it from the attachments.
    std::optional<vk::Extent2D> renderExtent;
    /// @brief Insert implicit synchronization around the dispatch.
    bool implicitBarrier{true};
    /// @brief Optional raw push-constant data.
    std::optional<RawDataId> pushData;
};

/// \brief Associates push-constant data with a shader in a data graph
struct ResolvedPushConstantMap {
    /// @brief Registered raw data containing push-constant bytes.
    RawDataId pushData;
    /// @brief Name of the target shader within the data graph.
    std::string shaderTarget;
};

/// \brief Replaces a named shader in a data graph with another shader resource
struct ResolvedShaderSubstitution {
    /// @brief Replacement shader resource.
    ShaderId shader;
    /// @brief Name of the shader replaced within the data graph.
    std::string target;
};

/// \brief Command that dispatches a VGF data graph
struct DispatchDataGraphData {
    /// @brief Construct a command for a VGF data graph.
    /// @param dataGraph Registered VGF data graph to dispatch.
    explicit DispatchDataGraphData(DataGraphId dataGraph) : dataGraph(dataGraph) {}

    /// @brief VGF data graph to dispatch.
    DataGraphId dataGraph;
    /// @brief Human-readable name used in diagnostics.
    std::string debugName;
    /// @brief External resources bound to graph interfaces.
    std::vector<TypedBinding> bindings;
    /// @brief Push-constant data associated with graph shaders.
    std::vector<ResolvedPushConstantMap> pushConstants;
    /// @brief Shader replacements applied to the graph.
    std::vector<ResolvedShaderSubstitution> shaderSubstitutions;
    /// @brief Insert implicit synchronization around the dispatch.
    bool implicitBarrier{true};
};

/// \brief Command that dispatches a SPIR-V™ data graph
struct DispatchSpirvGraphData {
    /// @brief Construct a command for a SPIR-V™ data graph.
    /// @param graphShader Registered SPIR-V™ graph shader to dispatch.
    explicit DispatchSpirvGraphData(ShaderId graphShader) : graphShader(graphShader) {}

    /// @brief SPIR-V™ graph shader to dispatch.
    ShaderId graphShader;
    /// @brief Human-readable name used in diagnostics.
    std::string debugName;
    /// @brief External resources bound to graph interfaces.
    std::vector<TypedBinding> bindings;
    /// @brief Constants supplied to the graph.
    std::vector<GraphConstantResourceId> graphConstants;
    /// @brief Insert implicit synchronization around the dispatch.
    bool implicitBarrier{true};
};

/// \brief Command that dispatches an optical flow data graph
struct DispatchOpticalFlowData {
    /// @brief Construct an optical-flow command from its required images.
    /// @param search Search image binding.
    /// @param reference Template image binding.
    /// @param output Output flow image binding.
    DispatchOpticalFlowData(TypedBinding search, TypedBinding reference, TypedBinding output)
        : searchImage(search), templateImage(reference), outputImage(output) {}

    /// @brief Human-readable name used in diagnostics.
    std::string debugName;
    /// @brief Search image.
    TypedBinding searchImage;
    /// @brief Template image.
    TypedBinding templateImage;
    /// @brief Output motion-vector image.
    TypedBinding outputImage;
    /// @brief Optional motion-vector hints.
    std::optional<TypedBinding> hintMotionVectors;
    /// @brief Optional output cost image.
    std::optional<TypedBinding> outputCost;
    /// @brief Input image width in pixels.
    uint32_t width{0};
    /// @brief Input image height in pixels.
    uint32_t height{0};
    /// @brief Requested performance preset.
    OpticalFlowPerformanceLevel performanceLevel{OpticalFlowPerformanceLevel::Medium};
    /// @brief Vulkan® optical-flow execution flags.
    uint32_t executionFlags{0};
    /// @brief Output motion-vector grid size.
    OpticalFlowGridSize gridSize{OpticalFlowGridSize::e1x1};
    /// @brief Mean-flow L1-norm hint supplied to the implementation.
    uint32_t meanFlowL1NormHint{0};

    /// @brief Insert implicit synchronization around the dispatch.
    bool implicitBarrier{true};
};

/// \brief Command that executes a collection of registered barriers
struct DispatchBarrierData {
    /// @brief Global memory barriers to execute.
    std::vector<MemoryBarrierId> memoryBarriers;
    /// @brief Image barriers to execute.
    std::vector<ImageBarrierId> imageBarriers;
    /// @brief Tensor barriers to execute.
    std::vector<TensorBarrierId> tensorBarriers;
    /// @brief Buffer barriers to execute.
    std::vector<BufferBarrierId> bufferBarriers;
};

/// \brief Command that marks resources at a profiling boundary
struct MarkBoundaryData {
    /// @brief Buffers included in the boundary marker.
    std::vector<BufferId> buffers;
    /// @brief Images included in the boundary marker.
    std::vector<ImageId> images;
    /// @brief Tensors included in the boundary marker.
    std::vector<TensorId> tensors;
};

} // namespace mlsdk::scenariorunner
