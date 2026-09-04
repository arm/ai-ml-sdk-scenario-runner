/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "iscenario_builder.hpp"
#include "scenario_builder.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

namespace mlsdk::scenariorunner {
namespace {

template <typename Id> void bindResourceId(py::module_ &m, const char *name) {
    py::class_<Id>(m, name)
        .def_property_readonly("value", &Id::value)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__hash__", [](Id id) { return std::hash<Id>{}(id); })
        .def("__repr__", [name](Id id) { return std::string{name} + "(" + std::to_string(id.value()) + ")"; });
}

void bindResourceIds(py::module_ &m) {
    bindResourceId<ShaderId>(m, "ShaderId");
    bindResourceId<RawDataId>(m, "RawDataId");
    bindResourceId<DataGraphId>(m, "DataGraphId");
    bindResourceId<GraphConstantResourceId>(m, "GraphConstantResourceId");
    bindResourceId<MemoryGroupId>(m, "MemoryGroupId");
    bindResourceId<ImageBarrierId>(m, "ImageBarrierId");
    bindResourceId<BufferBarrierId>(m, "BufferBarrierId");
    bindResourceId<TensorBarrierId>(m, "TensorBarrierId");
    bindResourceId<MemoryBarrierId>(m, "MemoryBarrierId");
}

MemoryResourceId memoryResourceFromPython(const py::object &resource) {
    if (py::isinstance<BufferId>(resource)) {
        return resource.cast<BufferId>();
    }
    if (py::isinstance<ImageId>(resource)) {
        return resource.cast<ImageId>();
    }
    if (py::isinstance<TensorId>(resource)) {
        return resource.cast<TensorId>();
    }
    throw py::type_error("Expected a BufferId, ImageId, or TensorId");
}

py::object memoryResourceToPython(const MemoryResourceId &resource) {
    return std::visit([](auto id) { return py::cast(id); }, resource);
}

void bindEnums(py::module_ &m) {
    py::enum_<vk::Format>(m, "Format")
        .value("Undefined", vk::Format::eUndefined)
        .value("R8BoolArm", vk::Format::eR8BoolARM)
        .value("R8Uint", vk::Format::eR8Uint)
        .value("R8Sint", vk::Format::eR8Sint)
        .value("R8Unorm", vk::Format::eR8Unorm)
        .value("R8Snorm", vk::Format::eR8Snorm)
        .value("R16Uint", vk::Format::eR16Uint)
        .value("R16Sint", vk::Format::eR16Sint)
        .value("R16Unorm", vk::Format::eR16Unorm)
        .value("R16Snorm", vk::Format::eR16Snorm)
        .value("R16Sfloat", vk::Format::eR16Sfloat)
        .value("R32Uint", vk::Format::eR32Uint)
        .value("R32Sint", vk::Format::eR32Sint)
        .value("R32Sfloat", vk::Format::eR32Sfloat)
        .value("R8G8Uint", vk::Format::eR8G8Uint)
        .value("R8G8Sint", vk::Format::eR8G8Sint)
        .value("R8G8Unorm", vk::Format::eR8G8Unorm)
        .value("R8G8Snorm", vk::Format::eR8G8Snorm)
        .value("R16G16Uint", vk::Format::eR16G16Uint)
        .value("R16G16Sint", vk::Format::eR16G16Sint)
        .value("R16G16Unorm", vk::Format::eR16G16Unorm)
        .value("R16G16Snorm", vk::Format::eR16G16Snorm)
        .value("R16G16Sfloat", vk::Format::eR16G16Sfloat)
        .value("R32G32Uint", vk::Format::eR32G32Uint)
        .value("R32G32Sint", vk::Format::eR32G32Sint)
        .value("R32G32Sfloat", vk::Format::eR32G32Sfloat)
        .value("R8G8B8Uint", vk::Format::eR8G8B8Uint)
        .value("R8G8B8Sint", vk::Format::eR8G8B8Sint)
        .value("R8G8B8Unorm", vk::Format::eR8G8B8Unorm)
        .value("R8G8B8Snorm", vk::Format::eR8G8B8Snorm)
        .value("R8G8B8Srgb", vk::Format::eR8G8B8Srgb)
        .value("B8G8R8Unorm", vk::Format::eB8G8R8Unorm)
        .value("B8G8R8Srgb", vk::Format::eB8G8R8Srgb)
        .value("R8G8B8A8Uint", vk::Format::eR8G8B8A8Uint)
        .value("R8G8B8A8Sint", vk::Format::eR8G8B8A8Sint)
        .value("R8G8B8A8Unorm", vk::Format::eR8G8B8A8Unorm)
        .value("R8G8B8A8Snorm", vk::Format::eR8G8B8A8Snorm)
        .value("R8G8B8A8Srgb", vk::Format::eR8G8B8A8Srgb)
        .value("B8G8R8A8Unorm", vk::Format::eB8G8R8A8Unorm)
        .value("B8G8R8A8Srgb", vk::Format::eB8G8R8A8Srgb)
        .value("R16G16B16A16Uint", vk::Format::eR16G16B16A16Uint)
        .value("R16G16B16A16Sint", vk::Format::eR16G16B16A16Sint)
        .value("R16G16B16A16Unorm", vk::Format::eR16G16B16A16Unorm)
        .value("R16G16B16A16Snorm", vk::Format::eR16G16B16A16Snorm)
        .value("R16G16B16A16Sfloat", vk::Format::eR16G16B16A16Sfloat)
        .value("R32G32B32A32Uint", vk::Format::eR32G32B32A32Uint)
        .value("R32G32B32A32Sint", vk::Format::eR32G32B32A32Sint)
        .value("R32G32B32A32Sfloat", vk::Format::eR32G32B32A32Sfloat)
        .value("R64Sint", vk::Format::eR64Sint)
        .value("B10G11R11UfloatPack32", vk::Format::eB10G11R11UfloatPack32)
        .value("D32Sfloat", vk::Format::eD32Sfloat)
        .value("D24UnormS8Uint", vk::Format::eD24UnormS8Uint)
        .value("D32SfloatS8Uint", vk::Format::eD32SfloatS8Uint)
        .value("R16Bfloat16Arm", vk::Format::eR16SfloatFpencodingBfloat16ARM)
        .value("R8Float8E4M3Arm", vk::Format::eR8SfloatFpencodingFloat8E4M3ARM)
        .value("R8Float8E5M2Arm", vk::Format::eR8SfloatFpencodingFloat8E5M2ARM);

    py::enum_<vk::DescriptorType>(m, "DescriptorType")
        .value("StorageBuffer", vk::DescriptorType::eStorageBuffer)
        .value("StorageImage", vk::DescriptorType::eStorageImage)
        .value("CombinedImageSampler", vk::DescriptorType::eCombinedImageSampler)
        .value("TensorArm", vk::DescriptorType::eTensorARM);

    py::enum_<FilterMode>(m, "FilterMode")
        .value("Linear", FilterMode::Linear)
        .value("Nearest", FilterMode::Nearest)
        .value("Unknown", FilterMode::Unknown);
    py::enum_<AddressMode>(m, "AddressMode")
        .value("ClampBorder", AddressMode::ClampBorder)
        .value("ClampEdge", AddressMode::ClampEdge)
        .value("Repeat", AddressMode::Repeat)
        .value("MirroredRepeat", AddressMode::MirroredRepeat)
        .value("Unknown", AddressMode::Unknown);
    py::enum_<BorderColor>(m, "BorderColor")
        .value("FloatTransparentBlack", BorderColor::FloatTransparentBlack)
        .value("FloatOpaqueBlack", BorderColor::FloatOpaqueBlack)
        .value("FloatOpaqueWhite", BorderColor::FloatOpaqueWhite)
        .value("IntTransparentBlack", BorderColor::IntTransparentBlack)
        .value("IntOpaqueBlack", BorderColor::IntOpaqueBlack)
        .value("IntOpaqueWhite", BorderColor::IntOpaqueWhite)
        .value("FloatCustomExt", BorderColor::FloatCustomEXT)
        .value("IntCustomExt", BorderColor::IntCustomEXT)
        .value("Unknown", BorderColor::Unknown);
    py::enum_<Tiling>(m, "Tiling")
        .value("Optimal", Tiling::Optimal)
        .value("Linear", Tiling::Linear)
        .value("Unknown", Tiling::Unknown);
    py::enum_<MemoryAccess>(m, "MemoryAccess")
        .value("ComputeShaderWrite", MemoryAccess::ComputeShaderWrite)
        .value("MemoryWrite", MemoryAccess::MemoryWrite)
        .value("MemoryRead", MemoryAccess::MemoryRead)
        .value("GraphWrite", MemoryAccess::GraphWrite)
        .value("GraphRead", MemoryAccess::GraphRead)
        .value("ComputeShaderRead", MemoryAccess::ComputeShaderRead)
        .value("Unknown", MemoryAccess::Unknown);
    py::enum_<PipelineStage>(m, "PipelineStage")
        .value("Graph", PipelineStage::Graph)
        .value("Compute", PipelineStage::Compute)
        .value("Graphics", PipelineStage::Graphics)
        .value("All", PipelineStage::All)
        .value("Unknown", PipelineStage::Unknown);
    py::enum_<ImageLayout>(m, "ImageLayout")
        .value("General", ImageLayout::General)
        .value("TensorAliasing", ImageLayout::TensorAliasing)
        .value("Undefined", ImageLayout::Undefined)
        .value("Unknown", ImageLayout::Unknown);
    py::enum_<ShaderType>(m, "ShaderType")
        .value("Unknown", ShaderType::Unknown)
        .value("SpirV", ShaderType::SPIR_V)
        .value("Glsl", ShaderType::GLSL)
        .value("Hlsl", ShaderType::HLSL);
    py::enum_<ShaderStage>(m, "ShaderStage")
        .value("Unknown", ShaderStage::Unknown)
        .value("Compute", ShaderStage::Compute)
        .value("Vertex", ShaderStage::Vertex)
        .value("Fragment", ShaderStage::Fragment);
    py::enum_<OpticalFlowGridSize>(m, "OpticalFlowGridSize")
        .value("Invalid", OpticalFlowGridSize::Invalid)
        .value("OneByOne", OpticalFlowGridSize::e1x1)
        .value("TwoByTwo", OpticalFlowGridSize::e2x2)
        .value("FourByFour", OpticalFlowGridSize::e4x4)
        .value("EightByEight", OpticalFlowGridSize::e8x8);
    py::enum_<OpticalFlowPerformanceLevel>(m, "OpticalFlowPerformanceLevel")
        .value("Invalid", OpticalFlowPerformanceLevel::Invalid)
        .value("Unknown", OpticalFlowPerformanceLevel::Unknown)
        .value("Slow", OpticalFlowPerformanceLevel::Slow)
        .value("Medium", OpticalFlowPerformanceLevel::Medium)
        .value("Fast", OpticalFlowPerformanceLevel::Fast);
}

void bindResourceInfo(py::module_ &m) {
    py::class_<BufferInfo>(m, "BufferInfo")
        .def(py::init<>())
        .def_readwrite("debug_name", &BufferInfo::debugName)
        .def_readwrite("size", &BufferInfo::size)
        .def_readwrite("memory_offset", &BufferInfo::memoryOffset);

    py::class_<RawDataInfo>(m, "RawDataInfo")
        .def(py::init<>())
        .def_readwrite("debug_name", &RawDataInfo::debugName)
        .def_readwrite("src", &RawDataInfo::src);

    py::class_<TensorInfo>(m, "TensorInfo")
        .def(py::init<>())
        .def_readwrite("debug_name", &TensorInfo::debugName)
        .def_readwrite("shape", &TensorInfo::shape)
        .def_readwrite("format", &TensorInfo::format)
        .def_readwrite("sparsity_dimension", &TensorInfo::sparsityDimension)
        .def_readwrite("descriptor_buffer_capture_replay", &TensorInfo::descriptorBufferCaptureReplay)
        .def_readwrite("tiling", &TensorInfo::tiling)
        .def_readwrite("memory_offset", &TensorInfo::memoryOffset);

    py::class_<SamplerSettings>(m, "SamplerSettings")
        .def(py::init<>())
        .def_readwrite("min_filter", &SamplerSettings::minFilter)
        .def_readwrite("mag_filter", &SamplerSettings::magFilter)
        .def_readwrite("mip_filter", &SamplerSettings::mipFilter)
        .def_readwrite("address_mode_u", &SamplerSettings::addressModeU)
        .def_readwrite("address_mode_v", &SamplerSettings::addressModeV)
        .def_readwrite("address_mode_w", &SamplerSettings::addressModeW)
        .def_readwrite("border_color", &SamplerSettings::borderColor)
        .def_readwrite("custom_border_color", &SamplerSettings::customBorderColor);

    py::class_<ImageInfo>(m, "ImageInfo")
        .def(py::init<>())
        .def_readwrite("debug_name", &ImageInfo::debugName)
        .def_readwrite("shape", &ImageInfo::shape)
        .def_readwrite("format", &ImageInfo::format)
        .def_readwrite("target_format", &ImageInfo::targetFormat)
        .def_readwrite("is_input", &ImageInfo::isInput)
        .def_readwrite("sampler_settings", &ImageInfo::samplerSettings)
        .def_readwrite("mips", &ImageInfo::mips)
        .def_readwrite("is_sampled", &ImageInfo::isSampled)
        .def_readwrite("is_storage", &ImageInfo::isStorage)
        .def_readwrite("is_color_attachment", &ImageInfo::isColorAttachment)
        .def_readwrite("tiling", &ImageInfo::tiling)
        .def_readwrite("memory_offset", &ImageInfo::memoryOffset);

    py::class_<SpecializationConstant>(m, "SpecializationConstant")
        .def_static("from_int32",
                    [](int id, int32_t value) {
                        SpecializationConstant result{};
                        result.id = id;
                        result.value.i = value;
                        return result;
                    })
        .def_static("from_uint32",
                    [](int id, uint32_t value) {
                        SpecializationConstant result{};
                        result.id = id;
                        result.value.ui = value;
                        return result;
                    })
        .def_static("from_float32",
                    [](int id, float value) {
                        SpecializationConstant result{};
                        result.id = id;
                        result.value.f = value;
                        return result;
                    })
        .def_readwrite("id", &SpecializationConstant::id);

    py::class_<SpecializationConstantMap>(m, "SpecializationConstantMap")
        .def(py::init<>())
        .def_readwrite("specialization_constants", &SpecializationConstantMap::specializationConstants)
        .def_readwrite("shader_target", &SpecializationConstantMap::shaderTarget);

    py::class_<DataGraphInfo>(m, "DataGraphInfo")
        .def(py::init<>())
        .def_readwrite("debug_name", &DataGraphInfo::debugName)
        .def_readwrite("src", &DataGraphInfo::src)
        .def_readwrite("push_constants_size", &DataGraphInfo::pushConstantsSize)
        .def_readwrite("specialization_constant_maps", &DataGraphInfo::specializationConstantMaps);

    py::class_<ShaderInfo>(m, "ShaderInfo")
        .def(py::init<>())
        .def_readwrite("debug_name", &ShaderInfo::debugName)
        .def_readwrite("entry", &ShaderInfo::entry)
        .def_readwrite("push_constants_size", &ShaderInfo::pushConstantsSize)
        .def_readwrite("specialization_constants", &ShaderInfo::specializationConstants)
        .def_readwrite("src", &ShaderInfo::src)
        .def_readwrite("shader_type", &ShaderInfo::shaderType)
        .def_readwrite("stage", &ShaderInfo::stage)
        .def_readwrite("build_options", &ShaderInfo::buildOpts)
        .def_readwrite("include_dirs", &ShaderInfo::includeDirs);

    py::class_<GraphConstantInfo>(m, "GraphConstantInfo")
        .def(py::init<>())
        .def_readwrite("format", &GraphConstantInfo::format)
        .def_readwrite("dims", &GraphConstantInfo::dims)
        .def_readwrite("data", &GraphConstantInfo::data)
        .def_readwrite("debug_name", &GraphConstantInfo::debugName);
}

void bindBarriers(py::module_ &m) {
    py::class_<SubresourceRange>(m, "SubresourceRange")
        .def(py::init<>())
        .def_readwrite("base_mip_level", &SubresourceRange::baseMipLevel)
        .def_readwrite("level_count", &SubresourceRange::levelCount)
        .def_readwrite("base_array_layer", &SubresourceRange::baseArrayLayer)
        .def_readwrite("layer_count", &SubresourceRange::layerCount);

    py::class_<BaseBarrierInfo>(m, "BaseBarrierInfo")
        .def(py::init<>())
        .def_readwrite("debug_name", &BaseBarrierInfo::debugName)
        .def_readwrite("src_access", &BaseBarrierInfo::srcAccess)
        .def_readwrite("dst_access", &BaseBarrierInfo::dstAccess)
        .def_readwrite("src_stages", &BaseBarrierInfo::srcStages)
        .def_readwrite("dst_stages", &BaseBarrierInfo::dstStages);
    py::class_<ImageBarrierInfo, BaseBarrierInfo>(m, "ImageBarrierInfo")
        .def(py::init<>())
        .def_readwrite("image", &ImageBarrierInfo::image)
        .def_readwrite("old_layout", &ImageBarrierInfo::oldLayout)
        .def_readwrite("new_layout", &ImageBarrierInfo::newLayout)
        .def_readwrite("range", &ImageBarrierInfo::range);
    py::class_<BufferBarrierInfo, BaseBarrierInfo>(m, "BufferBarrierInfo")
        .def(py::init<>())
        .def_readwrite("buffer", &BufferBarrierInfo::buffer)
        .def_readwrite("offset", &BufferBarrierInfo::offset)
        .def_readwrite("size", &BufferBarrierInfo::size);
    py::class_<TensorBarrierInfo, BaseBarrierInfo>(m, "TensorBarrierInfo")
        .def(py::init<>())
        .def_readwrite("tensor", &TensorBarrierInfo::tensor);
    py::class_<MemoryBarrierInfo, BaseBarrierInfo>(m, "MemoryBarrierInfo").def(py::init<>());
}

void bindCommands(py::module_ &m) {
    py::class_<vk::Extent2D>(m, "Extent2D")
        .def(py::init<uint32_t, uint32_t>())
        .def_readwrite("width", &vk::Extent2D::width)
        .def_readwrite("height", &vk::Extent2D::height);

    py::class_<TypedBinding>(m, "TypedBinding")
        .def(py::init([](uint32_t set, uint32_t binding, const py::object &resource, vk::DescriptorType descriptorType,
                         std::optional<uint32_t> lod) {
                 return TypedBinding{set, binding, memoryResourceFromPython(resource), lod, descriptorType};
             }),
             py::arg("set"), py::arg("binding"), py::arg("resource"), py::arg("descriptor_type"), py::kw_only(),
             py::arg("lod") = std::nullopt)
        .def_readwrite("set", &TypedBinding::set)
        .def_readwrite("binding", &TypedBinding::id)
        .def_property(
            "resource", [](const TypedBinding &binding) { return memoryResourceToPython(binding.resource); },
            [](TypedBinding &binding, const py::object &resource) {
                binding.resource = memoryResourceFromPython(resource);
            })
        .def_readwrite("lod", &TypedBinding::lod)
        .def_readwrite("descriptor_type", &TypedBinding::vkDescriptorType);

    py::class_<ComputeDispatch>(m, "ComputeDispatch")
        .def(py::init<>())
        .def_readwrite("group_count_x", &ComputeDispatch::gwcx)
        .def_readwrite("group_count_y", &ComputeDispatch::gwcy)
        .def_readwrite("group_count_z", &ComputeDispatch::gwcz)
        .def_readwrite("profile_name", &ComputeDispatch::profileName);

    py::class_<DispatchComputeData>(m, "DispatchComputeData")
        .def(py::init<ShaderId>(), py::arg("shader"))
        .def_readwrite("debug_name", &DispatchComputeData::debugName)
        .def_readwrite("bindings", &DispatchComputeData::bindings)
        .def_readwrite("compute_dispatch", &DispatchComputeData::computeDispatch)
        .def_readwrite("shader", &DispatchComputeData::shader)
        .def_readwrite("implicit_barrier", &DispatchComputeData::implicitBarrier)
        .def_readwrite("push_data", &DispatchComputeData::pushData);

    using FragmentAttachment = DispatchFragmentData::Attachment;
    py::class_<FragmentAttachment>(m, "FragmentAttachment")
        .def(py::init([](ImageId resource, std::optional<uint32_t> lod) { return FragmentAttachment{resource, lod}; }),
             py::arg("resource"), py::kw_only(), py::arg("lod") = std::nullopt)
        .def_readwrite("resource", &FragmentAttachment::resource)
        .def_readwrite("lod", &FragmentAttachment::lod);
    py::class_<DispatchFragmentData>(m, "DispatchFragmentData")
        .def(py::init<ShaderId, ShaderId>(), py::arg("vertex_shader"), py::arg("fragment_shader"))
        .def_readwrite("debug_name", &DispatchFragmentData::debugName)
        .def_readwrite("bindings", &DispatchFragmentData::bindings)
        .def_readwrite("vertex_shader", &DispatchFragmentData::vertexShader)
        .def_readwrite("fragment_shader", &DispatchFragmentData::fragmentShader)
        .def_readwrite("color_attachments", &DispatchFragmentData::colorAttachments)
        .def_readwrite("render_extent", &DispatchFragmentData::renderExtent)
        .def_readwrite("implicit_barrier", &DispatchFragmentData::implicitBarrier)
        .def_readwrite("push_data", &DispatchFragmentData::pushData);

    py::class_<ResolvedPushConstantMap>(m, "PushConstantMap")
        .def(py::init<RawDataId, std::string>(), py::arg("push_data"), py::arg("shader_target"))
        .def_readwrite("push_data", &ResolvedPushConstantMap::pushData)
        .def_readwrite("shader_target", &ResolvedPushConstantMap::shaderTarget);
    py::class_<ResolvedShaderSubstitution>(m, "ShaderSubstitution")
        .def(py::init<ShaderId, std::string>(), py::arg("shader"), py::arg("target"))
        .def_readwrite("shader", &ResolvedShaderSubstitution::shader)
        .def_readwrite("target", &ResolvedShaderSubstitution::target);
    py::class_<DispatchDataGraphData>(m, "DispatchDataGraphData")
        .def(py::init<DataGraphId>(), py::arg("data_graph"))
        .def_readwrite("data_graph", &DispatchDataGraphData::dataGraph)
        .def_readwrite("debug_name", &DispatchDataGraphData::debugName)
        .def_readwrite("bindings", &DispatchDataGraphData::bindings)
        .def_readwrite("push_constants", &DispatchDataGraphData::pushConstants)
        .def_readwrite("shader_substitutions", &DispatchDataGraphData::shaderSubstitutions)
        .def_readwrite("implicit_barrier", &DispatchDataGraphData::implicitBarrier);
    py::class_<DispatchSpirvGraphData>(m, "DispatchSpirvGraphData")
        .def(py::init<ShaderId>(), py::arg("graph_shader"))
        .def_readwrite("graph_shader", &DispatchSpirvGraphData::graphShader)
        .def_readwrite("debug_name", &DispatchSpirvGraphData::debugName)
        .def_readwrite("bindings", &DispatchSpirvGraphData::bindings)
        .def_readwrite("graph_constants", &DispatchSpirvGraphData::graphConstants)
        .def_readwrite("implicit_barrier", &DispatchSpirvGraphData::implicitBarrier);

    py::class_<DispatchOpticalFlowData>(m, "DispatchOpticalFlowData")
        .def(py::init<TypedBinding, TypedBinding, TypedBinding>(), py::arg("search"), py::arg("reference"),
             py::arg("output"))
        .def_readwrite("debug_name", &DispatchOpticalFlowData::debugName)
        .def_readwrite("search_image", &DispatchOpticalFlowData::searchImage)
        .def_readwrite("template_image", &DispatchOpticalFlowData::templateImage)
        .def_readwrite("output_image", &DispatchOpticalFlowData::outputImage)
        .def_readwrite("hint_motion_vectors", &DispatchOpticalFlowData::hintMotionVectors)
        .def_readwrite("output_cost", &DispatchOpticalFlowData::outputCost)
        .def_readwrite("width", &DispatchOpticalFlowData::width)
        .def_readwrite("height", &DispatchOpticalFlowData::height)
        .def_readwrite("performance_level", &DispatchOpticalFlowData::performanceLevel)
        .def_readwrite("execution_flags", &DispatchOpticalFlowData::executionFlags)
        .def_readwrite("grid_size", &DispatchOpticalFlowData::gridSize)
        .def_readwrite("mean_flow_l1_norm_hint", &DispatchOpticalFlowData::meanFlowL1NormHint)
        .def_readwrite("implicit_barrier", &DispatchOpticalFlowData::implicitBarrier);

    py::class_<DispatchBarrierData>(m, "DispatchBarrierData")
        .def(py::init<>())
        .def_readwrite("memory_barriers", &DispatchBarrierData::memoryBarriers)
        .def_readwrite("image_barriers", &DispatchBarrierData::imageBarriers)
        .def_readwrite("tensor_barriers", &DispatchBarrierData::tensorBarriers)
        .def_readwrite("buffer_barriers", &DispatchBarrierData::bufferBarriers);
    py::class_<MarkBoundaryData>(m, "MarkBoundaryData")
        .def(py::init<>())
        .def_readwrite("buffers", &MarkBoundaryData::buffers)
        .def_readwrite("images", &MarkBoundaryData::images)
        .def_readwrite("tensors", &MarkBoundaryData::tensors);
}

void bindBuilder(py::module_ &m) {
    py::class_<IScenarioBuilder>(m, "IScenarioBuilder")
        .def("add_buffer", &IScenarioBuilder::addBuffer)
        .def(
            "add_buffer",
            [](IScenarioBuilder &builder, uint32_t size, const std::string &debugName, uint64_t memoryOffset) {
                return builder.addBuffer(BufferInfo{debugName, size, memoryOffset});
            },
            py::arg("size"), py::kw_only(), py::arg("debug_name") = "", py::arg("memory_offset") = 0)
        .def("add_image", &IScenarioBuilder::addImage)
        .def(
            "add_image",
            [](IScenarioBuilder &builder, const std::vector<int64_t> &shape, vk::Format format,
               const std::string &debugName, bool isInput, bool isSampled, bool isStorage, bool isColorAttachment,
               uint32_t mips, std::optional<Tiling> tiling, uint64_t memoryOffset) {
                ImageInfo info{};
                info.debugName = debugName;
                info.shape = shape;
                info.format = format;
                info.targetFormat = format;
                info.isInput = isInput;
                info.isSampled = isSampled;
                info.isStorage = isStorage;
                info.isColorAttachment = isColorAttachment;
                info.mips = mips;
                info.tiling = tiling;
                info.memoryOffset = memoryOffset;
                return builder.addImage(info);
            },
            py::arg("shape"), py::arg("format"), py::kw_only(), py::arg("debug_name") = "", py::arg("is_input") = false,
            py::arg("is_sampled") = false, py::arg("is_storage") = false, py::arg("is_color_attachment") = false,
            py::arg("mips") = 1, py::arg("tiling") = py::none(), py::arg("memory_offset") = 0)
        .def("add_tensor", &IScenarioBuilder::addTensor)
        .def(
            "add_tensor",
            [](IScenarioBuilder &builder, const std::vector<int64_t> &shape, vk::Format format,
               const std::string &debugName, int64_t sparsityDimension, bool descriptorBufferCaptureReplay,
               Tiling tiling, uint64_t memoryOffset) {
                return builder.addTensor(TensorInfo{debugName, shape, format, sparsityDimension,
                                                    descriptorBufferCaptureReplay, tiling, memoryOffset});
            },
            py::arg("shape"), py::arg("format"), py::kw_only(), py::arg("debug_name") = "",
            py::arg("sparsity_dimension") = -1, py::arg("descriptor_buffer_capture_replay") = false,
            py::arg("tiling") = Tiling::Linear, py::arg("memory_offset") = 0)
        .def("add_shader", &IScenarioBuilder::addShader)
        .def("add_raw_data", &IScenarioBuilder::addRawData)
        .def("add_data_graph", &IScenarioBuilder::addDataGraph)
        .def("add_graph_constant", &IScenarioBuilder::addGraphConstant)
        .def("add_image_barrier", &IScenarioBuilder::addImageBarrier)
        .def("add_buffer_barrier", &IScenarioBuilder::addBufferBarrier)
        .def("add_tensor_barrier", &IScenarioBuilder::addTensorBarrier)
        .def("add_memory_barrier", &IScenarioBuilder::addMemoryBarrier)
        .def("create_memory_group", &IScenarioBuilder::createMemoryGroup)
        .def("add_resource_to_memory_group",
             [](IScenarioBuilder &builder, MemoryGroupId group, const py::object &resource) {
                 builder.addResourceToMemoryGroup(group, memoryResourceFromPython(resource));
             })
        .def("add_dispatch_compute", &IScenarioBuilder::addDispatchCompute)
        .def("add_dispatch_fragment", &IScenarioBuilder::addDispatchFragment)
        .def("add_dispatch_data_graph", &IScenarioBuilder::addDispatchDataGraph)
        .def("add_dispatch_spirv_graph", &IScenarioBuilder::addDispatchSpirvGraph)
        .def("add_dispatch_optical_flow", &IScenarioBuilder::addDispatchOpticalFlow)
        .def("add_dispatch_barrier", &IScenarioBuilder::addDispatchBarrier)
        .def("add_mark_boundary", &IScenarioBuilder::addMarkBoundary)
        .def("build", &IScenarioBuilder::build, py::kw_only(), py::arg("options") = ScenarioOptions{},
             py::call_guard<py::gil_scoped_release>());

    py::class_<ScenarioBuilder, IScenarioBuilder>(m, "ScenarioBuilder").def(py::init<>());
}

} // namespace
} // namespace mlsdk::scenariorunner

void pyInitIScenarioBuilder(py::module_ &m) {
    using namespace mlsdk::scenariorunner;

    bindResourceIds(m);
    bindEnums(m);
    bindResourceInfo(m);
    bindBarriers(m);
    bindCommands(m);
    bindBuilder(m);
}
