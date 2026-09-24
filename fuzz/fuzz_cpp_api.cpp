// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include "fuzzers.hpp"

#include "scenario_builder_impl.hpp"

#include "vgf-utils/numpy.hpp"
#include "vgf/encoder.hpp"
#include "vgf/vulkan_helpers.generated.hpp"

#include "spirv-tools/libspirv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace mlsdk::scenariorunner;

namespace {

class ByteCursor {
  public:
    ByteCursor(const uint8_t *data, size_t size) : _data{data}, _size{size} {}

    uint8_t next() { return _offset < _size ? _data[_offset++] : 0; }
    bool empty() const { return _offset >= _size; }

  private:
    const uint8_t *_data;
    size_t _size;
    size_t _offset{};
};

constexpr size_t kOperationRecordSize = 16;
constexpr size_t kMinimumStructuredInputSize = 1 + 2 * kOperationRecordSize;
constexpr size_t kStatsReportInterval = 1024;

enum class VgfResourceKind : uint8_t { Buffer, Image, Tensor };

class FuzzStats {
  public:
    ~FuzzStats() {
        if (_inputs != 0) {
            report("final");
        }
    }

    void finishInput() {
        ++_inputs;
        if (_inputs % kStatsReportInterval == 0) {
            report("periodic");
        }
    }

    void operationRecord() { ++_operationRecords; }
    void operationRejected() { ++_operationRejected; }
    void vgfAdded(VgfResourceKind kind) { ++_vgfsByKind[static_cast<size_t>(kind)]; }
    void dataGraphAdded() { ++_dataGraphs; }
    void buildAttempt() { ++_buildAttempts; }
    void buildSucceeded() { ++_buildSucceeded; }
    void buildRejected(std::string_view message) {
        ++_buildRejected;
        if (_firstBuildRejection.empty()) {
            _firstBuildRejection.assign(message.substr(0, 256));
            std::replace(_firstBuildRejection.begin(), _firstBuildRejection.end(), '\n', ' ');
            std::replace(_firstBuildRejection.begin(), _firstBuildRejection.end(), '\r', ' ');
        }
    }
    void runAttempt() { ++_runAttempts; }
    void runCompleted(uint32_t iterations) {
        ++_runCompleted;
        _runIterations += iterations;
    }
    void endToEndCompleted() { ++_endToEndCompleted; }
    void reset() {
        _inputs = 0;
        _operationRecords = 0;
        _operationRejected = 0;
        _vgfsByKind = {};
        _dataGraphs = 0;
        _buildAttempts = 0;
        _buildSucceeded = 0;
        _buildRejected = 0;
        _runAttempts = 0;
        _runCompleted = 0;
        _runIterations = 0;
        _endToEndCompleted = 0;
        _firstBuildRejection.clear();
    }

  private:
    void report(const char *kind) const {
        std::fprintf(stderr,
                     "SCENARIO_FUZZER_STATS kind=%s inputs=%zu records=%zu record_rejections=%zu "
                     "vgf_buffers=%zu vgf_images=%zu vgf_tensors=%zu "
                     "data_graphs=%zu "
                     "build_attempts=%zu build_ok=%zu build_rejected=%zu run_attempts=%zu run_ok=%zu "
                     "run_iterations=%zu e2e_ok=%zu\n",
                     kind, _inputs, _operationRecords, _operationRejected,
                     _vgfsByKind[static_cast<size_t>(VgfResourceKind::Buffer)],
                     _vgfsByKind[static_cast<size_t>(VgfResourceKind::Image)],
                     _vgfsByKind[static_cast<size_t>(VgfResourceKind::Tensor)], _dataGraphs, _buildAttempts,
                     _buildSucceeded, _buildRejected, _runAttempts, _runCompleted, _runIterations, _endToEndCompleted);
        if (!_firstBuildRejection.empty()) {
            std::fprintf(stderr, "SCENARIO_FUZZER_FIRST_BUILD_REJECTION message=%s\n", _firstBuildRejection.c_str());
        }
    }

    size_t _inputs{};
    size_t _operationRecords{};
    size_t _operationRejected{};
    std::array<size_t, 3> _vgfsByKind{};
    size_t _dataGraphs{};
    size_t _buildAttempts{};
    size_t _buildSucceeded{};
    size_t _buildRejected{};
    size_t _runAttempts{};
    size_t _runCompleted{};
    size_t _runIterations{};
    size_t _endToEndCompleted{};
    std::string _firstBuildRejection;
};

FuzzStats fuzzStats;

class InputStatsGuard {
  public:
    ~InputStatsGuard() { fuzzStats.finishInput(); }
};

std::string debugName(uint8_t value) { return "fuzz_" + std::to_string(value); }

bool startsWith(std::string_view message, std::string_view prefix) {
    return message.size() >= prefix.size() && message.compare(0, prefix.size(), prefix) == 0;
}
class FuzzFileDirectory {
  public:
    FuzzFileDirectory() {
        const auto tempPath = std::filesystem::temp_directory_path();
        std::random_device random;
        for (size_t attempt = 0; attempt < 100; ++attempt) {
            _path = tempPath / ("scenario_runner_fuzzer_" + std::to_string(random()) + "_" + std::to_string(random()));
            if (std::filesystem::create_directory(_path)) {
                return;
            }
        }
        throw std::runtime_error("Could not create a unique fuzzer directory.");
    }

    ~FuzzFileDirectory() {
        std::error_code error;
        std::filesystem::remove_all(_path, error);
    }

    FuzzFileDirectory(const FuzzFileDirectory &) = delete;
    FuzzFileDirectory &operator=(const FuzzFileDirectory &) = delete;

    std::filesystem::path vgfPath(size_t index) const { return _path / ("recipe_" + std::to_string(index) + ".vgf"); }
    std::filesystem::path dataGraphPath(size_t index) const {
        return _path / ("data_graph_" + std::to_string(index) + ".spv");
    }
    std::filesystem::path rawDataPath(size_t index) const {
        return _path / ("raw_data_" + std::to_string(index) + ".npy");
    }

  private:
    std::filesystem::path _path;
};

const FuzzFileDirectory &fuzzFileDirectory() {
    static const FuzzFileDirectory directory;
    return directory;
}

std::string_view vgfModuleName(VgfResourceKind kind) {
    switch (kind) {
    case VgfResourceKind::Buffer:
        return "variable_copy";
    case VgfResourceKind::Image:
        return "variable_copy_image";
    case VgfResourceKind::Tensor:
        return "variable_copy_tensor";
    }
    std::abort();
}

VkDescriptorType vgfDescriptorType(VgfResourceKind kind) {
    switch (kind) {
    case VgfResourceKind::Buffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case VgfResourceKind::Image:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case VgfResourceKind::Tensor:
        return VK_DESCRIPTOR_TYPE_TENSOR_ARM;
    }
    std::abort();
}

std::string writeVgfRecipe(VgfResourceKind kind, vk::Format format, const std::vector<int64_t> &inputShape,
                           const std::vector<int64_t> &outputShape, const std::array<uint32_t, 3> &dispatchShape,
                           size_t index) {
    auto encoder = mlsdk::vgflib::CreateEncoder(123);
    const auto moduleName = vgfModuleName(kind);
    const auto module = encoder->AddModule(mlsdk::vgflib::ModuleType::COMPUTE, std::string(moduleName), "main");
    const auto descriptorType = mlsdk::vgflib::ToDescriptorType(vgfDescriptorType(kind));
    const auto formatType = mlsdk::vgflib::ToFormatType(static_cast<VkFormat>(format));
    const auto input = encoder->AddInputResource(descriptorType, formatType, inputShape, {});
    const auto output = encoder->AddOutputResource(descriptorType, formatType, outputShape, {});
    const auto inputBinding = encoder->AddBindingSlot(0, input);
    const auto outputBinding = encoder->AddBindingSlot(1, output);
    const std::vector<mlsdk::vgflib::DescriptorSetInfoRef> descriptors{
        encoder->AddDescriptorSetInfo({inputBinding, outputBinding})};
    const std::vector<mlsdk::vgflib::BindingSlotRef> inputs{inputBinding};
    const std::vector<mlsdk::vgflib::BindingSlotRef> outputs{outputBinding};
    const std::vector<mlsdk::vgflib::GraphConstantBindingRef> graphConstantBindings;
    const std::vector<mlsdk::vgflib::PushConstRangeRef> pushConstRanges;
    encoder->AddModelSequenceInputsOutputs(inputs, {"input"}, outputs, {"output"});
    encoder->AddSegmentInfo(module, std::string(moduleName) + "_segment", descriptors, inputs, outputs,
                            graphConstantBindings, dispatchShape, pushConstRanges);
    encoder->Finish();

    const auto path = fuzzFileDirectory().vgfPath(index);
    std::ofstream outputFile(path, std::ios::binary | std::ios::trunc);
    if (!outputFile || !encoder->WriteTo(outputFile)) {
        throw std::runtime_error("Could not write VGF recipe.");
    }
    outputFile.close();
    if (!outputFile) {
        throw std::runtime_error("Could not close VGF recipe.");
    }
    return path.string();
}

void replaceAll(std::string &text, std::string_view token, int64_t value) {
    const auto replacement = std::to_string(value);
    size_t position{};
    while ((position = text.find(token, position)) != std::string::npos) {
        text.replace(position, token.size(), replacement);
        position += replacement.size();
    }
}

std::string writeDataGraphShader(const std::vector<int64_t> &inputShape, const std::vector<int64_t> &outputShape,
                                 size_t index) {
    const std::filesystem::path templatePath = SCENARIO_FUZZER_SOURCE_DIR "/constant_output_graph.spvasm";
    std::ifstream templateFile(templatePath);
    if (!templateFile) {
        throw std::runtime_error("Could not read data-graph SPIR-V template.");
    }
    std::string assembly{std::istreambuf_iterator<char>{templateFile}, std::istreambuf_iterator<char>{}};
    for (size_t dimension = 0; dimension < 4; ++dimension) {
        replaceAll(assembly, "INPUT_DIM_" + std::to_string(dimension), inputShape[dimension]);
        replaceAll(assembly, "OUTPUT_DIM_" + std::to_string(dimension), outputShape[dimension]);
    }

    std::string diagnostic;
    spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
    tools.SetMessageConsumer(
        [&](spv_message_level_t, const char *, const spv_position_t &, const char *message) { diagnostic = message; });
    std::vector<uint32_t> binary;
    if (!tools.Assemble(assembly, &binary) || !tools.Validate(binary.data(), binary.size())) {
        throw std::runtime_error("Could not generate data-graph SPIR-V: " + diagnostic);
    }

    const auto path = fuzzFileDirectory().dataGraphPath(index);
    std::ofstream outputFile(path, std::ios::binary | std::ios::trunc);
    outputFile.write(reinterpret_cast<const char *>(binary.data()),
                     static_cast<std::streamsize>(binary.size() * sizeof(uint32_t)));
    outputFile.close();
    if (!outputFile) {
        throw std::runtime_error("Could not write data-graph SPIR-V.");
    }
    return path.string();
}

std::string writeRawData(uint8_t value, size_t index) {
    const auto path = fuzzFileDirectory().rawDataPath(index);
    const auto data = static_cast<char>(value);
    const mlsdk::vgfutils::numpy::DataPtr dataPtr{&data, {1}, {'u', sizeof(value)}};
    mlsdk::vgfutils::numpy::write(path.string(), dataPtr);
    return path.string();
}

enum class BuilderOperation {
    AddBuffer = 0,
    AddTensor,
    AddImage,
    AddResourceToMemoryGroup,
    AddShader,
    AddRawData,
    AddVgf,
    AddGraphConstant,
    AddImageBarrier,
    AddBufferBarrier,
    AddTensorBarrier,
    AddMemoryBarrier,
    AddComputeDispatch,
    AddFragmentDispatch,
    AddVgfDispatch,
    AddDataGraphDispatch,
    AddPipelineBarrier,
    AddOpticalFlowDispatch,
    AddFrameBoundary,
    Count,
};

struct FuzzResources {
    struct BufferRecord {
        BufferId id;
        BufferInfo info;
    };
    struct TensorRecord {
        TensorId id;
        TensorInfo info;
    };
    struct ImageRecord {
        ImageId id;
        ImageInfo info;
    };
    struct VgfRecord {
        VgfId id;
        VgfResourceKind kind;
        MemoryResourceId input;
        MemoryResourceId output;
        std::vector<int64_t> inputShape;
        std::vector<int64_t> outputShape;
        vk::Format format;
        ShaderId shader;
    };
    struct ComputeShaderRecord {
        ShaderId id;
        BufferId input;
        BufferId output;
        uint32_t inputSize;
        uint32_t outputSize;
    };
    struct GraphShaderRecord {
        ShaderId id;
        TensorId input;
        TensorId output;
        std::vector<int64_t> inputShape;
        std::vector<int64_t> outputShape;
        vk::Format format;
    };

    std::vector<BufferId> buffers;
    std::vector<BufferRecord> bufferRecords;
    std::vector<ImageId> images;
    std::vector<ImageRecord> imageRecords;
    std::vector<TensorId> tensors;
    std::vector<TensorRecord> tensorRecords;
    std::vector<MemoryGroupId> memoryGroups;
    std::vector<ComputeShaderRecord> computeShaderRecords;
    std::vector<ShaderId> vertexShaders;
    std::vector<ShaderId> fragmentShaders;
    std::vector<GraphShaderRecord> graphShaderRecords;
    std::vector<ImageId> sampledImages;
    std::vector<ImageId> colorAttachments;
    std::vector<ImageId> storageImages;
    std::vector<RawDataId> rawData;
    std::vector<VgfRecord> vgfs;
    std::array<std::optional<ShaderId>, 3> vgfShaders;
    std::vector<GraphConstantResourceId> graphConstants;
    std::vector<ImageBarrierId> imageBarriers;
    std::vector<BufferBarrierId> bufferBarriers;
    std::vector<TensorBarrierId> tensorBarriers;
    std::vector<MemoryBarrierId> memoryBarriers;
};

vk::DescriptorType vgfVkDescriptorType(VgfResourceKind kind) {
    switch (kind) {
    case VgfResourceKind::Buffer:
        return vk::DescriptorType::eStorageBuffer;
    case VgfResourceKind::Image:
        return vk::DescriptorType::eStorageImage;
    case VgfResourceKind::Tensor:
        return vk::DescriptorType::eTensorARM;
    }
    std::abort();
}

std::vector<int64_t> vgfShape(VgfResourceKind kind, const std::vector<int64_t> &shape) {
    if (kind == VgfResourceKind::Image) {
        // Scenario images use [N, W, H, depth], while VGF images use NHWC.
        return {1, shape[2], shape[1], 4};
    }
    return shape;
}

std::array<uint32_t, 3> vgfDispatchShape(VgfResourceKind kind, const std::vector<int64_t> &outputShape) {
    switch (kind) {
    case VgfResourceKind::Buffer:
        return {static_cast<uint32_t>(outputShape[0]), 1, 1};
    case VgfResourceKind::Image:
        return {static_cast<uint32_t>(outputShape[1]), static_cast<uint32_t>(outputShape[2]), 1};
    case VgfResourceKind::Tensor:
        return {static_cast<uint32_t>(outputShape[1]), static_cast<uint32_t>(outputShape[2]),
                static_cast<uint32_t>(outputShape[3])};
    }
    std::abort();
}

ShaderId getVgfShader(ScenarioBuilderImpl &builder, FuzzResources &resources, VgfResourceKind kind) {
    auto &shader = resources.vgfShaders[static_cast<size_t>(kind)];
    if (shader.has_value()) {
        return *shader;
    }

    ShaderInfo info{};
    info.debugName = std::string(vgfModuleName(kind));
    info.entry = "main";
    info.shaderType = ShaderType::GLSL;
    info.stage = ShaderStage::Compute;
    switch (kind) {
    case VgfResourceKind::Buffer:
        info.src = SCENARIO_FUZZER_SOURCE_DIR "/variable_copy.comp";
        break;
    case VgfResourceKind::Image:
        info.src = SCENARIO_FUZZER_SOURCE_DIR "/variable_copy_image.comp";
        break;
    case VgfResourceKind::Tensor:
        info.src = SCENARIO_FUZZER_SOURCE_DIR "/variable_copy_tensor.comp";
        break;
    }
    shader = builder.addShader(info);
    return *shader;
}

VgfId addVgf(ScenarioBuilderImpl &builder, FuzzResources &resources, VgfResourceKind kind, MemoryResourceId input,
             MemoryResourceId output, std::vector<int64_t> inputShape, std::vector<int64_t> outputShape,
             vk::Format format, std::string debugName) {
    VgfInfo info{};
    info.debugName = std::move(debugName);
    info.src = writeVgfRecipe(kind, format, vgfShape(kind, inputShape), vgfShape(kind, outputShape),
                              vgfDispatchShape(kind, outputShape), resources.vgfs.size());
    const auto id = builder.addVgf(info);
    const auto shader = getVgfShader(builder, resources, kind);
    resources.vgfs.push_back(
        {id, kind, std::move(input), std::move(output), std::move(inputShape), std::move(outputShape), format, shader});
    fuzzStats.vgfAdded(kind);
    return id;
}

ShaderId addDataGraphShader(ScenarioBuilderImpl &builder, FuzzResources &resources,
                            const FuzzResources::TensorRecord &input, const FuzzResources::TensorRecord &output,
                            std::string debugName) {
    ShaderInfo info{};
    info.debugName = std::move(debugName);
    info.entry = "main";
    info.src = writeDataGraphShader(input.info.shape, output.info.shape, resources.graphShaderRecords.size());
    info.shaderType = ShaderType::SPIR_V;
    info.stage = ShaderStage::Compute;
    const auto id = builder.addShader(info);
    resources.graphShaderRecords.push_back(
        {id, input.id, output.id, input.info.shape, output.info.shape, input.info.format});
    fuzzStats.dataGraphAdded();
    return id;
}

void seedComputeScenario(ScenarioBuilderImpl &builder, FuzzResources &resources) {
    auto &buffers = resources.buffers;
    // Seed every input with a small, valid compute scenario. This gives
    // build() a realistic baseline while the fuzz input varies the rest of
    // the resource and command graph.
    ShaderInfo seedShader{};
    seedShader.debugName = "seed_compute";
    seedShader.entry = "main";
    seedShader.src = SCENARIO_FUZZER_RESOURCE_DIR "/shaders/test_barrier/add_one.comp";
    seedShader.shaderType = ShaderType::GLSL;
    seedShader.stage = ShaderStage::Compute;
    const auto seedShaderId = builder.addShader(seedShader);
    const BufferInfo seedInput{"seed_input", 256, 0};
    const BufferInfo seedOutput{"seed_output", 256, 0};
    const auto seedInputId = builder.addBuffer(seedInput);
    const auto seedOutputId = builder.addBuffer(seedOutput);
    buffers.push_back(seedInputId);
    buffers.push_back(seedOutputId);
    resources.bufferRecords.push_back({seedInputId, seedInput});
    resources.bufferRecords.push_back({seedOutputId, seedOutput});
    resources.computeShaderRecords.push_back(
        {seedShaderId, seedInputId, seedOutputId, seedInput.size, seedOutput.size});

    DispatchComputeData seedDispatch{seedShaderId};
    seedDispatch.debugName = "seed_dispatch";
    seedDispatch.bindings = {{0, 0, seedInputId, std::nullopt, vk::DescriptorType::eStorageBuffer},
                             {0, 1, seedOutputId, std::nullopt, vk::DescriptorType::eStorageBuffer}};
    seedDispatch.computeDispatch.gwcx = 1;
    builder.addDispatchCompute(std::move(seedDispatch));
}

void seedGraphicsScenario(ScenarioBuilderImpl &builder, FuzzResources &resources) {
    auto &images = resources.images;
    auto &vertexShaders = resources.vertexShaders;
    auto &fragmentShaders = resources.fragmentShaders;
    ShaderInfo seedVertex{};
    seedVertex.debugName = "seed_vertex";
    seedVertex.entry = "main";
    seedVertex.src = SCENARIO_FUZZER_RESOURCE_DIR "/shaders/test_fragment/fullscreen_triangle.vert";
    seedVertex.shaderType = ShaderType::GLSL;
    seedVertex.stage = ShaderStage::Vertex;
    const auto seedVertexId = builder.addShader(seedVertex);
    vertexShaders.push_back(seedVertexId);

    ShaderInfo seedFragment{};
    seedFragment.debugName = "seed_fragment";
    seedFragment.entry = "main";
    seedFragment.src = SCENARIO_FUZZER_RESOURCE_DIR "/shaders/test_fragment/sampled_copy.frag";
    seedFragment.shaderType = ShaderType::GLSL;
    seedFragment.stage = ShaderStage::Fragment;
    const auto seedFragmentId = builder.addShader(seedFragment);
    fragmentShaders.push_back(seedFragmentId);

    ImageInfo seedInput{};
    seedInput.debugName = "seed_sampled_input";
    seedInput.shape = {1, 4, 4, 1};
    seedInput.format = vk::Format::eR8G8B8A8Unorm;
    seedInput.targetFormat = seedInput.format;
    seedInput.isInput = true;
    seedInput.isSampled = true;
    seedInput.mips = 1;
    seedInput.tiling = Tiling::Optimal;
    const auto seedInputImageId = builder.addImage(seedInput);
    images.push_back(seedInputImageId);
    resources.imageRecords.push_back({seedInputImageId, seedInput});
    resources.sampledImages.push_back(seedInputImageId);

    ImageInfo seedOutput = seedInput;
    seedOutput.debugName = "seed_color_output";
    seedOutput.isInput = false;
    seedOutput.isSampled = false;
    seedOutput.isColorAttachment = true;
    const auto seedOutputImageId = builder.addImage(seedOutput);
    images.push_back(seedOutputImageId);
    resources.imageRecords.push_back({seedOutputImageId, seedOutput});
    resources.colorAttachments.push_back(seedOutputImageId);

    DispatchFragmentData seedFragmentDispatch{seedVertexId, seedFragmentId};
    seedFragmentDispatch.debugName = "seed_fragment_dispatch";
    seedFragmentDispatch.bindings = {{0, 0, seedInputImageId, std::nullopt, vk::DescriptorType::eCombinedImageSampler}};
    seedFragmentDispatch.colorAttachments.push_back({seedOutputImageId, std::nullopt});
    builder.addDispatchFragment(std::move(seedFragmentDispatch));
}

void seedVgfImageResources(ScenarioBuilderImpl &builder, FuzzResources &resources) {
    for (size_t index = 0; index < 2; ++index) {
        ImageInfo info{};
        info.debugName = "seed_vgf_image_" + std::to_string(index);
        info.shape = {1, 4, 4, 1};
        info.format = vk::Format::eR8G8B8A8Unorm;
        info.targetFormat = info.format;
        info.isInput = true;
        info.isStorage = true;
        info.mips = 1;
        info.tiling = Tiling::Optimal;
        const auto id = builder.addImage(info);
        resources.images.push_back(id);
        resources.imageRecords.push_back({id, info});
        resources.storageImages.push_back(id);
    }
}

void seedVgfScenario(ScenarioBuilderImpl &builder, FuzzResources &resources) {
    const auto vgfId = addVgf(builder, resources, VgfResourceKind::Buffer, resources.buffers[0], resources.buffers[1],
                              {resources.bufferRecords[0].info.size}, {resources.bufferRecords[1].info.size},
                              vk::Format::eR8Sint, "seed_vgf");
    const auto &vgf = resources.vgfs.back();

    DispatchVgfData dispatch{vgfId};
    dispatch.debugName = "seed_vgf_dispatch";
    dispatch.bindings = {{0, 0, vgf.input, std::nullopt, vk::DescriptorType::eStorageBuffer},
                         {0, 1, vgf.output, std::nullopt, vk::DescriptorType::eStorageBuffer}};
    dispatch.shaderSubstitutions.push_back({vgf.shader, std::string(vgfModuleName(vgf.kind))});
    builder.addDispatchVgf(std::move(dispatch));
}

void seedDataGraphScenario(ScenarioBuilderImpl &builder, FuzzResources &resources) {
    const auto &input = resources.tensorRecords[1];
    const auto &output = resources.tensorRecords[2];
    const auto shader = addDataGraphShader(builder, resources, input, output, "seed_data_graph");

    DispatchDataGraphData dispatch{shader};
    dispatch.debugName = "seed_data_graph_dispatch";
    dispatch.bindings = {{0, 0, input.id, std::nullopt, vk::DescriptorType::eTensorARM},
                         {0, 1, output.id, std::nullopt, vk::DescriptorType::eTensorARM}};
    builder.addDispatchDataGraph(std::move(dispatch));
}

void seedScenario(ScenarioBuilderImpl &builder, FuzzResources &resources) {
    seedComputeScenario(builder, resources);
    seedGraphicsScenario(builder, resources);
    seedVgfImageResources(builder, resources);
    const TensorInfo seedTensor{"seed_transfer_tensor", {1, 4, 4, 1}, vk::Format::eR8Uint};
    const auto seedTensorId = builder.addTensor(seedTensor);
    resources.tensors.push_back(seedTensorId);
    resources.tensorRecords.push_back({seedTensorId, seedTensor});
    for (size_t index = 0; index < 2; ++index) {
        const TensorInfo vgfTensor{"seed_vgf_tensor_" + std::to_string(index), {1, 4, 4, 1}, vk::Format::eR8Sint};
        const auto id = builder.addTensor(vgfTensor);
        resources.tensors.push_back(id);
        resources.tensorRecords.push_back({id, vgfTensor});
    }
    seedVgfScenario(builder, resources);
    seedDataGraphScenario(builder, resources);
}

struct RejectionScenario {
    RejectionScenario() {
        seedScenario(builder, resources);
        scenario = builder.build({});
    }

    ScenarioBuilderImpl builder;
    FuzzResources resources;
    std::unique_ptr<Scenario> scenario;
};

RejectionScenario &rejectionScenario() {
    static RejectionScenario scenario;
    return scenario;
}

struct BuilderOperationContext {
    ScenarioBuilderImpl &builder;
    ByteCursor &operation;
    FuzzResources &resources;
};

BufferId selectCompatibleBuffer(const FuzzResources &resources, uint32_t size, uint8_t selector, BufferId fallback,
                                std::optional<BufferId> excluded = std::nullopt) {
    const auto start = static_cast<size_t>(selector) % resources.bufferRecords.size();
    for (size_t offset = 0; offset < resources.bufferRecords.size(); ++offset) {
        const auto &candidate = resources.bufferRecords[(start + offset) % resources.bufferRecords.size()];
        if (candidate.info.size == size && (!excluded.has_value() || candidate.id != *excluded)) {
            return candidate.id;
        }
    }
    return fallback;
}

ImageId selectCompatibleImage(const FuzzResources &resources, const std::vector<int64_t> &shape, vk::Format format,
                              uint8_t selector, ImageId fallback, std::optional<ImageId> excluded = std::nullopt) {
    const auto start = static_cast<size_t>(selector) % resources.imageRecords.size();
    for (size_t offset = 0; offset < resources.imageRecords.size(); ++offset) {
        const auto &candidate = resources.imageRecords[(start + offset) % resources.imageRecords.size()];
        if (candidate.info.isStorage && candidate.info.shape == shape && candidate.info.format == format &&
            (!excluded.has_value() || candidate.id != *excluded)) {
            return candidate.id;
        }
    }
    return fallback;
}

TensorId selectCompatibleTensor(const FuzzResources &resources, const std::vector<int64_t> &shape, vk::Format format,
                                uint8_t selector, TensorId fallback, std::optional<TensorId> excluded = std::nullopt) {
    const auto start = static_cast<size_t>(selector) % resources.tensorRecords.size();
    for (size_t offset = 0; offset < resources.tensorRecords.size(); ++offset) {
        const auto &candidate = resources.tensorRecords[(start + offset) % resources.tensorRecords.size()];
        if (candidate.info.shape == shape && candidate.info.format == format &&
            (!excluded.has_value() || candidate.id != *excluded)) {
            return candidate.id;
        }
    }
    return fallback;
}

void applyAddBuffer(BuilderOperationContext &context) {
    const auto duplicate = (context.operation.next() & 1u) != 0 && !context.resources.bufferRecords.empty();
    BufferInfo info;
    if (duplicate) {
        info = context.resources.bufferRecords[context.operation.next() % context.resources.bufferRecords.size()].info;
        info.debugName = debugName(context.operation.next());
    } else {
        info = BufferInfo{debugName(context.operation.next()),
                          static_cast<uint32_t>((context.operation.next() % 32 + 1) * 4), 0};
    }
    // Consume the offset byte while keeping normal inputs aligned.
    (void)context.operation.next();
    const auto id = context.builder.addBuffer(info);
    context.resources.buffers.push_back(id);
    context.resources.bufferRecords.push_back({id, info});
    return;
}

void applyAddTensor(BuilderOperationContext &context) {
    const auto duplicate = (context.operation.next() & 1u) != 0 && !context.resources.tensorRecords.empty();
    TensorInfo info;
    if (duplicate) {
        info = context.resources.tensorRecords[context.operation.next() % context.resources.tensorRecords.size()].info;
        info.debugName = debugName(context.operation.next());
    } else {
        info = TensorInfo{debugName(context.operation.next()),
                          {1, static_cast<int64_t>(context.operation.next() % 8 + 1), 1, 1},
                          vk::Format::eR8Sint};
    }
    // Keep normal structured inputs valid for Vulkan's alignment
    // requirement. Misaligned offsets belong in the malformed-input
    // target, not in the deep execution campaign.
    (void)context.operation.next();
    info.memoryOffset = 0;
    const auto id = context.builder.addTensor(info);
    context.resources.tensors.push_back(id);
    context.resources.tensorRecords.push_back({id, info});
    return;
}

void applyAddImage(BuilderOperationContext &context) {
    const auto duplicate = (context.operation.next() & 1u) != 0 && !context.resources.imageRecords.empty();
    ImageInfo info{};
    if (duplicate) {
        info = context.resources.imageRecords[context.operation.next() % context.resources.imageRecords.size()].info;
        info.debugName = debugName(context.operation.next());
    } else {
        const auto role = context.operation.next() % 3;
        const auto extent = static_cast<int64_t>(context.operation.next() % 8 + 1);
        info.debugName = debugName(context.operation.next());
        info.shape = {1, extent, extent, 1};
        info.format = vk::Format::eR8G8B8A8Unorm;
        info.targetFormat = info.format;
        info.isInput = role != 1;
        info.isSampled = role == 0;
        info.isColorAttachment = role == 1;
        info.isStorage = role == 2;
        info.mips = 1;
        info.tiling = Tiling::Optimal;
    }
    const auto id = context.builder.addImage(info);
    context.resources.images.push_back(id);
    context.resources.imageRecords.push_back({id, info});
    if (info.isSampled) {
        context.resources.sampledImages.push_back(id);
    }
    if (info.isColorAttachment) {
        context.resources.colorAttachments.push_back(id);
    }
    if (info.isStorage) {
        context.resources.storageImages.push_back(id);
    }
    return;
}

void applyAddResourceToMemoryGroup(BuilderOperationContext &context) {
    const auto resourceType = context.operation.next() % 3;
    const auto resourceIndex = context.operation.next();
    const auto groupIndex = context.operation.next() % (context.resources.memoryGroups.size() + 1);
    if (groupIndex == context.resources.memoryGroups.size()) {
        context.resources.memoryGroups.push_back(context.builder.createMemoryGroup());
    }
    const auto group = context.resources.memoryGroups[groupIndex];
    switch (resourceType) {
    case 0:
        context.builder.addResourceToMemoryGroup(
            group, context.resources.buffers[resourceIndex % context.resources.buffers.size()]);
        break;
    case 1:
        context.builder.addResourceToMemoryGroup(
            group, context.resources.images[resourceIndex % context.resources.images.size()]);
        break;
    case 2:
        if (!context.resources.tensors.empty()) {
            context.builder.addResourceToMemoryGroup(
                group, context.resources.tensors[resourceIndex % context.resources.tensors.size()]);
        }
        break;
    }
    return;
}

void applyAddShader(BuilderOperationContext &context) {
    ShaderInfo info{};
    info.debugName = debugName(context.operation.next());
    info.entry = "main";
    const auto kind = context.operation.next() % 4;
    if (kind == 0) {
        const auto inputIndex = static_cast<size_t>(context.operation.next()) % context.resources.bufferRecords.size();
        auto outputIndex = static_cast<size_t>(context.operation.next()) % context.resources.bufferRecords.size();
        if (outputIndex == inputIndex && context.resources.bufferRecords.size() > 1) {
            outputIndex = (outputIndex + 1) % context.resources.bufferRecords.size();
        }
        const auto &input = context.resources.bufferRecords[inputIndex];
        const auto &output = context.resources.bufferRecords[outputIndex];
        info.src = SCENARIO_FUZZER_SOURCE_DIR "/variable_copy.comp";
        info.shaderType = ShaderType::GLSL;
        info.stage = ShaderStage::Compute;
        const auto id = context.builder.addShader(info);
        context.resources.computeShaderRecords.push_back({id, input.id, output.id, input.info.size, output.info.size});
    } else if (kind == 1) {
        info.src = SCENARIO_FUZZER_RESOURCE_DIR "/shaders/test_fragment/fullscreen_triangle.vert";
        info.shaderType = ShaderType::GLSL;
        info.stage = ShaderStage::Vertex;
        context.resources.vertexShaders.push_back(context.builder.addShader(info));
    } else if (kind == 2) {
        info.src = SCENARIO_FUZZER_RESOURCE_DIR "/shaders/test_fragment/sampled_copy.frag";
        info.shaderType = ShaderType::GLSL;
        info.stage = ShaderStage::Fragment;
        context.resources.fragmentShaders.push_back(context.builder.addShader(info));
    } else {
        std::vector<const FuzzResources::TensorRecord *> candidates;
        for (const auto &candidate : context.resources.tensorRecords) {
            if (candidate.info.shape.size() == 4 && candidate.info.format == vk::Format::eR8Sint) {
                candidates.push_back(&candidate);
            }
        }
        if (candidates.size() < 2) {
            return;
        }
        const auto inputIndex = static_cast<size_t>(context.operation.next()) % candidates.size();
        auto outputIndex = static_cast<size_t>(context.operation.next()) % candidates.size();
        if (outputIndex == inputIndex) {
            outputIndex = (outputIndex + 1) % candidates.size();
        }
        addDataGraphShader(context.builder, context.resources, *candidates[inputIndex], *candidates[outputIndex],
                           info.debugName);
    }
    return;
}

void applyAddRawData(BuilderOperationContext &context) {
    const auto value = context.operation.next();
    RawDataInfo info{debugName(value), writeRawData(value, context.resources.rawData.size())};
    context.resources.rawData.push_back(context.builder.addRawData(info));
    return;
}

void applyAddVgf(BuilderOperationContext &context) {
    const auto kind = static_cast<VgfResourceKind>(context.operation.next() % 3);
    if (kind == VgfResourceKind::Buffer) {
        const auto inputIndex = static_cast<size_t>(context.operation.next()) % context.resources.bufferRecords.size();
        auto outputIndex = static_cast<size_t>(context.operation.next()) % context.resources.bufferRecords.size();
        if (outputIndex == inputIndex && context.resources.bufferRecords.size() > 1) {
            outputIndex = (outputIndex + 1) % context.resources.bufferRecords.size();
        }
        const auto &input = context.resources.bufferRecords[inputIndex];
        const auto &output = context.resources.bufferRecords[outputIndex];
        addVgf(context.builder, context.resources, kind, input.id, output.id, {input.info.size}, {output.info.size},
               vk::Format::eR8Sint, debugName(context.operation.next()));
        return;
    }

    if (kind == VgfResourceKind::Image) {
        std::vector<const FuzzResources::ImageRecord *> candidates;
        for (const auto &candidate : context.resources.imageRecords) {
            if (candidate.info.isStorage && candidate.info.format == vk::Format::eR8G8B8A8Unorm) {
                candidates.push_back(&candidate);
            }
        }
        if (candidates.size() < 2) {
            return;
        }
        const auto inputIndex = static_cast<size_t>(context.operation.next()) % candidates.size();
        auto outputIndex = static_cast<size_t>(context.operation.next()) % candidates.size();
        if (outputIndex == inputIndex) {
            outputIndex = (outputIndex + 1) % candidates.size();
        }
        const auto &input = *candidates[inputIndex];
        const auto &output = *candidates[outputIndex];
        addVgf(context.builder, context.resources, kind, input.id, output.id, input.info.shape, output.info.shape,
               input.info.format, debugName(context.operation.next()));
        return;
    }

    std::vector<const FuzzResources::TensorRecord *> candidates;
    for (const auto &candidate : context.resources.tensorRecords) {
        if (candidate.info.shape.size() == 4 && candidate.info.format == vk::Format::eR8Sint) {
            candidates.push_back(&candidate);
        }
    }
    if (candidates.size() < 2) {
        return;
    }
    const auto start = static_cast<size_t>(context.operation.next()) % candidates.size();
    const FuzzResources::TensorRecord *input{};
    const FuzzResources::TensorRecord *output{};
    for (size_t offset = 0; offset < candidates.size() && output == nullptr; ++offset) {
        const auto *candidate = candidates[(start + offset) % candidates.size()];
        for (size_t outputOffset = 1; outputOffset < candidates.size(); ++outputOffset) {
            const auto *compatible = candidates[(start + offset + outputOffset) % candidates.size()];
            if (candidate->info.shape == compatible->info.shape) {
                input = candidate;
                output = compatible;
                break;
            }
        }
    }
    if (output != nullptr) {
        addVgf(context.builder, context.resources, kind, input->id, output->id, input->info.shape, output->info.shape,
               input->info.format, debugName(context.operation.next()));
    }
    return;
}

void applyAddGraphConstant(BuilderOperationContext &context) {
    GraphConstantInfo info{debugName(context.operation.next()), vk::Format::eR8Uint, {1}};
    info.data.push_back(context.operation.next());
    context.resources.graphConstants.push_back(context.builder.addGraphConstant(info));
    return;
}

void applyAddImageBarrier(BuilderOperationContext &context) {
    ImageBarrierInfo info{};
    info.debugName = debugName(context.operation.next());
    info.image = context.resources.images[context.operation.next() % context.resources.images.size()];
    info.srcAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    info.dstAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    info.srcStages = {static_cast<PipelineStage>(context.operation.next() % 4)};
    info.dstStages = {static_cast<PipelineStage>(context.operation.next() % 4)};
    info.oldLayout = static_cast<ImageLayout>(context.operation.next() % 3);
    info.newLayout = static_cast<ImageLayout>(context.operation.next() % 3);
    info.range = {0, 1, 0, 1};
    context.resources.imageBarriers.push_back(context.builder.addImageBarrier(info));
    return;
}

void applyAddBufferBarrier(BuilderOperationContext &context) {
    BufferBarrierInfo info{};
    info.debugName = debugName(context.operation.next());
    // The two seed buffers are always 256 bytes, keeping the
    // generated range valid while access/stage fields vary.
    info.buffer = context.resources.buffers[context.operation.next() % 2];
    info.srcAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    info.dstAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    info.srcStages = {static_cast<PipelineStage>(context.operation.next() % 4)};
    info.dstStages = {static_cast<PipelineStage>(context.operation.next() % 4)};
    info.offset = context.operation.next() % 128;
    info.size = context.operation.next() % (256 - info.offset) + 1;
    context.resources.bufferBarriers.push_back(context.builder.addBufferBarrier(info));
    return;
}

void applyAddTensorBarrier(BuilderOperationContext &context) {
    if (context.resources.tensors.empty()) {
        return;
    }
    TensorBarrierInfo info{};
    info.debugName = debugName(context.operation.next());
    info.tensor = context.resources.tensors[context.operation.next() % context.resources.tensors.size()];
    info.srcAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    info.dstAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    info.srcStages = {static_cast<PipelineStage>(context.operation.next() % 4)};
    info.dstStages = {static_cast<PipelineStage>(context.operation.next() % 4)};
    context.resources.tensorBarriers.push_back(context.builder.addTensorBarrier(info));

    return;
}

void applyAddMemoryBarrier(BuilderOperationContext &context) {
    MemoryBarrierInfo info{};
    info.srcAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    info.dstAccess = static_cast<MemoryAccess>(context.operation.next() % 6);
    context.resources.memoryBarriers.push_back(context.builder.addMemoryBarrier(info));
    return;
}

void applyAddComputeDispatch(BuilderOperationContext &context) {
    const auto &shader =
        context.resources
            .computeShaderRecords[context.operation.next() % context.resources.computeShaderRecords.size()];
    const auto input =
        selectCompatibleBuffer(context.resources, shader.inputSize, context.operation.next(), shader.input);
    const auto output =
        selectCompatibleBuffer(context.resources, shader.outputSize, context.operation.next(), shader.output, input);
    DispatchComputeData command{shader.id};
    command.debugName = debugName(context.operation.next());
    command.bindings = {{0, 0, input, std::nullopt, vk::DescriptorType::eStorageBuffer},
                        {0, 1, output, std::nullopt, vk::DescriptorType::eStorageBuffer}};
    command.computeDispatch.gwcx = shader.outputSize;
    command.implicitBarrier = (context.operation.next() & 1u) != 0;
    context.builder.addDispatchCompute(std::move(command));
    return;
}

void applyAddFragmentDispatch(BuilderOperationContext &context) {
    DispatchFragmentData command{
        context.resources.vertexShaders[context.operation.next() % context.resources.vertexShaders.size()],
        context.resources.fragmentShaders[context.operation.next() % context.resources.fragmentShaders.size()]};
    command.debugName = debugName(context.operation.next());
    command.bindings = {
        {0, 0, context.resources.sampledImages[context.operation.next() % context.resources.sampledImages.size()],
         std::nullopt, vk::DescriptorType::eCombinedImageSampler}};
    command.colorAttachments.push_back(
        {context.resources.colorAttachments[context.operation.next() % context.resources.colorAttachments.size()],
         std::nullopt});
    command.implicitBarrier = (context.operation.next() & 1u) != 0;
    context.builder.addDispatchFragment(std::move(command));
    return;
}

void applyAddVgfDispatch(BuilderOperationContext &context) {
    if (!context.resources.vgfs.empty()) {
        const auto &vgf = context.resources.vgfs[context.operation.next() % context.resources.vgfs.size()];
        MemoryResourceId input = vgf.input;
        MemoryResourceId output = vgf.output;
        switch (vgf.kind) {
        case VgfResourceKind::Buffer: {
            const auto fallbackInput = std::get<BufferId>(vgf.input);
            const auto fallbackOutput = std::get<BufferId>(vgf.output);
            const auto selectedInput = selectCompatibleBuffer(
                context.resources, static_cast<uint32_t>(vgf.inputShape[0]), context.operation.next(), fallbackInput);
            input = selectedInput;
            output = selectCompatibleBuffer(context.resources, static_cast<uint32_t>(vgf.outputShape[0]),
                                            context.operation.next(), fallbackOutput, selectedInput);
            break;
        }
        case VgfResourceKind::Image: {
            const auto fallbackInput = std::get<ImageId>(vgf.input);
            const auto fallbackOutput = std::get<ImageId>(vgf.output);
            const auto selectedInput = selectCompatibleImage(context.resources, vgf.inputShape, vgf.format,
                                                             context.operation.next(), fallbackInput);
            input = selectedInput;
            output = selectCompatibleImage(context.resources, vgf.outputShape, vgf.format, context.operation.next(),
                                           fallbackOutput, selectedInput);
            break;
        }
        case VgfResourceKind::Tensor: {
            const auto fallbackInput = std::get<TensorId>(vgf.input);
            const auto fallbackOutput = std::get<TensorId>(vgf.output);
            const auto selectedInput = selectCompatibleTensor(context.resources, vgf.inputShape, vgf.format,
                                                              context.operation.next(), fallbackInput);
            input = selectedInput;
            output = selectCompatibleTensor(context.resources, vgf.outputShape, vgf.format, context.operation.next(),
                                            fallbackOutput, selectedInput);
            break;
        }
        }
        DispatchVgfData command{vgf.id};
        const auto descriptorType = vgfVkDescriptorType(vgf.kind);
        command.bindings = {{0, 0, input, std::nullopt, descriptorType}, {0, 1, output, std::nullopt, descriptorType}};
        command.shaderSubstitutions.push_back({vgf.shader, std::string(vgfModuleName(vgf.kind))});
        context.builder.addDispatchVgf(std::move(command));
    }
    return;
}

void applyAddDataGraphDispatch(BuilderOperationContext &context) {
    if (context.resources.graphShaderRecords.empty()) {
        return;
    }
    const auto &shader =
        context.resources.graphShaderRecords[context.operation.next() % context.resources.graphShaderRecords.size()];
    const auto input = selectCompatibleTensor(context.resources, shader.inputShape, shader.format,
                                              context.operation.next(), shader.input);
    const auto output = selectCompatibleTensor(context.resources, shader.outputShape, shader.format,
                                               context.operation.next(), shader.output, input);
    DispatchDataGraphData command{shader.id};
    command.debugName = debugName(context.operation.next());
    command.bindings = {{0, 0, input, std::nullopt, vk::DescriptorType::eTensorARM},
                        {0, 1, output, std::nullopt, vk::DescriptorType::eTensorARM}};
    command.implicitBarrier = (context.operation.next() & 1u) != 0;
    context.builder.addDispatchDataGraph(std::move(command));
    return;
}

void applyAddPipelineBarrier(BuilderOperationContext &context) {
    PipelineBarrierData command;
    if (!context.resources.memoryBarriers.empty()) {
        command.memoryBarriers.push_back(
            context.resources.memoryBarriers[context.operation.next() % context.resources.memoryBarriers.size()]);
    }
    if (!context.resources.imageBarriers.empty()) {
        command.imageBarriers.push_back(
            context.resources.imageBarriers[context.operation.next() % context.resources.imageBarriers.size()]);
    }
    if (!context.resources.tensorBarriers.empty()) {
        command.tensorBarriers.push_back(
            context.resources.tensorBarriers[context.operation.next() % context.resources.tensorBarriers.size()]);
    }
    if (!context.resources.bufferBarriers.empty()) {
        command.bufferBarriers.push_back(
            context.resources.bufferBarriers[context.operation.next() % context.resources.bufferBarriers.size()]);
    }
    context.builder.addPipelineBarrier(std::move(command));
    return;
}

void applyAddOpticalFlowDispatch(BuilderOperationContext &context) {
    if (context.resources.storageImages.size() < 3) {
        return;
    }
    const auto selected =
        context.resources.storageImages[context.operation.next() % context.resources.storageImages.size()];
    const auto selectedRecord =
        std::find_if(context.resources.imageRecords.begin(), context.resources.imageRecords.end(),
                     [&](const FuzzResources::ImageRecord &record) { return record.id == selected; });
    if (selectedRecord == context.resources.imageRecords.end()) {
        return;
    }
    std::vector<ImageId> compatible;
    for (const auto &candidate : context.resources.imageRecords) {
        if (candidate.info.isStorage && candidate.info.shape == selectedRecord->info.shape &&
            candidate.info.format == selectedRecord->info.format &&
            candidate.info.targetFormat == selectedRecord->info.targetFormat) {
            compatible.push_back(candidate.id);
        }
    }
    if (compatible.size() < 3) {
        return;
    }
    auto makeImageBinding = [&](ImageId image) {
        return TypedBinding{0, 0, image, std::nullopt, vk::DescriptorType::eStorageImage};
    };
    const auto start = static_cast<size_t>(context.operation.next()) % compatible.size();
    DispatchOpticalFlowData command{makeImageBinding(compatible[start]),
                                    makeImageBinding(compatible[(start + 1) % compatible.size()]),
                                    makeImageBinding(compatible[(start + 2) % compatible.size()])};
    command.width = static_cast<uint32_t>(selectedRecord->info.shape[2]);
    command.height = static_cast<uint32_t>(selectedRecord->info.shape[1]);
    context.builder.addDispatchOpticalFlow(std::move(command));
    return;
}

void applyAddFrameBoundary(BuilderOperationContext &context) {
    FrameBoundaryData command;
    command.buffers.push_back(context.resources.buffers[context.operation.next() % context.resources.buffers.size()]);
    command.images.push_back(context.resources.images[context.operation.next() % context.resources.images.size()]);
    if (!context.resources.tensors.empty()) {
        command.tensors.push_back(
            context.resources.tensors[context.operation.next() % context.resources.tensors.size()]);
    }
    context.builder.addFrameBoundary(std::move(command));
    return;
}

void applyInvalidResourceOperation(BuilderOperationContext &context) {
    // Deliberately invalid IDs should be rejected synchronously by
    // the context.builder. Treat acceptance as a fuzzing invariant failure.
    const auto invalid = std::numeric_limits<size_t>::max() - context.operation.next();
    std::string_view expectedMessage;
    bool rejected = false;
    try {
        switch (context.operation.next() % 4) {
        case 0:
            expectedMessage = "Shader resource does not exist";
            context.builder.addDispatchCompute(DispatchComputeData{ShaderId{invalid}});
            break;
        case 1: {
            expectedMessage = "Image resource does not exist";
            ImageBarrierInfo info{};
            info.image = ImageId{invalid};
            context.builder.addImageBarrier(info);
            break;
        }
        case 2:
            expectedMessage = "Memory group does not exist";
            context.builder.addResourceToMemoryGroup(MemoryGroupId{invalid}, context.resources.buffers[0]);
            break;
        case 3: {
            expectedMessage = "Memory barrier resource does not exist";
            PipelineBarrierData command;
            command.memoryBarriers.push_back(MemoryBarrierId{invalid});
            context.builder.addPipelineBarrier(std::move(command));
            break;
        }
        }
    } catch (const std::runtime_error &error) {
        rejected = error.what() == expectedMessage;
    }
    if (!rejected) {
        std::abort();
    }
    return;
}

void fuzzBuilderOperations(ScenarioBuilderImpl &builder, ByteCursor &cursor, FuzzResources &resources) {

    // Decode fixed-size records so mutations change one operation without
    // shifting every subsequent field. Short final records are zero-padded.
    size_t operationCount{};
    while (!cursor.empty() && operationCount++ < 32) {
        std::array<uint8_t, kOperationRecordSize> record{};
        for (auto &byte : record) {
            byte = cursor.next();
        }
        ByteCursor operation{record.data(), record.size()};
        fuzzStats.operationRecord();

        try {
            BuilderOperationContext context{builder, operation, resources};
            const auto selector = operation.next();
            const auto builderOperation =
                static_cast<BuilderOperation>(selector % static_cast<int>(BuilderOperation::Count));
            switch (builderOperation) {
            case BuilderOperation::AddBuffer:
                applyAddBuffer(context);
                break;
            case BuilderOperation::AddTensor:
                applyAddTensor(context);
                break;
            case BuilderOperation::AddImage:
                applyAddImage(context);
                break;
            case BuilderOperation::AddResourceToMemoryGroup:
                applyAddResourceToMemoryGroup(context);
                break;
            case BuilderOperation::AddShader:
                applyAddShader(context);
                break;
            case BuilderOperation::AddRawData:
                applyAddRawData(context);
                break;
            case BuilderOperation::AddVgf:
                applyAddVgf(context);
                break;
            case BuilderOperation::AddGraphConstant:
                applyAddGraphConstant(context);
                break;
            case BuilderOperation::AddImageBarrier:
                applyAddImageBarrier(context);
                break;
            case BuilderOperation::AddBufferBarrier:
                applyAddBufferBarrier(context);
                break;
            case BuilderOperation::AddTensorBarrier:
                applyAddTensorBarrier(context);
                break;
            case BuilderOperation::AddMemoryBarrier:
                applyAddMemoryBarrier(context);
                break;
            case BuilderOperation::AddComputeDispatch:
                applyAddComputeDispatch(context);
                break;
            case BuilderOperation::AddFragmentDispatch:
                applyAddFragmentDispatch(context);
                break;
            case BuilderOperation::AddVgfDispatch:
                applyAddVgfDispatch(context);
                break;
            case BuilderOperation::AddDataGraphDispatch:
                applyAddDataGraphDispatch(context);
                break;
            case BuilderOperation::AddPipelineBarrier:
                applyAddPipelineBarrier(context);
                break;
            case BuilderOperation::AddOpticalFlowDispatch:
                applyAddOpticalFlowDispatch(context);
                break;
            case BuilderOperation::AddFrameBoundary:
                applyAddFrameBoundary(context);
                break;
            case BuilderOperation::Count:
                std::abort();
            }
        } catch (const std::runtime_error &error) {
            if (std::string_view{error.what()} != "Resource already belongs to a different group") {
                throw;
            }
            fuzzStats.operationRejected();
        }
    }
}

std::vector<uint8_t> fuzzPayload(const uint8_t *data, size_t size, size_t payloadSize, uint8_t iteration) {
    std::vector<uint8_t> payload(payloadSize);
    if (data == nullptr || size == 0) {
        return payload;
    }
    for (size_t i = 0; i < payload.size(); ++i) {
        payload[i] = data[i % size] ^ iteration;
    }
    return payload;
}

void fuzzTransferValidation(Scenario &scenario, const FuzzResources &resources, uint8_t selector) {
    const auto invalid = std::numeric_limits<size_t>::max();
    const std::array<uint8_t, 256> payload{};
    const auto operation = selector % 16;
    std::string_view expectedMessage;
    try {
        switch (operation) {
        case 0:
            expectedMessage = "Scenario::upload: Buffer resource not found.";
            scenario.upload(BufferId{invalid}, {payload.data(), payload.size()});
            break;
        case 1:
            expectedMessage = "Scenario::upload: Image resource not found.";
            scenario.upload(ImageId{invalid}, {payload.data(), 64, {1, 4, 4, 1}, vk::Format::eR8G8B8A8Unorm});
            break;
        case 2:
            expectedMessage = "Scenario::upload: Tensor resource not found.";
            scenario.upload(TensorId{invalid}, {payload.data(), 16, {1, 4, 4, 1}, vk::Format::eR8Uint});
            break;
        case 3:
            expectedMessage = "Scenario::download: Buffer resource not found.";
            (void)scenario.download(BufferId{invalid});
            break;
        case 4:
            expectedMessage = "Scenario::download: Image resource not found.";
            (void)scenario.download(ImageId{invalid});
            break;
        case 5:
            expectedMessage = "Scenario::download: Tensor resource not found.";
            (void)scenario.download(TensorId{invalid});
            break;
        case 6:
            expectedMessage = "Buffer::upload: size mismatch";
            scenario.upload(resources.buffers[0], {payload.data(), 255});
            break;
        case 7:
            expectedMessage = "Image::upload: expected packed mip data size to be ";
            scenario.upload(resources.images[0], {payload.data(), 63, {1, 4, 4, 1}, vk::Format::eR8G8B8A8Unorm});
            break;
        case 8:
            expectedMessage = "Tensor::upload: size does not match logical data size";
            scenario.upload(resources.tensors[0], {payload.data(), 15, {1, 4, 4, 1}, vk::Format::eR8Uint});
            break;
        case 9:
            expectedMessage = "Image::upload: provided shape does not match image shape";
            scenario.upload(resources.images[0], {payload.data(), 64, {1, 2, 8, 1}, vk::Format::eR8G8B8A8Unorm});
            break;
        case 10:
            expectedMessage = "Tensor::upload: provided shape does not match tensor shape";
            scenario.upload(resources.tensors[0], {payload.data(), 16, {1, 2, 8, 1}, vk::Format::eR8Uint});
            break;
        case 11:
            expectedMessage = "Image::upload: provided format does not match image format";
            scenario.upload(resources.images[0], {payload.data(), 64, {1, 4, 4, 1}, vk::Format::eR8Uint});
            break;
        case 12:
            expectedMessage = "Tensor::upload: provided format does not match tensor format";
            scenario.upload(resources.tensors[0], {payload.data(), 16, {1, 4, 4, 1}, vk::Format::eR16Uint});
            break;
        case 13:
            expectedMessage = "Image::upload: mipLevels must be between 1 and ";
            scenario.upload(resources.images[0], {payload.data(), 64, {1, 4, 4, 1}, vk::Format::eR8G8B8A8Unorm, 0});
            break;
        case 14:
            expectedMessage = "Scenario repeat count must be greater than zero; received 0.";
            scenario.run(0, false);
            break;
        case 15:
            expectedMessage = "Scenario repeat count must be greater than zero; received -1.";
            scenario.run(-1, false);
            break;
        }
    } catch (const std::invalid_argument &error) {
        if (operation >= 14 && std::string_view{error.what()} == expectedMessage) {
            return;
        }
        std::abort();
    } catch (const std::runtime_error &error) {
        if (operation < 14 && startsWith(error.what(), expectedMessage)) {
            return;
        }
        std::abort();
    }
    // Invalid IDs, metadata and repeat counts must be rejected with the
    // documented exception type and diagnostic.
    std::abort();
}

bool fuzzScenarioConstruction(const uint8_t *data, size_t size, std::string *buildRejectionMessage = nullptr) {
    InputStatsGuard inputStatsGuard;
    ByteCursor cursor{data, std::min(size, size_t{512})};
    const auto scenarioFlags = cursor.next();
    ScenarioBuilderImpl builder;
    FuzzResources resources;

    seedScenario(builder, resources);
    fuzzBuilderOperations(builder, cursor, resources);

    ScenarioOptions options{};
    // Pipeline caching requires a valid process-local cache path. Keep it
    // out of the structured campaign; cache-specific validation belongs in
    // a separate targeted test.
    options.enablePipelineCaching = false;
    options.clearPipelineCache = false;
    options.failOnPipelineCacheMiss = (scenarioFlags & 0x03u) == 0x03u;
    // The emulation layer currently forwards tensor debug names to
    // Lavapipe as native handles. Keep this external-layer path out of
    // the structured execution campaign.
    options.enableGPUDebugMarkers = false;
    options.captureFrame = scenarioFlags == 0xffu;
    options.enableRobustnessFeatures = (scenarioFlags & 0xc0u) == 0xc0u;

    std::unique_ptr<Scenario> scenario;
    fuzzStats.buildAttempt();
    try {
        scenario = builder.build(options);
    } catch (const std::runtime_error &error) {
        // Invalid generated scenarios and unavailable device features may reject
        // construction. Treat these as rejected inputs; unexpected exceptions
        // after a successful build remain fuzzer findings.
        fuzzStats.buildRejected(error.what());
        if (buildRejectionMessage != nullptr) {
            *buildRejectionMessage = error.what();
        }
        return false;
    }
    fuzzStats.buildSucceeded();

    const std::array<uint8_t, 256> initialOutput{};
    for (uint8_t iteration = 0; iteration < 1 + ((scenarioFlags >> 4u) & 1u); ++iteration) {
        const auto bufferInput = fuzzPayload(data, size, 256, iteration);
        const auto imageInput = fuzzPayload(data, size, 64, iteration);
        const auto tensorInput = fuzzPayload(data, size, 16, iteration);
        scenario->upload(resources.buffers[0], {bufferInput.data(), bufferInput.size()});
        scenario->upload(resources.buffers[1], {initialOutput.data(), initialOutput.size()});
        scenario->upload(resources.images[0],
                         {imageInput.data(), imageInput.size(), {1, 4, 4, 1}, vk::Format::eR8G8B8A8Unorm});
        scenario->upload(resources.tensors[0],
                         {tensorInput.data(), tensorInput.size(), {1, 4, 4, 1}, vk::Format::eR8Uint});
        fuzzStats.runAttempt();
        if (iteration == 0) {
            scenario->run();
            fuzzStats.runCompleted(1);
        } else {
            const auto repeatCount = static_cast<uint32_t>(1 + ((scenarioFlags >> 5u) & 1u));
            scenario->run(static_cast<int>(repeatCount), (scenarioFlags & 0x08u) != 0);
            fuzzStats.runCompleted(repeatCount);
        }
        (void)scenario->download(resources.buffers[1]);
        (void)scenario->download(resources.images[1]);
        (void)scenario->download(resources.tensors[0]);
    }
    fuzzStats.endToEndCompleted();
    return true;
}

void fuzzScenarioRejectionsImpl(const uint8_t *data, size_t size) {
    const auto operation = static_cast<uint8_t>(data[0] % 21u);
    if (operation < 4) {
        ScenarioBuilderImpl builder;
        FuzzResources resources;
        const auto buffer = builder.addBuffer(BufferInfo{"rejection_buffer", 4, 0});
        resources.buffers.push_back(buffer);
        const std::array<uint8_t, 2> record{size > 1 ? data[1] : uint8_t{}, operation};
        ByteCursor cursor{record.data(), record.size()};
        BuilderOperationContext context{builder, cursor, resources};
        applyInvalidResourceOperation(context);
        return;
    }

    auto &rejection = rejectionScenario();

    if (operation == 4) {
        bool rejected = false;
        try {
            rejection.builder.createMemoryGroup();
        } catch (const std::runtime_error &error) {
            rejected = std::string_view{error.what()} == "ScenarioBuilder cannot be modified after build";
        }
        if (!rejected) {
            std::abort();
        }
        return;
    }

    fuzzTransferValidation(*rejection.scenario, rejection.resources, static_cast<uint8_t>(operation - 5u));
}
} // namespace

void fuzzScenario(const uint8_t *data, size_t size) { (void)fuzzScenarioConstruction(data, size); }

void fuzzScenarioRejections(const uint8_t *data, size_t size) {
    if (data != nullptr && size != 0) {
        fuzzScenarioRejectionsImpl(data, size);
    }
}

void validateScenarioFuzzerEnvironment() {
    constexpr std::array<uint8_t, 1> canonicalInput{};
    std::string preflightBuildRejectionMessage;
    try {
        if (!fuzzScenarioConstruction(canonicalInput.data(), canonicalInput.size(), &preflightBuildRejectionMessage)) {
            std::fprintf(stderr, "Scenario fuzzer runtime preflight failed while building the seeded scenario: %s\n",
                         preflightBuildRejectionMessage.c_str());
            std::abort();
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Scenario fuzzer runtime preflight failed: %s\n", error.what());
        std::abort();
    }
    fuzzStats.reset();
}

extern "C" size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t maxSize);

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t maxSize, unsigned int seed) {
    size = LLVMFuzzerMutate(data, size, maxSize);
    const auto minimumSize = std::min(kMinimumStructuredInputSize, maxSize);
    uint32_t state = seed;
    while (size < minimumSize) {
        state = state * 1664525u + 1013904223u;
        data[size++] = static_cast<uint8_t>(state >> 24u);
    }
    return size;
}
