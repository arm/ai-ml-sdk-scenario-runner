/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "iscenario.hpp"
#include "scenario_options.hpp"
#include "utils.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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

py::buffer_info requireContiguousArray(const py::array &array) {
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error("Resource upload requires a C-contiguous NumPy array");
    }
    return array.request();
}

std::vector<int64_t> arrayShape(const py::buffer_info &buffer) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(buffer.ndim));
    for (const auto dimension : buffer.shape) {
        shape.push_back(static_cast<int64_t>(dimension));
    }
    return shape;
}

size_t arraySizeBytes(const py::buffer_info &buffer) {
    return static_cast<size_t>(buffer.size) * static_cast<size_t>(buffer.itemsize);
}

py::dtype numpyDType(vk::Format format, bool allowPackedFormat) {
    vgfutils::numpy::DType dataType;
    if (allowPackedFormat && numComponentsFromVkFormat(format) != 1) {
        dataType = vgfutils::numpy::DType{'V', elementSizeFromVkFormat(format)};
    } else {
        dataType = getDTypeFromVkFormat(format);
    }

    std::string descriptor;
    if (dataType.byteorder != '\0') {
        descriptor.push_back(dataType.byteorder);
    }
    descriptor.push_back(dataType.kind);
    descriptor += std::to_string(dataType.itemsize);
    return py::dtype(descriptor);
}

std::vector<py::ssize_t> numpyShape(const std::vector<int64_t> &shape) {
    std::vector<py::ssize_t> result;
    result.reserve(shape.size());
    for (const auto dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("Resource download returned a negative dimension");
        }
        result.push_back(static_cast<py::ssize_t>(dimension));
    }
    return result;
}

py::array copyToArray(const std::vector<std::byte> &data, const std::vector<int64_t> &shape, const py::dtype &dtype) {
    py::array array(dtype, numpyShape(shape));
    if (static_cast<size_t>(array.nbytes()) != data.size()) {
        throw std::runtime_error("Downloaded resource size does not match its shape and format");
    }
    std::memcpy(array.mutable_data(), data.data(), data.size());
    return array;
}

py::array downloadBuffer(const IScenario &scenario, BufferId id) {
    BufferData result;
    {
        py::gil_scoped_release release;
        result = scenario.download(id);
    }
    py::array_t<uint8_t> array(static_cast<py::ssize_t>(result.data.size()));
    std::memcpy(array.mutable_data(), result.data.data(), result.data.size());
    return array;
}

py::array downloadTensor(const IScenario &scenario, TensorId id) {
    TensorData result;
    {
        py::gil_scoped_release release;
        result = scenario.download(id);
    }
    if (!result.format.has_value()) {
        throw std::runtime_error("Downloaded tensor does not define a format");
    }
    return copyToArray(result.data, result.shape, numpyDType(result.format.value(), false));
}

py::array downloadImage(IScenario &scenario, ImageId id) {
    ImageData result;
    {
        py::gil_scoped_release release;
        result = scenario.download(id);
    }
    if (!result.format.has_value()) {
        throw std::runtime_error("Downloaded image does not define a format");
    }
    return copyToArray(result.data, result.shape, numpyDType(result.format.value(), true));
}

} // namespace
} // namespace mlsdk::scenariorunner

void pyInitIScenario(py::module_ &m) {
    using namespace mlsdk::scenariorunner;

    bindResourceId<BufferId>(m, "BufferId");
    bindResourceId<ImageId>(m, "ImageId");
    bindResourceId<TensorId>(m, "TensorId");
    py::enum_<vk::NeuralAcceleratorStatisticsModeARM>(m, "NeuralAcceleratorStatisticsMode")
        .value("Disabled", vk::NeuralAcceleratorStatisticsModeARM::eDisabled)
        .value("Statistics0", vk::NeuralAcceleratorStatisticsModeARM::eStatistics0)
        .value("Statistics1", vk::NeuralAcceleratorStatisticsModeARM::eStatistics1);

    py::class_<ScenarioOptions>(m, "ScenarioOptions")
        .def(py::init<>())
        .def_readwrite("enable_pipeline_caching", &ScenarioOptions::enablePipelineCaching)
        .def_readwrite("clear_pipeline_cache", &ScenarioOptions::clearPipelineCache)
        .def_readwrite("fail_on_pipeline_cache_miss", &ScenarioOptions::failOnPipelineCacheMiss)
        .def_readwrite("enable_gpu_debug_markers", &ScenarioOptions::enableGPUDebugMarkers)
        .def_readwrite("capture_frame", &ScenarioOptions::captureFrame)
        .def_readwrite("enable_robustness_features", &ScenarioOptions::enableRobustnessFeatures)
        .def_readwrite("pipeline_cache_path", &ScenarioOptions::pipelineCachePath)
        .def_readwrite("neural_debug_database_dump_dir", &ScenarioOptions::neuralDebugDatabaseDumpDir)
        .def_readwrite("neural_statistics_dump_dir", &ScenarioOptions::neuralStatisticsDumpDir)
        .def_readwrite("graph_profiling_dump_dir", &ScenarioOptions::graphProfilingDumpDir)
        .def_readwrite("session_rams_dump_dir", &ScenarioOptions::sessionRAMsDumpDir)
        .def_readwrite("perf_counters_path", &ScenarioOptions::perfCountersPath)
        .def_readwrite("profiling_path", &ScenarioOptions::profilingPath)
        .def_readwrite("neural_statistics_mode", &ScenarioOptions::neuralStatisticsMode)
        .def_readwrite("disabled_extensions", &ScenarioOptions::disabledExtensions);

    py::class_<IScenario, std::unique_ptr<IScenario>>(m, "IScenario")
        .def("run", py::overload_cast<int, bool>(&IScenario::run), py::kw_only(), py::arg("repeat_count") = 1,
             py::arg("dry_run") = false, py::call_guard<py::gil_scoped_release>())
        .def("get_buffer_id", &IScenario::getBufferId)
        .def("get_image_id", &IScenario::getImageId)
        .def("get_tensor_id", &IScenario::getTensorId)
        .def(
            "upload",
            [](IScenario &scenario, BufferId id, const py::array &array) {
                const auto buffer = requireContiguousArray(array);
                const BufferDataView view{buffer.ptr, arraySizeBytes(buffer)};
                py::gil_scoped_release release;
                scenario.upload(id, view);
            },
            py::arg("id"), py::arg("data"))
        .def(
            "upload",
            [](IScenario &scenario, TensorId id, const py::array &array) {
                const auto buffer = requireContiguousArray(array);
                const TensorDataView view{buffer.ptr, arraySizeBytes(buffer), arrayShape(buffer)};
                py::gil_scoped_release release;
                scenario.upload(id, view);
            },
            py::arg("id"), py::arg("data"))
        .def(
            "upload",
            [](IScenario &scenario, ImageId id, const py::array &array) {
                const auto buffer = requireContiguousArray(array);
                const ImageDataView view{buffer.ptr, arraySizeBytes(buffer), arrayShape(buffer), std::nullopt,
                                         /*mipLevels=*/1};
                py::gil_scoped_release release;
                scenario.upload(id, view);
            },
            py::arg("id"), py::arg("data"))
        .def("download", &downloadBuffer, py::arg("id"))
        .def("download", &downloadTensor, py::arg("id"))
        .def("download", &downloadImage, py::arg("id"));
}
