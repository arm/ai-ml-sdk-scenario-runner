/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "iscenario.hpp"
#include "scenario_options.hpp"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <memory>
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
        .def("run", py::overload_cast<int, bool>(&IScenario::run), py::arg("repeat_count") = 1,
             py::arg("dry_run") = false, py::call_guard<py::gil_scoped_release>())
        .def("get_buffer_id", &IScenario::getBufferId)
        .def("get_image_id", &IScenario::getImageId)
        .def("get_tensor_id", &IScenario::getTensorId);
}
