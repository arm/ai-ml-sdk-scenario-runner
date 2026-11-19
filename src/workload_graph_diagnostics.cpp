/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "workload_graph_diagnostics.hpp"

#include "graph_dump.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {
namespace {

// Keep in sync with the optional property defined by emulation-layer.
constexpr auto graphProfilingProperty = static_cast<vk::DataGraphPipelinePropertyARM>(0x7ffffffe);

std::vector<uint8_t> propertyData(const workloadlib::Session &session, uint32_t executableIndex,
                                  vk::DataGraphPipelinePropertyARM property) {
    const auto properties = session.dataGraphPipelineProperties(executableIndex);
    if (std::find(properties.begin(), properties.end(), property) == properties.end()) {
        return {};
    }
    return session.dataGraphPipelineProperty(executableIndex, property).data;
}

std::vector<uint8_t> readSessionMemory(const vk::raii::Device &device,
                                       const workloadlib::DataGraphSessionMemoryInfo &memory) {
    if (!(memory.memoryProperties & vk::MemoryPropertyFlagBits::eHostVisible)) {
        throw std::runtime_error("Graph session memory must be host-visible for dumping");
    }
    std::vector<uint8_t> data(memory.size);
    const auto *mapped = (*device).mapMemory(memory.memory, 0, vk::WholeSize, {}, *device.getDispatcher());
    ScopeExit<void()> unmap([&] { (*device).unmapMemory(memory.memory, *device.getDispatcher()); });
    if (!(memory.memoryProperties & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        device.invalidateMappedMemoryRanges({vk::MappedMemoryRange(memory.memory, 0, vk::WholeSize)});
    }
    std::memcpy(data.data(), mapped, data.size());
    return data;
}

} // namespace

bool dumpWorkloadGraphDebugDatabase(const workloadlib::Session &session, uint32_t executableIndex,
                                    const std::filesystem::path &path) {
    const auto data =
        propertyData(session, executableIndex, vk::DataGraphPipelinePropertyARM::eNeuralAcceleratorDebugDatabase);
    if (data.empty()) {
        return false;
    }
    if (writeGraphBinaryDump(path, data.data(), data.size())) {
        logging::info("Graph debug database stored");
    }
    return true;
}

bool dumpWorkloadGraphStatistics(const workloadlib::Session &session, uint32_t executableIndex,
                                 const vk::raii::Device &device, const std::filesystem::path &path) {
    if (propertyData(session, executableIndex, vk::DataGraphPipelinePropertyARM::eNeuralAcceleratorStatisticsInfo)
            .empty()) {
        return false;
    }
    std::optional<workloadlib::DataGraphSessionMemoryInfo> statistics;
    for (uint32_t i = 0; i < session.dataGraphSessionMemoryCount(executableIndex); ++i) {
        const auto memory = session.dataGraphSessionMemory(executableIndex, i);
        if (memory.neuralStatistics) {
            if (statistics) {
                throw std::runtime_error("More than one session memory instance for Neural Statistics");
            }
            statistics = memory;
        }
    }
    if (!statistics) {
        logging::warning("Expected one session memory instance for Neural Statistics per pipeline, but found none. "
                         "Neural Statistics file will not be saved.");
        return false;
    }
    const auto data = readSessionMemory(device, *statistics);
    if (writeGraphBinaryDump(path, data.data(), data.size())) {
        logging::info("Neural statistics saved to \"" + path.filename().string() + "\"");
    }
    return true;
}

void dumpWorkloadGraphProfiling(const workloadlib::Session &session, uint32_t executableIndex,
                                const std::filesystem::path &directory, uint32_t graphPipelineIndex) {
    const auto index = std::to_string(graphPipelineIndex);
    std::vector<uint8_t> data;
    try {
        data = propertyData(session, executableIndex, graphProfilingProperty);
    } catch (const std::exception &error) {
        logging::warning("Skipping graph profiling dump for graph pipeline " + index +
                         ": failed to query property data: " + error.what());
        return;
    }
    if (data.empty()) {
        logging::warning("Skipping graph profiling dump for graph pipeline " + index +
                         ": property 0x7ffffffe is unavailable or empty");
        return;
    }
    const auto filename = "EmulationLayerPipeline_" + index + ".json";
    if (writeGraphBinaryDump(directory / filename, data.data(), data.size())) {
        logging::info("Graph profiling data saved to \"" + filename + "\"");
    }
}

void dumpWorkloadGraphSessionRAMs(const workloadlib::Session &session, uint32_t executableIndex,
                                  const vk::raii::Device &device, const std::filesystem::path &directory,
                                  uint32_t &graphPipelineIndex) {
    bool allStored = true;
    for (uint32_t i = 0; i < session.dataGraphSessionMemoryCount(executableIndex); ++i) {
        const auto data = readSessionMemory(device, session.dataGraphSessionMemory(executableIndex, i));
        // Preserve SR's existing dump naming, including its allocation counter.
        const auto filename =
            "Graph_Pipeline_" + std::to_string(graphPipelineIndex++) + "_Session_RAM_" + std::to_string(i) + ".txt";
        allStored = writeGraphSessionRAMDump(directory / filename, data.data(), data.size()) && allStored;
    }
    if (allStored) {
        logging::info("Session RAM dump stored");
    }
}

} // namespace mlsdk::scenariorunner
