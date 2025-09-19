/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "json_writer.hpp"

#include <fstream>

#include "nlohmann/json.hpp"

namespace mlsdk::scenariorunner {
using json = nlohmann::json;

namespace {
json _profilingDataJsonOutput;

struct CommandTimestamps {
    CommandTimestamps() = default;
    CommandTimestamps(const std::string &commandType, const std::vector<uint64_t> &commandTimestamps,
                      const float timestampPeriod, const int iteration = 1)
        : type(commandType), timestamps(commandTimestamps), period(timestampPeriod), iteration(iteration) {}

    std::string type;
    std::vector<uint64_t> timestamps;
    float period{};
    int iteration{};
};

void to_json(json &j, const CommandTimestamps &commandTimestamps) {
    j = json{{"Command type", commandTimestamps.type},
             {"Cycle count before command", commandTimestamps.timestamps[0]},
             {"Cycle count after command", commandTimestamps.timestamps[1]},
             {"Cycle count for command", commandTimestamps.timestamps[1] - commandTimestamps.timestamps[0]},
             {"Timestamp Period", commandTimestamps.period},
             {"Time for command [ms]", float(commandTimestamps.timestamps[1] - commandTimestamps.timestamps[0]) *
                                           commandTimestamps.period / 1000000.0f},
             {"Iteration", commandTimestamps.iteration + 1}};
}

} // namespace

void to_json(json &j, const PerformanceCounter &perfCounter) {
    j = json{{"name", perfCounter.getName()}, {"value", perfCounter.getElapsedTime()}, {"unit", "microseconds"}};
}

void to_json(json &j, const AggregateStat &stat) {
    j = json{{"total time", stat.aggregateTime}, {"unit", "microseconds"}, {"counters", stat.counters}};
}

void writePerfCounters(std::vector<PerformanceCounter> &perfCounters, std::filesystem::path &path) {
    std::map<std::string, AggregateStat> map;
    int64_t timeToInference = 0;
    int64_t scenarioaggregate = 0;
    json outJson;

    for (auto &pc : perfCounters) {
        const auto &category = pc.getCategory();
        if (!map.count(category)) {
            map[category] = AggregateStat(category);
        }

        if (pc.isPartOfTimeToInference()) {
            timeToInference += pc.getElapsedTime();
        }

        scenarioaggregate += pc.getElapsedTime();
        map[category].aggregateTime += pc.getElapsedTime();
        map[category].counters.push_back(pc);
    }

    // Write aggregated time for the whole scenario
    outJson["Time to Inference"] = timeToInference;
    outJson["Total Scenario Time"] = scenarioaggregate;
    outJson["unit"] = "microseconds";

    // Write aggregated stats for categories
    for (const auto &stat : map) {
        if (!stat.first.empty())
            outJson[stat.first] = stat.second;
    }

    // Unaggregated stats not part of a category
    for (const auto &stat : map) {
        if (stat.first.empty()) {
            outJson["Uncategorized"] = stat.second.counters;
            break;
        }
    }

    std::ofstream ostream(path);
    ostream << outJson.dump(4);
    ostream.close();
}

void writeProfilingData(const std::vector<uint64_t> &timestamps, const float timestampPeriod,
                        const std::vector<std::string> &profiledCommands, const std::filesystem::path &path,
                        const int iteration, const int repeatCount) {
    if (profiledCommands.size() * 2 != timestamps.size()) {
        throw std::runtime_error("Cannot map all timestamps to their respective commands");
    }
    for (size_t idx = 0, commandIdx = 0; idx < timestamps.size(); idx += 2, ++commandIdx) {
        _profilingDataJsonOutput["Timestamps"] += CommandTimestamps(
            profiledCommands[commandIdx], {timestamps[idx], timestamps[idx + 1]}, timestampPeriod, iteration);
    }
    // Check if this is the last iteration
    if (iteration + 1 == repeatCount) {
        std::ofstream dumpFile(path.string());
        dumpFile << _profilingDataJsonOutput.dump(4);
        dumpFile.close();
    }
}

} // namespace mlsdk::scenariorunner
