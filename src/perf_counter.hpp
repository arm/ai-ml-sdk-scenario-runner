/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {

class PerformanceCounter {
  public:
    explicit PerformanceCounter(const std::string &name, const std::string &category = "",
                                bool isPartOfTimeToInference = false)
        : _startTimePoint(std::chrono::time_point<Clock>::min()), _elapsedTime(0), _name(name), _category(category),
          _isPartOfTimeToInference(isPartOfTimeToInference) {}

    void start() { _startTimePoint = Clock::now(); }

    void stop() {
        _elapsedTime += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - _startTimePoint);
    }

    void reset() {
        _elapsedTime = std::chrono::microseconds(0);
        _startTimePoint = std::chrono::time_point<Clock>::min();
    }

    int64_t getElapsedTime() const { return _elapsedTime.count(); }
    const std::string &getName() const { return _name; }
    const std::string &getCategory() const { return _category; }
    bool isPartOfTimeToInference() const { return _isPartOfTimeToInference; }

  private:
    using Clock = std::chrono::high_resolution_clock;

    std::chrono::time_point<Clock> _startTimePoint;
    std::chrono::microseconds _elapsedTime;
    std::string _name;
    std::string _category;
    bool _isPartOfTimeToInference;
};

struct AggregateStat {
    AggregateStat() = default;
    explicit AggregateStat(const std::string &name) : name(name), aggregateTime(0) {}

    std::string name;
    int64_t aggregateTime = 0;
    std::vector<PerformanceCounter> counters;
};

} // namespace mlsdk::scenariorunner
