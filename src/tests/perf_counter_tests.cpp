/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <thread>

#include "../perf_counter.hpp"

#include <gtest/gtest.h>

using namespace mlsdk::scenariorunner;

TEST(PerformanceCounter, RawStartStopAndReset) {
    PerformanceCounter counter("RawCounter", "UnitTest", true);

    counter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    counter.stop();

    const auto firstElapsed = counter.getElapsedTime();
    ASSERT_GT(firstElapsed, 0);

    counter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    counter.stop();

    ASSERT_GT(counter.getElapsedTime(), firstElapsed);

    counter.reset();
    ASSERT_EQ(counter.getElapsedTime(), 0);
}

TEST(PerformanceCounter, RawMetadata) {
    PerformanceCounter counter("CounterName", "Scenario Setup", false);

    ASSERT_EQ(counter.getName(), "CounterName");
    ASSERT_EQ(counter.getCategory(), "Scenario Setup");
    ASSERT_FALSE(counter.isPartOfTimeToInference());
}

TEST(PerformanceCounter, GuardStartsAndStopsCounter) {
    std::vector<PerformanceCounter> counters;

    {
        PerfCounterGuard guard(counters, "GuardCounter", "UnitTest", false);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    ASSERT_EQ(counters.size(), 1U);
    ASSERT_EQ(counters.front().getName(), "GuardCounter");
    ASSERT_EQ(counters.front().getCategory(), "UnitTest");
    ASSERT_FALSE(counters.front().isPartOfTimeToInference());
    ASSERT_GT(counters.front().getElapsedTime(), 0);
}
