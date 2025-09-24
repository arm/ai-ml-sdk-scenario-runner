/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "context.hpp"
#include "types.hpp"

#include "vulkan/vulkan_raii.hpp"

#include <filesystem>

namespace mlsdk::scenariorunner {

class PipelineCache {
  public:
    explicit PipelineCache(const Context &ctx, const std::filesystem::path &pipelineCachePath, bool clearCache,
                           bool failOnMiss);
    void save();

    const vk::raii::PipelineCache *get() const { return &_pipelineCache; }
    vk::PipelineCreationFeedbackCreateInfo *getCacheFeedbackCreateInfo() { return &_feedbackCreateInfo; }
    bool failOnCacheMiss() const { return _failOnMiss && !_cacheData.empty(); }

  private:
    bool isValidPipelineCache(const std::vector<unsigned char> &cacheData, uint32_t expectedVendorID,
                              uint32_t expectedDeviceID);
    std::filesystem::path _pipelineCachePath{};
    std::vector<uint8_t> _cacheData{};
    vk::raii::PipelineCache _pipelineCache{nullptr};
    vk::PipelineCreationFeedbackCreateInfo _feedbackCreateInfo{};
    vk::PipelineCreationFeedback _feedback{};
    vk::PipelineCreationFeedback _stagedFeedback{};
    bool _failOnMiss{false};
};

} // namespace mlsdk::scenariorunner
