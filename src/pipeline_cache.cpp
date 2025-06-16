/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline_cache.hpp"
#include "logging.hpp"

#include <filesystem>
#include <fstream>

namespace mlsdk::scenariorunner {

PipelineCache::PipelineCache(Context &ctx, const std::filesystem::path &pipelineCachePath, bool clearCache,
                             bool failOnMiss)
    : _pipelineCachePath(pipelineCachePath), _failOnMiss(failOnMiss) {
    vk::PipelineCacheCreateInfo cacheCreateInfo;
    cacheCreateInfo.sType = vk::StructureType::ePipelineCacheCreateInfo;
    cacheCreateInfo.flags = vk::PipelineCacheCreateFlagBits::eExternallySynchronized;

    if (clearCache) {
        std::filesystem::remove(pipelineCachePath);
        mlsdk::logging::info("Pipeline Cache cleared");
    } else if (std::filesystem::exists(pipelineCachePath)) {
        // Use cache file, if existing
        std::ifstream cacheFile(pipelineCachePath.string(), std::ifstream::binary);
        if (!cacheFile.is_open()) {
            throw std::runtime_error("Could not read from Pipeline Cache file: " + _pipelineCachePath.string());
        }
        cacheFile.exceptions(std::ios::badbit | std::ios::failbit);

        const auto dataPos = cacheFile.tellg();
        cacheFile.seekg(0, std::ios::end);
        const auto size = cacheFile.tellg() - dataPos;
        _cacheData.resize(size_t(size));
        cacheFile.seekg(dataPos);
        cacheFile.read(reinterpret_cast<char *>(_cacheData.data()), size);
        cacheFile.close();

        cacheCreateInfo.initialDataSize = _cacheData.size();
        cacheCreateInfo.pInitialData = _cacheData.data();
        mlsdk::logging::info("Pipeline Cache loaded");
    }

    _pipelineCache = vk::raii::PipelineCache(ctx.device(), cacheCreateInfo);

    _feedback = vk::PipelineCreationFeedback(vk::PipelineCreationFeedbackFlagBits::eValid, 0);

    _stagedFeedback = vk::PipelineCreationFeedback(vk::PipelineCreationFeedbackFlagBits::eValid, 0);
    _feedbackCreateInfo = vk::PipelineCreationFeedbackCreateInfo(&_feedback, 1, &_stagedFeedback);
}

void PipelineCache::save() {
    if (failOnCacheMiss()) {
        mlsdk::logging::info("Pipeline Cache not stored");
        return;
    }
    // Save updated cache to disk
    std::ofstream fstream(_pipelineCachePath.string(), std::ofstream::binary | std::ofstream::trunc);
    if (!fstream.is_open()) {
        throw std::runtime_error("Error storing pipeline cache into: " + _pipelineCachePath.string());
    }
    fstream.exceptions(std::ios::badbit | std::ios::failbit);
    fstream.write(reinterpret_cast<char *>(_pipelineCache.getData().data()),
                  std::streamsize(_pipelineCache.getData().size()));
    fstream.close();
    mlsdk::logging::info("Pipeline Cache stored");
}

} // namespace mlsdk::scenariorunner
