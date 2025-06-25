/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline_cache.hpp"
#include "logging.hpp"

#include <filesystem>
#include <fstream>

namespace mlsdk::scenariorunner {

bool PipelineCache::isValidPipelineCache(const std::vector<unsigned char> &cacheData, uint32_t expectedVendorID,
                                         uint32_t expectedDeviceID) {
    if (cacheData.size() < sizeof(VkPipelineCacheHeaderVersionOne)) {
        return false;
    }
    VkPipelineCacheHeaderVersionOne header{};
    std::memcpy(&header, cacheData.data(), sizeof(header));
    if (header.headerSize != sizeof(VkPipelineCacheHeaderVersionOne)) {
        mlsdk::logging::warning("Pipeline validation: Incorrect pipeline cache header size");
        return false;
    }
    if (header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
        std::ostringstream oss;
        oss << "Pipeline validation: Incorrect pipeline header version (" << header.headerVersion << "). Expected ("
            << VK_PIPELINE_CACHE_HEADER_VERSION_ONE << ")";
        mlsdk::logging::warning(oss.str());
        return false;
    }
    if (header.vendorID != expectedVendorID || header.deviceID != expectedDeviceID) {
        std::ostringstream oss;
        oss << "Pipeline validation: Incorrect device used with cache. (VendorID, DeviceID) = (" << header.vendorID
            << ", " << header.deviceID << "). Expected (" << expectedVendorID << ", " << expectedDeviceID << ")";
        mlsdk::logging::warning(oss.str());
        return false;
    }
    return true;
}

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
        if (size <= 0) {
            mlsdk::logging::warning("Pipeline Cache skipped: size invalid");
        } else {
            _cacheData.resize(size_t(size));
            cacheFile.seekg(dataPos);
            if (!cacheFile.read(reinterpret_cast<char *>(_cacheData.data()), size)) {
                throw std::runtime_error("Failed to read pipeline cache data.");
            }
            cacheFile.close();

            auto props = ctx.physicalDevice().getProperties();
            if (!isValidPipelineCache(_cacheData, props.vendorID, props.deviceID)) {
                mlsdk::logging::warning("Pipeline Cache skipped: failed to validate.");
                _cacheData.clear(); // Fallback to empty
            } else {
                cacheCreateInfo.initialDataSize = _cacheData.size();
                cacheCreateInfo.pInitialData = _cacheData.data();
                mlsdk::logging::info("Pipeline Cache loaded and validated.");
            }
        }
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
