/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "frame_capturer.hpp"

#ifdef ML_SDK_ENABLE_RDOC
// Rdoc implementation
#    include "logging.hpp"
#    include <renderdoc_app.h>
#    ifdef _WIN32
#        include <windows.h>
#    else
#        include <dlfcn.h>
#        include <unistd.h>
#    endif
#    include <cassert>
#    include <stdexcept>

namespace mlsdk::scenariorunner {
FrameCapturer::FrameCapturer() {
    int ret = 0;
#    if defined(_WIN32)
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
        auto RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_0_0, static_cast<void **>(&_captureApiData));
    }
#    elif defined(__linux__)
    if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
        auto RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
        ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_0_0, static_cast<void **>(&_captureApiData));
    }
#    else
#        error "invalid platform"
#    endif
    if (ret == 1) {
        mlsdk::logging::info("Rdoc frame capturer initialised");
    } else {
        _captureApiData = nullptr;
        mlsdk::logging::warning("Failed to initialise Rdoc frame capturer, ignoring");
    }
}

void FrameCapturer::begin() {
    if (_captureApiData == nullptr) {
        return;
    }

    auto *rdocApi = reinterpret_cast<RENDERDOC_API_1_0_0 *>(_captureApiData);

    assert(!rdocApi->IsFrameCapturing());

    rdocApi->StartFrameCapture(nullptr, nullptr);
}

void FrameCapturer::end() {
    if (_captureApiData == nullptr) {
        return;
    }

    auto *rdocApi = reinterpret_cast<RENDERDOC_API_1_0_0 *>(_captureApiData);

    assert(rdocApi->IsFrameCapturing());

    rdocApi->EndFrameCapture(nullptr, nullptr);
}

} // namespace mlsdk::scenariorunner
#else
// Mock implementation
#    include "logging.hpp"

namespace mlsdk::scenariorunner {
FrameCapturer::FrameCapturer() { mlsdk::logging::warning("No frame capturer implementation found, ignoring"); }
void FrameCapturer::begin() {}
void FrameCapturer::end() {}

} // namespace mlsdk::scenariorunner
#endif
