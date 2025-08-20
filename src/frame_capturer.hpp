/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace mlsdk::scenariorunner {
class FrameCapturer {
  public:
    FrameCapturer();
    void begin();
    void end();
#ifdef ML_SDK_ENABLE_RDOC
  private:
    void *_captureApiData{nullptr};
#endif
};

} // namespace mlsdk::scenariorunner
