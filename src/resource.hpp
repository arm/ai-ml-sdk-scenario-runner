/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace mlsdk::scenariorunner {

class Resource {
  public:
    virtual void store(Context &ctx, const std::string &filename) = 0;
    virtual ~Resource() = default;
};

} // namespace mlsdk::scenariorunner
