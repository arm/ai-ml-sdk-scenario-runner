/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "graph_dump.hpp"

#include "logging.hpp"

#include <fstream>
#include <iomanip>

namespace mlsdk::scenariorunner {
namespace {

bool closeDumpFile(std::ofstream &file, const std::filesystem::path &path) {
    // Closing also checks buffered writes, which may fail after write() succeeds.
    file.close();
    if (!file) {
        logging::warning("Failed to write graph dump to \"" + path.string() + "\"");
        return false;
    }
    return true;
}

} // namespace

bool writeGraphBinaryDump(const std::filesystem::path &path, const void *data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (file) {
        file.write(static_cast<const char *>(data), std::streamsize(size));
    }
    return closeDumpFile(file, path);
}

bool writeGraphSessionRAMDump(const std::filesystem::path &path, const unsigned char *data, size_t size) {
    std::ofstream file(path);
    if (file) {
        file << std::hex << std::uppercase << std::setfill('0');
        for (size_t i = 0; i < size; ++i) {
            if (i % 16 == 0) {
                file << std::endl << std::setw(8) << i << ":   ";
            }
            file << std::setw(2) << static_cast<unsigned>(data[i]) << " ";
        }
    }
    return closeDumpFile(file, path);
}

} // namespace mlsdk::scenariorunner
