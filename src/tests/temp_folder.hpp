/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <filesystem>
#include <random>
#include <string>

namespace mlsdk::testing {
namespace {
const std::filesystem::path system_temp_folder_path = std::filesystem::temp_directory_path();

std::string randomString() {
    std::string alpha_numeric("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::shuffle(alpha_numeric.begin(), alpha_numeric.end(), generator);
    return alpha_numeric.substr(0, 16);
}

std::filesystem::path make_non_preferred(std::filesystem::path path) {
    std::string path_str = path.generic_string();
    std::replace(path_str.begin(), path_str.end(), '\\', '/');
    return std::filesystem::path(path_str);
}
} // namespace

class TempFolder {
  public:
    template <class Source> explicit TempFolder(const Source &prefix) {
        std::string path_name = prefix + std::string("_") + randomString();
        temp_folder_path = system_temp_folder_path / path_name;
        temp_folder_path = make_non_preferred(temp_folder_path);
        std::filesystem::create_directories(temp_folder_path);
    }
    ~TempFolder() { std::filesystem::remove_all(temp_folder_path); }

    std::filesystem::path &path() { return temp_folder_path; }

    template <class Source> std::filesystem::path relative(const Source &path) const { return temp_folder_path / path; }

  private:
    std::filesystem::path temp_folder_path;
};

} // namespace mlsdk::testing
