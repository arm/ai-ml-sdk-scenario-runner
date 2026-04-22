/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <argparse/argparse.hpp>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "png_reader.hpp"
#include "vgf-utils/numpy.hpp"

using namespace mlsdk::scenariorunner;

namespace {

void generatePNGFile(uint32_t height, uint32_t width, const std::string &output) {
    // Trust underlying function for validating dimensions and size.
    const auto dataSize = static_cast<size_t>(static_cast<uint64_t>(width) * height * 4);
    const std::vector<char> data(dataSize, 0);
    const std::vector<int64_t> shape = {1, height, width, 4};
    ImageSaveOptions options{shape, vk::Format::eR8G8B8A8Unorm, data};
    saveDataToPNG(output, options);
}

ImageLoadResult load(const std::string &input) {
    auto result = loadDataFromPNG(input, {});
    if (result.initialFormat != vk::Format::eR8G8B8A8Unorm) {
        throw std::runtime_error("Unsupported decoded PNG format");
    }

    return result;
}

void convertPNGToNpy(const std::string &input, const std::string &output) {
    using namespace mlsdk::vgfutils;
    const auto result = load(input);
    const std::vector<int64_t> shape = {1, result.height, result.width, 4};
    numpy::DataPtr npData(reinterpret_cast<const char *>(result.data.data()), shape, numpy::DType('u', 1));
    numpy::write(output, npData);
}

bool compare(const std::string &input, const std::string &output) {
    const auto lhs = load(input);
    const auto rhs = load(output);
    if (lhs.width != rhs.width || lhs.height != rhs.height) {
        std::cerr << "PNG dimensions do not match." << std::endl;
        return false;
    }
    const auto sameData = lhs.data == rhs.data;
    if (!sameData) {
        std::cerr << "PNG data does not match." << std::endl;
    }
    return sameData;
}

} // namespace

int main(int argc, const char **argv) {
    try {
        argparse::ArgumentParser parser(argv[0]);

        parser.add_argument("--action")
            .choices("generate", "to_npy", "compare")
            .help("Required action: generate, to_npy, or compare")
            .required();

        parser.add_argument("--height").scan<'i', uint32_t>().help("Image height");
        parser.add_argument("--width").scan<'i', uint32_t>().help("Image width");
        parser.add_argument("--input").help("Path to the input PNG file");
        parser.add_argument("--output").help("Path to the output file");

        parser.parse_args(argc, argv);
        auto action = parser.get<std::string>("--action");

        if (action == "generate") {
            auto height = parser.get<uint32_t>("--height");
            auto width = parser.get<uint32_t>("--width");
            auto output = parser.get<std::string>("--output");

            generatePNGFile(height, width, output);
        } else if (action == "to_npy") {
            auto input = parser.get<std::string>("--input");
            auto output = parser.get<std::string>("--output");

            convertPNGToNpy(input, output);
        } else if (action == "compare") {
            auto input = parser.get<std::string>("--input");
            auto output = parser.get<std::string>("--output");

            bool same = compare(input, output);
            if (same) {
                std::cout << "PNG files match." << std::endl;
            }
            return same ? 0 : 1;
        } else {
            throw std::runtime_error("Unsupported action: " + action);
        }
    } catch (const std::exception &error) {
        std::cerr << "[ERROR]: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
