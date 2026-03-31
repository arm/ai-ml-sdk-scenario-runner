/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <argparse/argparse.hpp>
#include <cstdint>
#include <fstream>
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
    if (height == 0 || width == 0 || width > static_cast<uint64_t>(std::numeric_limits<int>::max() / 4) ||
        height > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        height > (static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / 4) / width) {
        throw std::runtime_error("Image dimensions exceed supported limits");
    }

    std::vector<uint8_t> data(static_cast<size_t>(static_cast<uint64_t>(width) * height * 4), 0);

    if (stbi_write_png(output.c_str(), static_cast<int>(width), static_cast<int>(height), 4, data.data(),
                       static_cast<int>(width) * 4) == 0) {
        throw std::runtime_error("Failed to write PNG: " + output);
    }
}

struct PNG {
    int width;
    int height;
    std::vector<uint8_t> data;
};

PNG load(const std::string &input) {
    auto result = loadDataFromPNG(input, {});
    if (result.initialFormat != vk::Format::eR8G8B8A8Unorm) {
        throw std::runtime_error("Unsupported decoded PNG format");
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_info(input.c_str(), &width, &height, &channels)) {
        throw std::runtime_error("Failed to inspect PNG: " + input);
    }
    return {width, height, std::move(result.data)};
}

void convertPNGToNpy(const std::string &input, const std::string &output) {
    PNG png = load(input);
    mlsdk::vgfutils::numpy::DataPtr npData(reinterpret_cast<char *>(png.data.data()), {1, png.height, png.width, 4},
                                           mlsdk::vgfutils::numpy::DType('u', 1));
    mlsdk::vgfutils::numpy::write(output, npData);
}

bool compare(const std::string &input, const std::string &output) {
    try {
        PNG lhs = load(input);
        PNG rhs = load(output);
        if (lhs.width != rhs.width || lhs.height != rhs.height) {
            return false;
        }
        return lhs.data == rhs.data;
    } catch (...) {
        return false;
    }
}

} // namespace

int main(int argc, const char **argv) {
    try {
        argparse::ArgumentParser parser(argv[0]);

        parser.add_argument("--action").choices("generate", "to_npy", "compare").help("Required action").required();

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
