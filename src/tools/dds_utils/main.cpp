/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <argparse/argparse.hpp>
#include <version.hpp>

#include "dds_reader.hpp"
#include "vgf-utils/numpy.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

using namespace mlsdk::scenariorunner;

using DDSContent = std::pair<DDSHeaderInfo, ImageLoadResult>;

std::vector<uint16_t> createRandomFloat16Data(uint32_t size) {
    std::vector<uint16_t> vec(size);
    std::random_device randomDevice;
    std::mt19937 mtGenerator(randomDevice());
    std::uniform_int_distribution<> distribution;
    for (uint32_t i = 0; i < size; i++) {
        vec[i] = static_cast<uint16_t>(static_cast<uint8_t>(distribution(mtGenerator) % 255) << 8);

        // Randomly choose an exponent bit to ensure no NaN values generated
        auto x = static_cast<uint8_t>(0b01000000 >> (distribution(mtGenerator) % 5));
        vec[i] = vec[i] | static_cast<uint16_t>(static_cast<uint8_t>(distribution(mtGenerator) % 255) & ~x);
    }
    return vec;
}

#define DXGI_FORMAT_ENTRY(format)                                                                                      \
    std::pair<std::string_view, DxgiFormat> { #format, format }

DxgiFormat getDxgiFormat(const std::string &format) {
    constexpr std::pair<std::string_view, DxgiFormat> supportedFormats[] = {
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16B16A16_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R11G11B10_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_D32_FLOAT_S8X24_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8B8A8_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32B32A32_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B8G8R8A8_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8B8A8_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16_UINT),
    };

    for (const auto &entry : supportedFormats) {
        if (format == entry.first) {
            return entry.second;
        }
    }

#ifdef SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT
    // These are image formats that haven't been fully tested yet.
    constexpr std::pair<std::string_view, DxgiFormat> experimentalFormats[] = {
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32B32A32_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32B32A32_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32B32_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32B32_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32B32_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32B32_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16B16A16_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16B16A16_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16B16A16_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16B16A16_SNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G32_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32G8X24_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R10G10B10A2_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R10G10B10A2_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R10G10B10A2_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8B8A8_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8B8A8_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16_SNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16G16_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_D32_FLOAT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R32_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R24G8_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_D24_UNORM_S8_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R24_UNORM_X8_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_X24_TYPELESS_G8_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8_SNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_D16_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16_SNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R16_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8_UINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8_SINT),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_A8_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R1_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R9G9B9E5_SHAREDEXP),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R8G8_B8G8_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_G8R8_G8B8_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC1_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC1_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC1_UNORM_SRGB),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC2_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC2_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC2_UNORM_SRGB),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC3_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC3_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC3_UNORM_SRGB),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC4_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC4_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC4_SNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC5_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC5_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC5_SNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B5G6R5_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B5G5R5A1_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B8G8R8X8_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B8G8R8A8_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B8G8R8X8_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B8G8R8X8_UNORM_SRGB),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC6H_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC6H_UF16),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC6H_SF16),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC7_TYPELESS),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC7_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_BC7_UNORM_SRGB),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_AYUV),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_Y410),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_Y416),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_NV12),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_P010),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_P016),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_420_OPAQUE),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_YUY2),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_Y210),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_Y216),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_NV11),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_AI44),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_IA44),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_P8),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_A8P8),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_B4G4R4A4_UNORM),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_P208),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_V208),
        DXGI_FORMAT_ENTRY(DXGI_FORMAT_V408),
    };

    for (const auto &entry : experimentalFormats) {
        if (format == entry.first) {
            return entry.second;
        }
    }
#endif

    throw std::runtime_error("Unsupported format " + format);
}

#undef DXGI_FORMAT_ENTRY

void generateDDSFile(uint32_t height, uint32_t width, const std::string &elementDtype, uint32_t elementSize,
                     const std::string &format, const std::string &output, bool headerOnly) {
    DDSHeaderInfo header = generateDefaultDDSHeader(height, width, elementSize, getDxgiFormat(format));

    std::ofstream fstream(output, std::ofstream::binary);
    saveHeaderToDDS(header, fstream);

    if (!headerOnly) {
        auto s = header.header.height * header.header.width;

        std::vector<uint8_t> testData;
        if (elementDtype == "fp16") {
            std::vector<uint16_t> data =
                createRandomFloat16Data(s * static_cast<uint32_t>(elementSize / sizeof(uint16_t)));
            auto *buffer = reinterpret_cast<uint8_t *>(data.data());
            auto sizeInBytes = data.size() * sizeof(uint16_t);
            testData = std::vector<uint8_t>(buffer, buffer + sizeInBytes);
        } else {
            testData = std::vector<uint8_t>(s * elementSize);
        }

        fstream.write(reinterpret_cast<char *>(testData.data()), std::streamsize(testData.size()));
    }

    fstream.close();
}

DDSContent load(const std::string &ddsPath) {
    std::ifstream ddsFile(ddsPath, std::ifstream::binary);
    DDSHeaderInfo ddsHeader = readDDSHeader(ddsFile);

    auto result = loadDataFromDDS(ddsPath, {});
    return {ddsHeader, std::move(result)};
}

void convertDDSToNpy(const std::string &input, const std::string &output, uint32_t elementSize) {
    using namespace mlsdk::vgfutils;

    const auto &[ddsHeader, ddsResult] = load(input);
    auto imageElementSize = ddsHeader.header.pitchOrLinearSize / ddsResult.width;

    if (elementSize > imageElementSize) {
        throw std::runtime_error("The image cannot be coverted to a NumPy file with bigger element size");
    }

    const std::vector<int64_t> shape = {1, ddsResult.height, ddsResult.width, imageElementSize / elementSize};
    numpy::DataPtr data(reinterpret_cast<const char *>(ddsResult.data.data()), shape, numpy::DType('i', elementSize));
    numpy::write(output, data);
}

bool compareDDSHeader(const DDSHeaderInfo &input, const DDSHeaderInfo &output) {
    return input.header.width == output.header.width && input.header.height == output.header.height &&
           input.header.depth == output.header.depth && input.header10.dxgiFormat == output.header10.dxgiFormat;
}

bool isFloat16NaN(uint16_t val) {
    constexpr uint16_t fP16Exponent = 0x7C00;
    return (val & fP16Exponent) == fP16Exponent;
}

bool compare(const std::string &input, const std::string &output, const std::string &elementDtype) {
    const auto &[inputHeader, inputResult] = load(input);
    const auto &[outputHeader, outputResult] = load(output);

    bool sameHeader = compareDDSHeader(inputHeader, outputHeader);
    if (!sameHeader) {
        std::cerr << "DDS headers do not match." << std::endl;
    }

    bool sameData = true;
    if (elementDtype == "fp16") {
        const auto *inputFp16 = reinterpret_cast<const uint16_t *>(inputResult.data.data());
        const auto *outputFp16 = reinterpret_cast<const uint16_t *>(outputResult.data.data());

        auto size = inputResult.data.size() / sizeof(uint16_t);
        for (size_t i = 0; i < size; i++) {
            if (isFloat16NaN(inputFp16[i]) && isFloat16NaN(outputFp16[i])) {
                continue;
            }

            if (inputFp16[i] != outputFp16[i]) {
                sameData = false;
                break;
            }
        }
    } else {
        sameData = inputResult.data == outputResult.data;
        if (!sameData) {
            std::cerr << "DDS data does not match." << std::endl;
        }
    }

    return sameHeader && sameData;
}

int main(int argc, const char **argv) {
    try {
        argparse::ArgumentParser parser(argv[0], details::version);

        parser.add_argument("--action")
            .choices("generate", "to_npy", "compare")
            .help("Required action: generate, to_npy, or compare")
            .required();

        parser.add_argument("--height").scan<'i', uint32_t>().help("Image height");
        parser.add_argument("--width").scan<'i', uint32_t>().help("Image width");
        parser.add_argument("--element-size").scan<'i', uint32_t>().help("Element size");
        parser.add_argument("--element-dtype").help("Element data type");
        parser.add_argument("--format").help("DXGI format");
        parser.add_argument("--header-only").default_value(false).implicit_value(true).help("Write DDS header only");
        parser.add_argument("--input").help("Path to the input file");
        parser.add_argument("--output").help("Path to the output file");

        parser.parse_args(argc, argv);
        auto action = parser.get<std::string>("--action");

        if (action == "generate") {
            auto height = parser.get<uint32_t>("--height");
            auto width = parser.get<uint32_t>("--width");
            auto elementSize = parser.get<uint32_t>("--element-size");
            auto elementDtype = parser.get<std::string>("--element-dtype");
            auto dxgiFormat = parser.get<std::string>("--format");
            auto output = parser.get<std::string>("--output");
            bool headerOnly = parser.get<bool>("--header-only");

            generateDDSFile(height, width, elementDtype, elementSize, dxgiFormat, output, headerOnly);
        } else if (action == "to_npy") {
            auto input = parser.get<std::string>("--input");
            auto output = parser.get<std::string>("--output");
            auto elementSize = parser.get<uint32_t>("--element-size");

            convertDDSToNpy(input, output, elementSize);
        } else if (action == "compare") {
            auto input = parser.get<std::string>("--input");
            auto output = parser.get<std::string>("--output");
            auto elementDtype = parser.get<std::string>("--element-dtype");

            bool same = compare(input, output, elementDtype);
            if (same) {
                std::cout << "DDS files match." << std::endl;
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
