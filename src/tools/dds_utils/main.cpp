/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <argparse/argparse.hpp>

#include "dds_reader.hpp"
#include "vgf-utils/numpy.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace mlsdk::scenariorunner;

using DDSContent = std::pair<DDSHeaderInfo, std::vector<uint8_t>>;

std::vector<uint16_t> createRandomFloat16Data(uint32_t size) {
    std::vector<uint16_t> vec(size);
    std::random_device randomDevice;
    std::mt19937 mtGenerator(randomDevice());
    std::uniform_int_distribution<> distribution;
    for (uint32_t i = 0; i < size; i++) {
        vec[i] = static_cast<uint16_t>(static_cast<uint8_t>(distribution(mtGenerator) % 255) << 8);

        // Randomly choose an exponent bit to ensure no NaN values generated
        uint8_t x = static_cast<uint8_t>(0b01000000 >> (distribution(mtGenerator) % 5));
        vec[i] = vec[i] | static_cast<uint16_t>(static_cast<uint8_t>(distribution(mtGenerator) % 255) & ~x);
    }
    return vec;
}

DxgiFormat getDxgiFormat(const std::string &format) {
    if (format == "DXGI_FORMAT_R16_FLOAT") {
        return DXGI_FORMAT_R16_FLOAT;
    }

    if (format == "DXGI_FORMAT_R16G16_FLOAT") {
        return DXGI_FORMAT_R16G16_FLOAT;
    }

    if (format == "DXGI_FORMAT_R8G8_SINT") {
        return DXGI_FORMAT_R8G8_SINT;
    }

    if (format == "DXGI_FORMAT_R32_FLOAT") {
        return DXGI_FORMAT_R32_FLOAT;
    }

    if (format == "DXGI_FORMAT_R16G16B16A16_FLOAT") {
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    }

    if (format == "DXGI_FORMAT_D32_FLOAT_S8X24_UINT") {
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    }

    if (format == "DXGI_FORMAT_R8G8B8A8_SINT") {
        return DXGI_FORMAT_R8G8B8A8_SINT;
    }

    if (format == "DXGI_FORMAT_R32G32B32A32_FLOAT") {
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    }

    if (format == "DXGI_FORMAT_R32_UINT") {
        return DXGI_FORMAT_R32_UINT;
    }

#ifdef SCENARIO_RUNNER_EXPERIMENTAL_IMAGE_FORMAT_SUPPORT
    // These are image formats that haven't been fully tested yet.
    if (format == "DXGI_FORMAT_R32G32B32A32_UINT") {
        return DXGI_FORMAT_R32G32B32A32_UINT;
    }
    if (format == "DXGI_FORMAT_R32G32B32A32_SINT") {
        return DXGI_FORMAT_R32G32B32A32_SINT;
    }
    if (format == "DXGI_FORMAT_R32G32B32_TYPELESS") {
        return DXGI_FORMAT_R32G32B32_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R32G32B32_FLOAT") {
        return DXGI_FORMAT_R32G32B32_FLOAT;
    }
    if (format == "DXGI_FORMAT_R32G32B32_UINT") {
        return DXGI_FORMAT_R32G32B32_UINT;
    }
    if (format == "DXGI_FORMAT_R32G32B32_SINT") {
        return DXGI_FORMAT_R32G32B32_SINT;
    }
    if (format == "DXGI_FORMAT_R16G16B16A16_TYPELESS") {
        return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R16G16B16A16_UNORM") {
        return DXGI_FORMAT_R16G16B16A16_UNORM;
    }
    if (format == "DXGI_FORMAT_R16G16B16A16_UINT") {
        return DXGI_FORMAT_R16G16B16A16_UINT;
    }
    if (format == "DXGI_FORMAT_R16G16B16A16_SNORM") {
        return DXGI_FORMAT_R16G16B16A16_SNORM;
    }
    if (format == "DXGI_FORMAT_R32G32_TYPELESS") {
        return DXGI_FORMAT_R32G32_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R32G32_FLOAT") {
        return DXGI_FORMAT_R32G32_FLOAT;
    }
    if (format == "DXGI_FORMAT_R32G32_UINT") {
        return DXGI_FORMAT_R32G32_UINT;
    }
    if (format == "DXGI_FORMAT_R32G32_SINT") {
        return DXGI_FORMAT_R32G32_SINT;
    }
    if (format == "DXGI_FORMAT_R32G8X24_TYPELESS") {
        return DXGI_FORMAT_R32G8X24_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS") {
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    }
    if (format == "DXGI_FORMAT_X32_TYPELESS_G8X24_UINT") {
        return DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    }
    if (format == "DXGI_FORMAT_R10G10B10A2_TYPELESS") {
        return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R10G10B10A2_UNORM") {
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    }
    if (format == "DXGI_FORMAT_R10G10B10A2_UINT") {
        return DXGI_FORMAT_R10G10B10A2_UINT;
    }
    if (format == "DXGI_FORMAT_R8G8B8A8_TYPELESS") {
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB") {
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    }
    if (format == "DXGI_FORMAT_R8G8B8A8_UINT") {
        return DXGI_FORMAT_R8G8B8A8_UINT;
    }
    if (format == "DXGI_FORMAT_R16G16_TYPELESS") {
        return DXGI_FORMAT_R16G16_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R16G16_UNORM") {
        return DXGI_FORMAT_R16G16_UNORM;
    }
    if (format == "DXGI_FORMAT_R16G16_UINT") {
        return DXGI_FORMAT_R16G16_UINT;
    }
    if (format == "DXGI_FORMAT_R16G16_SNORM") {
        return DXGI_FORMAT_R16G16_SNORM;
    }
    if (format == "DXGI_FORMAT_R16G16_SINT") {
        return DXGI_FORMAT_R16G16_SINT;
    }
    if (format == "DXGI_FORMAT_R32_TYPELESS") {
        return DXGI_FORMAT_R32_TYPELESS;
    }
    if (format == "DXGI_FORMAT_D32_FLOAT") {
        return DXGI_FORMAT_D32_FLOAT;
    }
    if (format == "DXGI_FORMAT_R32_SINT") {
        return DXGI_FORMAT_R32_SINT;
    }
    if (format == "DXGI_FORMAT_R24G8_TYPELESS") {
        return DXGI_FORMAT_R24G8_TYPELESS;
    }
    if (format == "DXGI_FORMAT_D24_UNORM_S8_UINT") {
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
    }
    if (format == "DXGI_FORMAT_R24_UNORM_X8_TYPELESS") {
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    }
    if (format == "DXGI_FORMAT_X24_TYPELESS_G8_UINT") {
        return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    }
    if (format == "DXGI_FORMAT_R8G8_TYPELESS") {
        return DXGI_FORMAT_R8G8_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R8G8_SNORM") {
        return DXGI_FORMAT_R8G8_SNORM;
    }
    if (format == "DXGI_FORMAT_R16_TYPELESS") {
        return DXGI_FORMAT_R16_TYPELESS;
    }
    if (format == "DXGI_FORMAT_D16_UNORM") {
        return DXGI_FORMAT_D16_UNORM;
    }
    if (format == "DXGI_FORMAT_R16_UNORM") {
        return DXGI_FORMAT_R16_UNORM;
    }
    if (format == "DXGI_FORMAT_R16_UINT") {
        return DXGI_FORMAT_R16_UINT;
    }
    if (format == "DXGI_FORMAT_R16_SNORM") {
        return DXGI_FORMAT_R16_SNORM;
    }
    if (format == "DXGI_FORMAT_R16_SINT") {
        return DXGI_FORMAT_R16_SINT;
    }
    if (format == "DXGI_FORMAT_R8_TYPELESS") {
        return DXGI_FORMAT_R8_TYPELESS;
    }
    if (format == "DXGI_FORMAT_R8_UINT") {
        return DXGI_FORMAT_R8_UINT;
    }
    if (format == "DXGI_FORMAT_R8_SINT") {
        return DXGI_FORMAT_R8_SINT;
    }
    if (format == "DXGI_FORMAT_A8_UNORM") {
        return DXGI_FORMAT_A8_UNORM;
    }
    if (format == "DXGI_FORMAT_R1_UNORM") {
        return DXGI_FORMAT_R1_UNORM;
    }
    if (format == "DXGI_FORMAT_R9G9B9E5_SHAREDEXP") {
        return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    }
    if (format == "DXGI_FORMAT_R8G8_B8G8_UNORM") {
        return DXGI_FORMAT_R8G8_B8G8_UNORM;
    }
    if (format == "DXGI_FORMAT_G8R8_G8B8_UNORM") {
        return DXGI_FORMAT_G8R8_G8B8_UNORM;
    }
    if (format == "DXGI_FORMAT_BC1_TYPELESS") {
        return DXGI_FORMAT_BC1_TYPELESS;
    }
    if (format == "DXGI_FORMAT_BC1_UNORM") {
        return DXGI_FORMAT_BC1_UNORM;
    }
    if (format == "DXGI_FORMAT_BC1_UNORM_SRGB") {
        return DXGI_FORMAT_BC1_UNORM_SRGB;
    }
    if (format == "DXGI_FORMAT_BC2_TYPELESS") {
        return DXGI_FORMAT_BC2_TYPELESS;
    }
    if (format == "DXGI_FORMAT_BC2_UNORM") {
        return DXGI_FORMAT_BC2_UNORM;
    }
    if (format == "DXGI_FORMAT_BC2_UNORM_SRGB") {
        return DXGI_FORMAT_BC2_UNORM_SRGB;
    }
    if (format == "DXGI_FORMAT_BC3_TYPELESS") {
        return DXGI_FORMAT_BC3_TYPELESS;
    }
    if (format == "DXGI_FORMAT_BC3_UNORM") {
        return DXGI_FORMAT_BC3_UNORM;
    }
    if (format == "DXGI_FORMAT_BC3_UNORM_SRGB") {
        return DXGI_FORMAT_BC3_UNORM_SRGB;
    }
    if (format == "DXGI_FORMAT_BC4_TYPELESS") {
        return DXGI_FORMAT_BC4_TYPELESS;
    }
    if (format == "DXGI_FORMAT_BC4_UNORM") {
        return DXGI_FORMAT_BC4_UNORM;
    }
    if (format == "DXGI_FORMAT_BC4_SNORM") {
        return DXGI_FORMAT_BC4_SNORM;
    }
    if (format == "DXGI_FORMAT_BC5_TYPELESS") {
        return DXGI_FORMAT_BC5_TYPELESS;
    }
    if (format == "DXGI_FORMAT_BC5_UNORM") {
        return DXGI_FORMAT_BC5_UNORM;
    }
    if (format == "DXGI_FORMAT_BC5_SNORM") {
        return DXGI_FORMAT_BC5_SNORM;
    }
    if (format == "DXGI_FORMAT_B5G6R5_UNORM") {
        return DXGI_FORMAT_B5G6R5_UNORM;
    }
    if (format == "DXGI_FORMAT_B5G5R5A1_UNORM") {
        return DXGI_FORMAT_B5G5R5A1_UNORM;
    }
    if (format == "DXGI_FORMAT_B8G8R8A8_UNORM") {
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    }
    if (format == "DXGI_FORMAT_B8G8R8X8_UNORM") {
        return DXGI_FORMAT_B8G8R8X8_UNORM;
    }
    if (format == "DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM") {
        return DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM;
    }
    if (format == "DXGI_FORMAT_B8G8R8A8_TYPELESS") {
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    }
    if (format == "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB") {
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    }
    if (format == "DXGI_FORMAT_B8G8R8X8_TYPELESS") {
        return DXGI_FORMAT_B8G8R8X8_TYPELESS;
    }
    if (format == "DXGI_FORMAT_B8G8R8X8_UNORM_SRGB") {
        return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
    }
    if (format == "DXGI_FORMAT_BC6H_TYPELESS") {
        return DXGI_FORMAT_BC6H_TYPELESS;
    }
    if (format == "DXGI_FORMAT_BC6H_UF16") {
        return DXGI_FORMAT_BC6H_UF16;
    }
    if (format == "DXGI_FORMAT_BC6H_SF16") {
        return DXGI_FORMAT_BC6H_SF16;
    }
    if (format == "DXGI_FORMAT_BC7_TYPELESS") {
        return DXGI_FORMAT_BC7_TYPELESS;
    }
    if (format == "DXGI_FORMAT_BC7_UNORM") {
        return DXGI_FORMAT_BC7_UNORM;
    }
    if (format == "DXGI_FORMAT_BC7_UNORM_SRGB") {
        return DXGI_FORMAT_BC7_UNORM_SRGB;
    }
    if (format == "DXGI_FORMAT_AYUV") {
        return DXGI_FORMAT_AYUV;
    }
    if (format == "DXGI_FORMAT_Y410") {
        return DXGI_FORMAT_Y410;
    }
    if (format == "DXGI_FORMAT_Y416") {
        return DXGI_FORMAT_Y416;
    }
    if (format == "DXGI_FORMAT_NV12") {
        return DXGI_FORMAT_NV12;
    }
    if (format == "DXGI_FORMAT_P010") {
        return DXGI_FORMAT_P010;
    }
    if (format == "DXGI_FORMAT_P016") {
        return DXGI_FORMAT_P016;
    }
    if (format == "DXGI_FORMAT_420_OPAQUE") {
        return DXGI_FORMAT_420_OPAQUE;
    }
    if (format == "DXGI_FORMAT_YUY2") {
        return DXGI_FORMAT_YUY2;
    }
    if (format == "DXGI_FORMAT_Y210") {
        return DXGI_FORMAT_Y210;
    }
    if (format == "DXGI_FORMAT_Y216") {
        return DXGI_FORMAT_Y216;
    }
    if (format == "DXGI_FORMAT_NV11") {
        return DXGI_FORMAT_NV11;
    }
    if (format == "DXGI_FORMAT_AI44") {
        return DXGI_FORMAT_AI44;
    }
    if (format == "DXGI_FORMAT_IA44") {
        return DXGI_FORMAT_IA44;
    }
    if (format == "DXGI_FORMAT_P8") {
        return DXGI_FORMAT_P8;
    }
    if (format == "DXGI_FORMAT_A8P8") {
        return DXGI_FORMAT_A8P8;
    }
    if (format == "DXGI_FORMAT_B4G4R4A4_UNORM") {
        return DXGI_FORMAT_B4G4R4A4_UNORM;
    }
    if (format == "DXGI_FORMAT_P208") {
        return DXGI_FORMAT_P208;
    }
    if (format == "DXGI_FORMAT_V208") {
        return DXGI_FORMAT_V208;
    }
    if (format == "DXGI_FORMAT_V408") {
        return DXGI_FORMAT_V408;
    }
#endif

    throw std::runtime_error("Unsupported format " + format);
}

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

    auto dataPos = ddsFile.tellg();
    ddsFile.seekg(0, std::ios::end);
    auto size = ddsFile.tellg() - dataPos;
    ddsFile.close();
    if (size < 0) {
        throw std::runtime_error("Failed to get DDS file size " + ddsPath);
    }
    std::vector<uint8_t> ddsData(static_cast<size_t>(size));
    vk::Format format;
    loadDataFromDDS(ddsPath, ddsData, format);

    return {ddsHeader, ddsData};
}

void convertDDSToNpy(const std::string &input, const std::string &output, uint32_t elementSize) {
    DDSContent ddsContent = load(input);
    auto &info = ddsContent.first;
    auto &ddsData = ddsContent.second;
    auto imageElementSize = info.header.pitchOrLinearSize / info.header.width;

    if (elementSize > imageElementSize) {
        throw std::runtime_error("The image cannot be coverted to a NumPy file with bigger element size");
    }

    mlsdk::vgfutils::numpy::DataPtr data(reinterpret_cast<char *>(ddsData.data()),
                                         {1, info.header.height, info.header.width, imageElementSize / elementSize},
                                         mlsdk::vgfutils::numpy::DType('i', elementSize));
    mlsdk::vgfutils::numpy::write(output, data);
}

bool compareDDSHeader(const DDSHeaderInfo &input, const DDSHeaderInfo &output) {
    return input.header.width == output.header.width && input.header.height == output.header.height &&
           input.header.depth == output.header.depth && input.header10.dxgiFormat == output.header10.dxgiFormat;
}

bool isFloat16NaN(uint16_t val) {
    constexpr uint16_t FP16_EXPONENT = 0x7C00;
    return (val & FP16_EXPONENT) == FP16_EXPONENT;
}

bool compare(const std::string &input, const std::string &output, const std::string &elementDtype) {
    auto [inputHeader, inputData] = load(input);
    auto [outputHeader, outputData] = load(output);

    bool sameHeader = compareDDSHeader(inputHeader, outputHeader);

    bool sameData = false;
    if (elementDtype == "fp16") {
        auto *inputFp16 = reinterpret_cast<uint16_t *>(inputData.data());
        auto *outputFp16 = reinterpret_cast<uint16_t *>(outputData.data());

        auto size = inputData.size() / sizeof(uint16_t);
        sameData = true;
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
        sameData = inputData == outputData;
    }

    return sameHeader && sameData;
}

int main(int argc, const char **argv) {
    try {
        argparse::ArgumentParser parser(argv[0]);

        parser.add_argument("--action").choices("generate", "to_npy", "compare").help("Required action").required();

        parser.add_argument("--height").scan<'i', uint32_t>().help("Image height");
        parser.add_argument("--width").scan<'i', uint32_t>().help("Image width");
        parser.add_argument("--element-size").scan<'i', uint32_t>().help("Element size");
        parser.add_argument("--element-dtype").help("Element data type");
        parser.add_argument("--format").help("DXGI format");
        parser.add_argument("--header-only").default_value(false).implicit_value(true).help("Write DDS header only");
        parser.add_argument("--input").help("Path to the input file");
        parser.add_argument("--output").help("Path to the output file");

        parser.parse_args(argc, argv);
        std::string action = parser.get<std::string>("--action");

        if (action == "generate") {
            uint32_t height = parser.get<uint32_t>("--height");
            uint32_t width = parser.get<uint32_t>("--width");
            uint32_t elementSize = parser.get<uint32_t>("--element-size");
            std::string elementDtype = parser.get<std::string>("--element-dtype");
            std::string dxgiFormat = parser.get<std::string>("--format");
            std::string output = parser.get<std::string>("--output");
            bool headerOnly = parser.get<bool>("--header-only");

            generateDDSFile(height, width, elementDtype, elementSize, dxgiFormat, output, headerOnly);
        } else if (action == "to_npy") {
            std::string input = parser.get<std::string>("--input");
            std::string output = parser.get<std::string>("--output");
            uint32_t elementSize = parser.get<uint32_t>("--element-size");

            convertDDSToNpy(input, output, elementSize);
        } else if (action == "compare") {
            std::string input = parser.get<std::string>("--input");
            std::string output = parser.get<std::string>("--output");
            std::string elementDtype = parser.get<std::string>("--element-dtype");

            bool same = compare(input, output, elementDtype);
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
