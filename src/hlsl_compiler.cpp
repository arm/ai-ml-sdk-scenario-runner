/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hlsl_compiler.hpp"

#include <cstring>
#include <fstream>

#if defined(_WIN32)
#    include <atlbase.h>
#    include <windows.h>
#endif

#include <dxc/dxcapi.h>

namespace mlsdk::scenariorunner {
namespace {
std::wstring stringToWstring(const std::string &inputString) {
// DXC api requires some args to be in std::wstring rather than std::string, this function converts
// std::string to std::wstring across all platforms
#if defined(_WIN32)
    // Windows: UTF-8 -> UTF-16
    int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, inputString.data(),
                                  static_cast<int>(inputString.size()), nullptr, 0);
    if (len < 0)
        throw std::runtime_error("Invalid UTF-8");

    std::wstring wide(len, 0);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, inputString.data(), static_cast<int>(inputString.size()),
                        wide.data(), len);
    return wide;

#else
    // POSIX: wchar_t is UTF-32
    std::wstring wide;
    wide.reserve(inputString.size());

    for (size_t i = 0; i < inputString.size();) {
        int cp = 0;
        unsigned char c = static_cast<unsigned char>(inputString[i]);

        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if ((c >> 5) == 0x6) {
            cp = ((c & 0x1F) << 6) | (inputString[i + 1] & 0x3F);
            i += 2;
        } else if ((c >> 4) == 0xE) {
            cp = ((c & 0x0F) << 12) | ((inputString[i + 1] & 0x3F) << 6) | (inputString[i + 2] & 0x3F);
            i += 3;
        } else if ((c >> 3) == 0x1E) {
            cp = ((c & 0x07) << 18) | ((inputString[i + 1] & 0x3F) << 12) | ((inputString[i + 2] & 0x3F) << 6) |
                 (inputString[i + 3] & 0x3F);
            i += 4;
        } else {
            throw std::runtime_error("Invalid UTF-8");
        }

        wide.push_back(static_cast<wchar_t>(cp));
    }

    return wide;
#endif
}

std::vector<DxcDefine> parsePreprocessorOptions(const std::string &options, std::vector<std::wstring> &storage) {
    std::vector<DxcDefine> defineStrings;
    const std::string delimiter = " ";
    size_t start = 0;
    while (start < options.size()) {
        size_t end = options.find(delimiter, start);
        if (end == std::string::npos) {
            end = options.size();
        }
        std::string op = options.substr(start + 2, end - (start + 2)); // +2 skips the -D
        // split the name and the value and create a DxcDefine object
        const size_t equal = op.find_first_of("=");
        DxcDefine defineStruct{};
        // name and value have to also be stored in a separate vector to avoid dangling pointers
        if (equal == std::string::npos) {
            storage.emplace_back(stringToWstring(op));
            const std::wstring &name = storage.back();
            defineStruct.Name = name.c_str();
            defineStruct.Value = nullptr;
        } else {
            const size_t name_idx = storage.size();
            storage.emplace_back(stringToWstring(op.substr(0, equal)));
            const size_t value_idx = storage.size();
            storage.emplace_back(stringToWstring(op.substr(equal + 1, op.size())));
            defineStruct.Name = storage[name_idx].c_str();
            defineStruct.Value = storage[value_idx].c_str();
        }
        defineStrings.push_back(defineStruct);
        start = end + 1;
    }
    return defineStrings;
}

} // namespace

HlslCompiler &HlslCompiler::get() {
    static HlslCompiler hlslCompiler;
    return hlslCompiler;
}

std::pair<std::string, std::vector<uint32_t>> HlslCompiler::compile(const std::string &source, const std::string &entry,
                                                                    const std::string &debugName,
                                                                    const std::string &preprocessorOptions,
                                                                    const std::vector<std::string> &shaderDirs) {
    // Create DXC instances for the IDxcUtils and IDxcCompiler interfaces
    CComPtr<IDxcUtils> utils;
    CComPtr<IDxcCompiler3> compiler;
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

    DxcBuffer buffer;
    buffer.Ptr = source.data();
    buffer.Size = source.size();
    buffer.Encoding = DXC_CP_UTF8;

    // Default include handler; shaderDirs are appended to extraArgs below
    CComPtr<IDxcIncludeHandler> includeHandler;
    utils->CreateDefaultIncludeHandler(&includeHandler);

    // IDxcCompiler3::Compile function requires string args to be std::wstring rather than std::string
    std::wstring name = stringToWstring(debugName);
    std::wstring entryPoint = stringToWstring(entry);

    std::vector<const wchar_t *> extraArgs;
    std::vector<std::wstring> intermediateArgs;
    extraArgs.emplace_back(L"-spirv");
    extraArgs.emplace_back(L"-P");                  // enable preprocessing
    extraArgs.emplace_back(L"-enable-16bit-types"); // enable int16_t
    for (const auto &dir : shaderDirs) {
        intermediateArgs.emplace_back(stringToWstring(dir)); // necessary to avoid dangling pointers
        extraArgs.emplace_back(intermediateArgs.back().c_str());
    }

    std::vector<std::wstring> defineIntermediate;
    std::vector<DxcDefine> defineStrings = parsePreprocessorOptions(preprocessorOptions, defineIntermediate);

    CComPtr<IDxcCompilerArgs> args;
    utils->BuildArguments(name.c_str(), entryPoint.c_str(),
                          L"cs_6_2", // shader type and version
                          extraArgs.data(), static_cast<uint32_t>(extraArgs.size()), defineStrings.data(),
                          static_cast<uint32_t>(defineStrings.size()), &args);

    // Preprocess
    CComPtr<IDxcResult> result;
    compiler->Compile(&buffer, args->GetArguments(), args->GetCount(), includeHandler, IID_PPV_ARGS(&result));

    std::string log;
    CComPtr<IDxcBlobUtf8> blob;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&blob), nullptr);
    if ((blob) && (blob->GetStringLength() > 0)) {
        log.append(blob->GetStringPointer(), blob->GetStringLength());

        if (result->HasOutput(DXC_OUT_PDB)) {
            CComPtr<IDxcBlobUtf8> debugInfo;
            result->GetOutput(DXC_OUT_PDB, IID_PPV_ARGS(&debugInfo), nullptr);
            log.append(debugInfo->GetStringPointer(), debugInfo->GetStringLength());
        }
        return {log, {}};
    }

    // Compile
    std::vector<const wchar_t *> compileArgs;
    compileArgs.emplace_back(L"-spirv");
    compileArgs.emplace_back(L"-enable-16bit-types");
    for (const auto &dir : intermediateArgs) {
        compileArgs.emplace_back(dir.c_str());
    }

    CComPtr<IDxcCompilerArgs> args1;
    utils->BuildArguments(name.c_str(), entryPoint.c_str(), L"cs_6_2", compileArgs.data(),
                          static_cast<uint32_t>(compileArgs.size()), defineStrings.data(),
                          static_cast<uint32_t>(defineStrings.size()), &args1);

    CComPtr<IDxcResult> compileResult;
    compiler->Compile(&buffer, args1->GetArguments(), args1->GetCount(), includeHandler, IID_PPV_ARGS(&compileResult));
    CComPtr<IDxcBlobUtf8> compileBlob;
    compileResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&compileBlob), nullptr);
    if (compileBlob && compileBlob->GetStringLength() > 0) {
        log.append(compileBlob->GetStringPointer(), compileBlob->GetStringLength());
        if (compileResult->HasOutput(DXC_OUT_PDB)) {
            CComPtr<IDxcBlobUtf8> debugInfo;
            compileResult->GetOutput(DXC_OUT_PDB, IID_PPV_ARGS(&debugInfo), nullptr);
            log.append(debugInfo->GetStringPointer(), debugInfo->GetStringLength());
        }
        return {log, {}};
    }

    // Get SPIR-V blob
    CComPtr<IDxcBlob> obj;
    compileResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&obj), nullptr);
    const void *data = obj->GetBufferPointer();
    size_t size = obj->GetBufferSize();
    if (size % sizeof(uint32_t) != 0) {
        return {"object blob size is not 4-byte aligned", {}};
    }
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    std::memcpy(spirv.data(), data, size);
    return {log, spirv};
}

bool HlslCompiler::load(const std::string &fname, std::string &hlsl) {
    std::ifstream ifile{fname, std::ios::in | std::ios::binary};
    if (!ifile.is_open()) {
        return false;
    }

    hlsl = std::string((std::istreambuf_iterator<char>(ifile)), (std::istreambuf_iterator<char>()));
    if (ifile.bad()) {
        return false;
    }

    return true;
}

bool HlslCompiler::save(const std::vector<uint32_t> &mod, const std::string &fname) {
    std::ofstream ofile{fname, std::ios::out | std::ios::binary};
    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(reinterpret_cast<const char *>(mod.data()), std::streamsize(mod.size() * sizeof(uint32_t)));
    return ofile.good();
}
} // namespace mlsdk::scenariorunner
