/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdint>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {
/// \brief HLSL Compiler Singleton
class HlslCompiler {
  public:
    HlslCompiler(const HlslCompiler &) = delete;
    HlslCompiler &operator=(const HlslCompiler &) = delete;

    /// \brief Compiler instance accessor
    ///
    /// \return The static instance of the HlslCompiler
    static HlslCompiler &get();

    /// \brief Compile a HLSL source given in textual form
    ///
    /// \param source Source HLSL program to compile
    /// \param entry Entry point for the shader
    /// \param debugName Name of the HLSL program for debug purposes
    /// \param preprocessorOptions Pre-processor options for the HLSL program to compile
    /// \param shaderDirs List of directories where to find shader headers to include
    ///
    /// \return A pair containing the compilation log and the SPIR-V code if
    /// successful else empty
    std::pair<std::string, std::vector<uint32_t>> compile(const std::string &source, const std::string &entry,
                                                          const std::string &debugName,
                                                          const std::string &preprocessorOptions = "",
                                                          const std::vector<std::string> &shaderDirs = {});

    /// \brief Save a SPIR-V module to file
    ///
    /// \param mod SPIR-V module to save
    /// \param fname Name of the file to save the module to
    /// \return True if successful, false otherwise
    bool save(const std::vector<uint32_t> &mod, const std::string &fname);

    /// \brief Load a HLSL module from file
    ///
    /// \param fname Name of the file to load the module from
    /// \param hlsl The output string containing the file content
    /// \return True if successful, false otherwise
    bool load(const std::string &fname, std::string &hlsl);

  private:
    /// \brief Default constructor
    HlslCompiler() = default;
};
} // namespace mlsdk::scenariorunner
