/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdint>
#include <string>
#include <vector>

namespace mlsdk::scenariorunner {
/// \brief GLSL Compiler Singleton
class GlslCompiler {
  public:
    GlslCompiler(const GlslCompiler &) = delete;
    GlslCompiler &operator=(const GlslCompiler &) = delete;

    /// \brief Compiler instance accessor
    ///
    /// \return The static instance of the GlslCompiler
    static GlslCompiler &get();

    /// \brief Compile a GLSL source given in textual form
    ///
    /// \param source Source GLSL program to compile
    /// \param preprocessorOptions Pre-processor options for the GLSL program to compile
    /// \param shaderDirs List of directories where to find shader headers to include
    ///
    /// \return A pair containing the compilation log and the SPIR-V code if
    /// successful else empty
    std::pair<std::string, std::vector<uint32_t>> compile(const std::string &source,
                                                          const std::string &preprocessorOptions = "",
                                                          const std::vector<std::string> &shaderDirs = {});

    /// \brief Save a SPIR-V module to file
    ///
    /// \param mod SPIR-V module to save
    /// \param fname Name of the file to save the module to
    /// \return True if successful, false otherwise
    bool save(const std::vector<uint32_t> &mod, const std::string &fname);

    /// \brief Load a GLSL module from file
    ///
    /// \param fname Name of the file to load the module from
    /// \param glsl The output string containing the file content
    /// \return True if successful, false otherwise
    bool load(const std::string &fname, std::string &glsl);

  private:
    /// \brief Default constructor
    GlslCompiler();
    /// \brief Destructor
    ~GlslCompiler();
};
} // namespace mlsdk::scenariorunner
