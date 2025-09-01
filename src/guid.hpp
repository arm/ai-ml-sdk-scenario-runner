/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <functional>
#include <string>

namespace mlsdk::scenariorunner {

class Guid {
  private:
    using HashType = std::size_t;
    static constexpr HashType invalidValue() { return HashType(-1); }

    HashType _hash{invalidValue()};

  public:
    Guid() = default;

    Guid(const std::string &s) : _hash(std::hash<std::string>{}(s)) {} // cppcheck-suppress noExplicitConstructor

    // Move/Copy constructor
    Guid(Guid &&other) = default;
    Guid(const Guid &other) = default;

    // Move/Copy Assignment operators
    Guid &operator=(Guid &&other) = default;
    Guid &operator=(const Guid &other) = default;
    Guid &operator=(const std::string &s) {
        _hash = std::hash<std::string>{}(s);
        return *this;
    }

    // Comparison
    bool operator==(const Guid &other) const { return _hash == other._hash; }
    bool operator!=(const Guid &other) const { return _hash != other._hash; }
    bool operator<(const Guid &other) const { return _hash < other._hash; }

    bool isValid() const { return _hash != invalidValue(); }

    friend struct std::hash<Guid>;
};

} // namespace mlsdk::scenariorunner

/// Skip Doxygen® for this to fix warning
/// \cond
template <> struct std::hash<mlsdk::scenariorunner::Guid> {
    std::size_t operator()(const mlsdk::scenariorunner::Guid &guid) const noexcept { return guid._hash; }
};
/// \endcond
