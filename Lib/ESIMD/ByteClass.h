#pragma once

#include <cassert>
#include <concepts>
#include <cstdint>

// ESIMD: a set of byte values, for the byte-scanning primitives in ESIMD.h.
//
// A ByteClass holds up to `maxValues` individual bytes and up to `maxRanges`
// inclusive ranges, and `|` unions two of them. The builders read like the
// set they describe:
//
//     anyOf('"', '\\') | below(0x20)      a quote, a backslash or a control char
//     anyOf(' ', '\t', '\n', '\r')         JSON whitespace
//     inRange('0', '9') | anyOf('.', '-')  the bytes a decimal number is made of
//     above(0x7F)                          anything outside ASCII
//
// It is a plain value with no architecture conditionals: the kernels compile it
// into broadcast registers once per call, so building one is cheap and a
// `static constexpr` one costs nothing at all.
namespace esimd
{

struct ByteClass
{
    static constexpr int maxValues = 8;
    static constexpr int maxRanges = 2;

    // An inclusive [low, high] range. The default (low > high) is empty, which
    // the kernels rely on: the "not outside" form of the test matches nothing
    // without a branch.
    struct Range
    {
        std::uint8_t low = 1;
        std::uint8_t high = 0;
    };

    constexpr ByteClass() = default;

    // A single byte, so `findFirst(text, '"')` reads naturally.
    template <std::integral T>
    constexpr ByteClass(T byte)
    {
        values[0] = static_cast<std::uint8_t>(byte);
        valueCount = 1;
    }

    // The scalar definition of membership. The SIMD kernels mirror it lane for
    // lane, and the tests hold them to it.
    constexpr bool contains(std::uint8_t byte) const
    {
        for (auto k = 0; k < valueCount; ++k)
            if (values[k] == byte)
                return true;

        for (const auto& range: ranges)
            if (range.low <= byte && byte <= range.high)
                return true;

        return false;
    }

    std::uint8_t values[maxValues] {};
    int valueCount = 0;
    Range ranges[maxRanges] {};
};

// Any of the listed bytes.
template <std::integral... Bytes>
constexpr ByteClass anyOf(Bytes... bytes)
{
    static_assert(sizeof...(Bytes) <= ByteClass::maxValues,
                  "anyOf holds at most ByteClass::maxValues bytes");

    auto result = ByteClass {};
    ((result.values[result.valueCount++] = static_cast<std::uint8_t>(bytes)), ...);
    return result;
}

// Every byte from `low` to `high`, both inclusive.
constexpr ByteClass inRange(std::uint8_t low, std::uint8_t high)
{
    auto result = ByteClass {};
    result.ranges[0] = {low, high};
    return result;
}

// Every byte strictly below `limit` (so below(0x20) is the C0 control set).
constexpr ByteClass below(std::uint8_t limit)
{
    assert(limit > 0);
    return inRange(0, static_cast<std::uint8_t>(limit - 1));
}

// Every byte strictly above `limit` (so above(0x7F) is everything non-ASCII).
constexpr ByteClass above(std::uint8_t limit)
{
    assert(limit < 255);
    return inRange(static_cast<std::uint8_t>(limit + 1), 255);
}

// The union of two classes. The combined value and range counts must still fit
// the fixed capacity; that is asserted, not silently truncated.
constexpr ByteClass operator|(const ByteClass& a, const ByteClass& b)
{
    assert(a.valueCount + b.valueCount <= ByteClass::maxValues);

    auto result = a;

    for (auto k = 0; k < b.valueCount; ++k)
        result.values[result.valueCount++] = b.values[k];

    auto nextRange = 0;
    for (const auto& range: a.ranges)
        if (range.low <= range.high)
            result.ranges[nextRange++] = range;

    for (const auto& range: b.ranges)
    {
        if (range.low > range.high)
            continue;

        assert(nextRange < ByteClass::maxRanges);
        result.ranges[nextRange++] = range;
    }

    for (; nextRange < ByteClass::maxRanges; ++nextRange)
        result.ranges[nextRange] = {};

    return result;
}

} // namespace esimd
