#pragma once

#include "../Backend/Scalar.h"
#include "../ByteClass.h"

#include <bit>

namespace esimd::kernels
{

// Everything here has internal linkage on purpose. These templates are
// instantiated once per ISA translation unit, and Tu/Avx2.cpp is compiled with
// -mavx2: if its instantiation of, say, the Scalar tail shared a (COMDAT)
// symbol with the one in Tu/Scalar.cpp, the linker would keep one copy at
// random, and the AVX2-compiled copy could end up behind the scalar entry
// point on a CPU that cannot run it. An anonymous namespace gives every
// translation unit its own private instantiations, so no such merge can happen.
namespace
{

// A ByteClass compiled for backend B: every value and range bound broadcast
// into a register once, so the per-vector test below is pure compares and ORs.
template <class B>
struct ByteClassMatcher
{
    using V = typename B::U8;

    explicit ByteClassMatcher(const ByteClass& cls)
        : valueCount(cls.valueCount)
    {
        for (auto k = 0; k < valueCount; ++k)
            values[k] = V::broadcast(cls.values[k]);

        for (auto r = 0; r < ByteClass::maxRanges; ++r)
        {
            lows[r] = V::broadcast(cls.ranges[r].low);
            highs[r] = V::broadcast(cls.ranges[r].high);
        }
    }

    // All-ones in every lane whose byte is in the class.
    V matches(V bytes) const
    {
        // A range is tested as "not (below low or above high)". An empty range
        // has low > high, so every byte is outside it and the test is all-zeros
        // with no branch on whether the range is in use.
        auto hits = ~(bytes.lessThan(lows[0]) | highs[0].lessThan(bytes));

        for (auto r = 1; r < ByteClass::maxRanges; ++r)
            hits = hits | ~(bytes.lessThan(lows[r]) | highs[r].lessThan(bytes));

        for (auto k = 0; k < valueCount; ++k)
            hits = hits | bytes.equals(values[k]);

        return hits;
    }

    V values[ByteClass::maxValues] {};
    V lows[ByteClass::maxRanges] {};
    V highs[ByteClass::maxRanges] {};
    int valueCount;
};

// Index of the first byte in [data, data + count) that is in `cls` (or, with
// Negate, the first that is not), or `count` when there is none. The main loop
// takes B::U8::lanes bytes per step; the remainder goes through the Scalar
// backend, which is also the whole algorithm when B is Scalar.
template <class B, bool Negate>
int findFirstImpl(const std::uint8_t* data, int count, const ByteClass& cls)
{
    using V = typename B::U8;
    const auto matcher = ByteClassMatcher<B>(cls);

    auto i = 0;
    for (; i + V::lanes <= count; i += V::lanes)
    {
        auto hits = matcher.matches(V::load(data + i));

        if constexpr (Negate)
            hits = ~hits;

        if (const auto mask = hits.bitmask(); mask != 0)
            return i + std::countr_zero(mask);
    }

    if constexpr (V::lanes > 1)
        if (i < count)
            return i
                   + findFirstImpl<backend::Scalar, Negate>(
                       data + i, count - i, cls);

    return count;
}

// How many bytes in [data, data + count) are in `cls`.
template <class B>
int countOfImpl(const std::uint8_t* data, int count, const ByteClass& cls)
{
    using V = typename B::U8;
    const auto matcher = ByteClassMatcher<B>(cls);

    auto total = 0;
    auto i = 0;
    for (; i + V::lanes <= count; i += V::lanes)
        total += std::popcount(matcher.matches(V::load(data + i)).bitmask());

    if constexpr (V::lanes > 1)
        if (i < count)
            total += countOfImpl<backend::Scalar>(data + i, count - i, cls);

    return total;
}

} // namespace
} // namespace esimd::kernels
