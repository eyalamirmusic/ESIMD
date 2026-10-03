#pragma once

#include "../Common.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

// AVX2 backend (256-bit). Only ever compiled inside Tu/Avx2.cpp, which carries
// the -mavx2/-mfma (or /arch:AVX2) flag and is reached only through the runtime
// dispatcher when the CPU supports AVX2 + FMA.
namespace esimd::backend
{

struct Avx2
{
    // Thirty-two byte lanes (256-bit). Comparisons yield all-ones / all-zeros.
    struct U8
    {
        U8 operator&(U8 o) const { return {_mm256_and_si256(v, o.v)}; }
        U8 operator|(U8 o) const { return {_mm256_or_si256(v, o.v)}; }

        U8 operator~() const { return {_mm256_xor_si256(v, _mm256_set1_epi32(-1))}; }

        U8 equals(U8 o) const { return {_mm256_cmpeq_epi8(v, o.v)}; }

        // Unsigned this < o, via the same sign-bit flip as SSE2 (AVX2 has only
        // the signed greater-than, so the operands swap).
        U8 lessThan(U8 o) const
        {
            const auto flip = _mm256_set1_epi8(static_cast<char>(0x80));
            return {_mm256_cmpgt_epi8(_mm256_xor_si256(o.v, flip),
                                      _mm256_xor_si256(v, flip))};
        }

        static U8 broadcast(std::uint8_t x)
        {
            return {_mm256_set1_epi8(static_cast<char>(x))};
        }

        static U8 load(const std::uint8_t* p)
        {
            return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))};
        }

        // One bit per lane, set where the lane is all-ones. The lanes must be
        // comparison results (all-ones or all-zeros).
        std::uint32_t bitmask() const
        {
            return static_cast<std::uint32_t>(_mm256_movemask_epi8(v));
        }

        __m256i v;
        static constexpr int lanes = 32;
    };

    // Eight unsigned 32-bit lanes (256-bit).
    struct U32
    {
        U32 operator&(U32 o) const { return {_mm256_and_si256(v, o.v)}; }
        U32 operator|(U32 o) const { return {_mm256_or_si256(v, o.v)}; }

        template <int N>
        U32 shl() const
        {
            return {_mm256_slli_epi32(v, N)};
        }

        template <int N>
        U32 shr() const
        {
            return {_mm256_srli_epi32(v, N)};
        }

        static U32 broadcast(std::uint32_t x)
        {
            return {_mm256_set1_epi32(static_cast<int>(x))};
        }

        static U32 load(const std::uint8_t* p)
        {
            return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))};
        }

        static void store(std::uint8_t* p, U32 a)
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a.v);
        }

        __m256i v;
        static constexpr int lanes = 8;
    };
};

} // namespace esimd::backend

#endif
