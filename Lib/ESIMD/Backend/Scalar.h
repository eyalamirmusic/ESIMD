#pragma once

#include "../Common.h"

#include <bit>
#include <limits>

// The Scalar backend is the reference implementation of the primitive set:
// SIMD backends mirror it lane-for-lane, the swap kernel's width-remainder tail
// runs through it, and it is the oracle SIMD kernels are validated against.
namespace esimd::backend
{

struct Scalar
{
    // A natural-width vector of byte lanes (one lane here). The comparisons
    // yield 0xFF for true and 0x00 for false, the way the SIMD backends do.
    struct U8
    {
        U8 operator&(U8 o) const { return {static_cast<std::uint8_t>(v & o.v)}; }
        U8 operator|(U8 o) const { return {static_cast<std::uint8_t>(v | o.v)}; }
        U8 operator~() const { return {static_cast<std::uint8_t>(~v)}; }

        U8 equals(U8 o) const
        {
            return {static_cast<std::uint8_t>(v == o.v ? 0xFF : 0x00)};
        }

        // Unsigned this < o.
        U8 lessThan(U8 o) const
        {
            return {static_cast<std::uint8_t>(v < o.v ? 0xFF : 0x00)};
        }

        static U8 broadcast(std::uint8_t x) { return {x}; }
        static U8 load(const std::uint8_t* p) { return {*p}; }

        // One bit per lane, set where the lane is all-ones. The lanes must be
        // comparison results (all-ones or all-zeros).
        std::uint32_t bitmask() const { return v >> 7; }

        std::uint8_t v;
        static constexpr int lanes = 1;
    };

    // A natural-width vector of unsigned 32-bit lanes (one lane here).
    struct U32
    {
        U32 operator&(U32 o) const { return {v & o.v}; }
        U32 operator|(U32 o) const { return {v | o.v}; }

        template <int N>
        U32 shl() const
        {
            return {v << N};
        }

        template <int N>
        U32 shr() const
        {
            return {v >> N};
        }

        static U32 broadcast(std::uint32_t x) { return {x}; }

        static U32 load(const std::uint8_t* p)
        {
            std::uint32_t v;
            std::memcpy(&v, p, sizeof(v));
            return {v};
        }

        static void store(std::uint8_t* p, U32 a)
        {
            std::memcpy(p, &a.v, sizeof(a.v));
        }

        std::uint32_t v;
        static constexpr int lanes = 1;
    };

    // The four channels of one RGBA pixel held as floats.
    struct F4
    {
        F4 operator+(F4 o) const
        {
            return {{v[0] + o.v[0], v[1] + o.v[1], v[2] + o.v[2], v[3] + o.v[3]}};
        }

        F4 operator*(F4 o) const
        {
            return {{v[0] * o.v[0], v[1] * o.v[1], v[2] * o.v[2], v[3] * o.v[3]}};
        }

        static F4 broadcast(float x) { return {{x, x, x, x}}; }

        static F4 loadPixel(const std::uint8_t* p)
        {
            return {{static_cast<float>(p[0]),
                     static_cast<float>(p[1]),
                     static_cast<float>(p[2]),
                     static_cast<float>(p[3])}};
        }

        // Round each (non-negative) lane to a byte: lround + clamp to [0, 255].
        static void storePixel(std::uint8_t* out, F4 a)
        {
            for (int c = 0; c < 4; ++c)
            {
                const int r = static_cast<int>(std::lround(a.v[c]));
                out[c] = static_cast<std::uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
            }
        }

        float v[4];
    };

    // A natural-width vector of signed 32-bit lanes (one lane here): the
    // integer companion of F32, for exponent fields and comparison masks. The
    // arithmetic wraps the way the SIMD lanes do, and shr is arithmetic.
    struct I32
    {
        I32 operator+(I32 o) const { return {wrap(raw(v) + raw(o.v))}; }
        I32 operator-(I32 o) const { return {wrap(raw(v) - raw(o.v))}; }
        I32 operator&(I32 o) const { return {v & o.v}; }
        I32 operator|(I32 o) const { return {v | o.v}; }
        I32 operator^(I32 o) const { return {v ^ o.v}; }

        template <int N>
        I32 shl() const
        {
            return {wrap(raw(v) << N)};
        }

        template <int N>
        I32 shr() const
        {
            return {v >> N};
        }

        static I32 broadcast(std::int32_t x) { return {x}; }

        static std::uint32_t raw(std::int32_t x)
        {
            return static_cast<std::uint32_t>(x);
        }

        static std::int32_t wrap(std::uint32_t x)
        {
            return static_cast<std::int32_t>(x);
        }

        std::int32_t v;
        static constexpr int lanes = 1;
    };

    // A natural-width vector of float lanes (one lane here). Comparisons are
    // ordered (false when either side is NaN) and yield an I32 mask of
    // all-ones or all-zeros, the way the SIMD backends do.
    struct F32
    {
        F32 operator+(F32 o) const { return {v + o.v}; }
        F32 operator-(F32 o) const { return {v - o.v}; }
        F32 operator*(F32 o) const { return {v * o.v}; }
        F32 operator/(F32 o) const { return {v / o.v}; }

        I32 lessThan(F32 o) const { return mask(v < o.v); }
        I32 lessEqual(F32 o) const { return mask(v <= o.v); }
        I32 equals(F32 o) const { return mask(v == o.v); }
        I32 isNaN() const { return mask(v != v); }

        I32 bits() const { return {std::bit_cast<std::int32_t>(v)}; }

        // Round toward zero. Out of range or NaN gives INT32_MIN, as cvttps
        // does; the kernels only convert values already known to be in range.
        I32 truncate() const
        {
            if (!(std::fabs(v) < 2147483648.f))
                return {std::numeric_limits<std::int32_t>::min()};
            return {static_cast<std::int32_t>(v)};
        }

        static F32 fromBits(I32 a) { return {std::bit_cast<float>(a.v)}; }
        static F32 fromInt(I32 a) { return {static_cast<float>(a.v)}; }

        // Bitwise, like the SIMD blends: `mask` must be all-ones or all-zeros.
        static F32 select(I32 mask, F32 ifTrue, F32 ifFalse)
        {
            return fromBits((mask & ifTrue.bits())
                            | (I32 {~mask.v} & ifFalse.bits()));
        }

        static F32 broadcast(float x) { return {x}; }
        static F32 load(const float* p) { return {*p}; }
        static void store(float* p, F32 a) { *p = a.v; }

        static I32 mask(bool condition) { return {condition ? -1 : 0}; }

        float v;
        static constexpr int lanes = 1;
    };
};

} // namespace esimd::backend
