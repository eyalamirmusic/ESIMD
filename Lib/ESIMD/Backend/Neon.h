#pragma once

#include "../Common.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

// NEON backend (128-bit). NEON is mandatory on AArch64, so it is the
// unconditional baseline there and needs no runtime dispatch or compile flag.
namespace esimd::backend
{

struct Neon
{
    // Sixteen byte lanes (128-bit). Comparisons yield all-ones / all-zeros.
    struct U8
    {
        U8 operator&(U8 o) const { return {vandq_u8(v, o.v)}; }
        U8 operator|(U8 o) const { return {vorrq_u8(v, o.v)}; }
        U8 operator~() const { return {vmvnq_u8(v)}; }

        U8 equals(U8 o) const { return {vceqq_u8(v, o.v)}; }

        // Unsigned this < o.
        U8 lessThan(U8 o) const { return {vcltq_u8(v, o.v)}; }

        static U8 broadcast(std::uint8_t x) { return {vdupq_n_u8(x)}; }
        static U8 load(const std::uint8_t* p) { return {vld1q_u8(p)}; }

        // One bit per lane, set where the lane is all-ones. The lanes must be
        // comparison results (all-ones or all-zeros).
        //
        // NEON has no movemask. Each lane is masked down to the bit weight of
        // its position within its 8-lane half, and a horizontal add of each
        // half packs those weights into one byte: the weights are disjoint, so
        // the add is an OR.
        std::uint32_t bitmask() const
        {
            static constexpr std::uint8_t weights[16] = {
                1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

            const auto weighted = vandq_u8(v, vld1q_u8(weights));
            const auto low = vaddv_u8(vget_low_u8(weighted));
            const auto high = vaddv_u8(vget_high_u8(weighted));
            return static_cast<std::uint32_t>(low)
                   | (static_cast<std::uint32_t>(high) << 8);
        }

        uint8x16_t v;
        static constexpr int lanes = 16;
    };

    // Four unsigned 32-bit lanes (128-bit).
    struct U32
    {
        U32 operator&(U32 o) const { return {vandq_u32(v, o.v)}; }
        U32 operator|(U32 o) const { return {vorrq_u32(v, o.v)}; }

        template <int N>
        U32 shl() const
        {
            return {vshlq_n_u32(v, N)};
        }

        template <int N>
        U32 shr() const
        {
            return {vshrq_n_u32(v, N)};
        }

        static U32 broadcast(std::uint32_t x) { return {vdupq_n_u32(x)}; }

        static U32 load(const std::uint8_t* p)
        {
            return {vld1q_u32(reinterpret_cast<const std::uint32_t*>(p))};
        }

        static void store(std::uint8_t* p, U32 a)
        {
            vst1q_u32(reinterpret_cast<std::uint32_t*>(p), a.v);
        }

        uint32x4_t v;
        static constexpr int lanes = 4;
    };

    // The four channels of one RGBA pixel held as floats (128-bit).
    struct F4
    {
        F4 operator+(F4 o) const { return {vaddq_f32(v, o.v)}; }
        F4 operator*(F4 o) const { return {vmulq_f32(v, o.v)}; }

        static F4 broadcast(float x) { return {vdupq_n_f32(x)}; }

        // Load four packed bytes (one RGBA pixel) and widen to four floats.
        static F4 loadPixel(const std::uint8_t* p)
        {
            std::uint32_t packed;
            std::memcpy(&packed, p, 4);
            const auto bytes = vreinterpret_u8_u32(vdup_n_u32(packed));
            const auto words = vget_low_u16(vmovl_u8(bytes));
            return {vcvtq_f32_u32(vmovl_u16(words))};
        }

        // Round each (non-negative) lane with floor(v + 0.5) -- equal to lround
        // here -- saturate to [0, 255], and store the four bytes.
        static void storePixel(std::uint8_t* out, F4 a)
        {
            const auto rounded = vcvtq_s32_f32(vaddq_f32(a.v, vdupq_n_f32(0.5f)));
            const auto words = vqmovun_s32(rounded);
            const auto bytes = vqmovn_u16(vcombine_u16(words, words));
            const auto packed = vget_lane_u32(vreinterpret_u32_u8(bytes), 0);
            std::memcpy(out, &packed, 4);
        }

        float32x4_t v;
    };

    // Four signed 32-bit lanes (128-bit), the integer companion of F32.
    struct I32
    {
        I32 operator+(I32 o) const { return {vaddq_s32(v, o.v)}; }
        I32 operator-(I32 o) const { return {vsubq_s32(v, o.v)}; }
        I32 operator&(I32 o) const { return {vandq_s32(v, o.v)}; }
        I32 operator|(I32 o) const { return {vorrq_s32(v, o.v)}; }
        I32 operator^(I32 o) const { return {veorq_s32(v, o.v)}; }

        template <int N>
        I32 shl() const
        {
            return {vshlq_n_s32(v, N)};
        }

        template <int N>
        I32 shr() const
        {
            return {vshrq_n_s32(v, N)};
        }

        static I32 broadcast(std::int32_t x) { return {vdupq_n_s32(x)}; }

        int32x4_t v;
        static constexpr int lanes = 4;
    };

    // Four float lanes (128-bit). Comparisons are ordered and yield I32 masks.
    // truncate saturates where cvttps gives INT32_MIN, which is why the
    // kernels only convert values already known to be in range.
    struct F32
    {
        F32 operator+(F32 o) const { return {vaddq_f32(v, o.v)}; }
        F32 operator-(F32 o) const { return {vsubq_f32(v, o.v)}; }
        F32 operator*(F32 o) const { return {vmulq_f32(v, o.v)}; }
        F32 operator/(F32 o) const { return {vdivq_f32(v, o.v)}; }

        I32 lessThan(F32 o) const { return mask(vcltq_f32(v, o.v)); }
        I32 lessEqual(F32 o) const { return mask(vcleq_f32(v, o.v)); }
        I32 equals(F32 o) const { return mask(vceqq_f32(v, o.v)); }
        I32 isNaN() const { return mask(vmvnq_u32(vceqq_f32(v, v))); }

        I32 bits() const { return {vreinterpretq_s32_f32(v)}; }
        I32 truncate() const { return {vcvtq_s32_f32(v)}; }

        static F32 fromBits(I32 a) { return {vreinterpretq_f32_s32(a.v)}; }
        static F32 fromInt(I32 a) { return {vcvtq_f32_s32(a.v)}; }

        static F32 select(I32 mask, F32 ifTrue, F32 ifFalse)
        {
            return {vbslq_f32(vreinterpretq_u32_s32(mask.v), ifTrue.v, ifFalse.v)};
        }

        static F32 broadcast(float x) { return {vdupq_n_f32(x)}; }
        static F32 load(const float* p) { return {vld1q_f32(p)}; }
        static void store(float* p, F32 a) { vst1q_f32(p, a.v); }

        static I32 mask(uint32x4_t m) { return {vreinterpretq_s32_u32(m)}; }

        float32x4_t v;
        static constexpr int lanes = 4;
    };
};

} // namespace esimd::backend

#endif
