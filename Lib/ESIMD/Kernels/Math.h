#pragma once

#include "../Backend/Scalar.h"

#include <bit>
#include <initializer_list>
#include <limits>

// Elementwise exp, exp2, log, log2, sin, cos and tanh, each written once over a
// backend B. They are built only from lane operations that are exact IEEE
// single precision and identical on every ISA -- add, subtract, multiply,
// divide, ordered compares, bitwise selects, bit casts, integer adds and
// shifts. No FMA, no reciprocal or rsqrt estimate, no min/max (SSE and NEON
// disagree on NaN), and float-to-int conversion only of values already known
// to be integral and in range. Range reduction and polynomial order are fixed
// here, so every backend computes the same bits, and the Scalar backend
// finishes the remainder of every array.
//
// Every NaN a kernel returns is either the input passed through by a select or
// a constant: a NaN produced by arithmetic has a different sign bit on x86
// than on ARM.
namespace esimd::kernels
{

// Internal linkage on purpose; see Kernels/FindByte.h.
namespace
{

constexpr auto infinity = std::numeric_limits<float>::infinity();
constexpr auto quietNaN = std::bit_cast<float>(0x7FC00000u);
constexpr auto smallestNormal = std::numeric_limits<float>::min();

// exp(x) is finite up to the float below ln(FLT_MAX) and overflows from the
// next one. Below expLowest the result is far under FLT_MIN and flushed.
constexpr auto expHighest = 0x1.62e42ep+6f;
constexpr auto expLowest = -88.f;
constexpr auto exp2Highest = 0x1.fffffep+6f;
constexpr auto exp2Lowest = -127.f;

// ln 2 split so that n * ln2Hi is exact for every |n| <= 128.
constexpr auto log2OfE = 1.44269504088896341f;
constexpr auto ln2Hi = 0.693359375f;
constexpr auto ln2Lo = -2.12194440e-4f;

constexpr auto sqrtHalf = 0.707106781186547524f;
constexpr auto log2OfEMinusOne = 0.44269504088896340736f;

// pi/2 as five consecutive 9-bit windows of its binary expansion and a
// full-precision tail (residual ~1e-22). No chunk has more than 8 significant
// bits, so every j * chunk is exact for |j| < 2^16; that is what keeps sin and
// cos within 1.3 ULP up to |x| = 2^16, even right next to a multiple of pi/2.
constexpr auto twoOverPi = 0.636619772367581343f;
constexpr auto halfPi1 = 0x1.92p+0f;
constexpr auto halfPi2 = 0x1.f8p-12f;
constexpr auto halfPi3 = 0x1.aap-19f;
constexpr auto halfPi4 = 0x1.1p-30f;
constexpr auto halfPi5 = 0x1.68p-39f;
constexpr auto halfPi6 = 0x1.84698ap-48f;

// Past 2^22 the quadrant no longer fits the rounding trick below; such
// arguments are not reduced at all (sin gives 0 and cos gives 1).
constexpr auto sinCosLargest = 0x1p22f;

// Above this, tanh is 1 to float precision; clamping there also keeps the
// exponential far from overflow.
constexpr auto tanhSaturation = 10.f;
constexpr auto tanhExpThreshold = 0.625f;

template <class B>
struct Lanes
{
    using F = typename B::F32;
    using I = typename B::I32;

    static F splat(float x) { return F::broadcast(x); }
    static I splatInt(std::int32_t x) { return I::broadcast(x); }

    // c[0] * x^n + ... + c[n], evaluated by Horner's rule in exactly this
    // order, one rounding per multiply and per add.
    static F horner(F x, std::initializer_list<float> coefficients)
    {
        auto it = coefficients.begin();
        auto sum = splat(*it);
        for (++it; it != coefficients.end(); ++it)
            sum = sum * x + splat(*it);
        return sum;
    }

    // Round to the nearest integer, ties to even, for |v| < 2^22: adding and
    // subtracting 1.5 * 2^23 pushes the fraction bits out. Two adds on every
    // ISA, independent of the rounding instructions each one has or lacks.
    static F roundToNearest(F v)
    {
        const auto shifter = splat(0x1.8p+23f);
        return (v + shifter) - shifter;
    }

    // The clamp of x to [low, high] by compare and select. A NaN lane lands
    // on `low`, the same on every backend.
    static F clamp(F x, float low, float high)
    {
        const auto lowest = splat(low);
        const auto highest = splat(high);
        const auto aboveLow = F::select(lowest.lessThan(x), x, lowest);
        return F::select(aboveLow.lessThan(highest), aboveLow, highest);
    }

    static I signOf(F x)
    {
        return x.bits() & splatInt(std::numeric_limits<std::int32_t>::min());
    }

    static F abs(F x) { return F::fromBits(x.bits() & splatInt(0x7FFFFFFF)); }

    static F withSign(F magnitude, I sign)
    {
        return F::fromBits(magnitude.bits() | sign);
    }

    static F flipSign(F x, I sign) { return F::fromBits(x.bits() ^ sign); }

    // All-ones where bit `Bit` of n is set.
    template <int Bit>
    static I maskOfBit(I n)
    {
        return n.template shl<31 - Bit>().template shr<31>();
    }

    // 2^n for n in [-126, 127], written straight into the exponent field.
    static F powerOfTwo(I n)
    {
        return F::fromBits((n + splatInt(127)).template shl<23>());
    }

    // v * 2^n for n in [-127, 128]. Two normal factors, so nothing rounds
    // before the final product.
    static F scaleByPowerOfTwo(F v, I n)
    {
        const auto half = n.template shr<1>();
        return (v * powerOfTwo(half)) * powerOfTwo(n - half);
    }

    // Results below FLT_MIN become +0, decided on the value itself, so the
    // answer is the same whether or not the CPU flushes subnormals.
    static F flushSubnormal(F y)
    {
        return F::select(y.lessThan(splat(smallestNormal)), splat(0.f), y);
    }

    // e^x for x in [expLowest, expHighest]: x = n ln2 + r with |r| <= ln2/2,
    // e^r from a degree-7 polynomial, then scaled by 2^n.
    static F expOfClamped(F x)
    {
        const auto n = roundToNearest(x * splat(log2OfE));
        const auto r = (x - n * splat(ln2Hi)) - n * splat(ln2Lo);
        const auto p = horner(r,
                              {1.9875691500e-4f,
                               1.3981999507e-3f,
                               8.3334519073e-3f,
                               4.1665795894e-2f,
                               1.6666665459e-1f,
                               5.0000001201e-1f});
        const auto expR = (p * (r * r) + r) + splat(1.f);
        return scaleByPowerOfTwo(expR, n.truncate());
    }

    // x = 2^exponent * (1 + mantissaMinusOne), with the mantissa in
    // [sqrt(1/2), sqrt(2)) so that log of it stays small near x = 1.
    // Subnormal x is scaled up by 2^23 first. Meaningless for x <= 0, inf and
    // NaN, which the caller patches.
    struct Decomposed
    {
        F mantissaMinusOne;
        F exponent;
    };

    static Decomposed decompose(F x)
    {
        const auto subnormal = x.lessThan(splat(smallestNormal));
        const auto bits = F::select(subnormal, x * splat(0x1p+23f), x).bits();

        const auto biased = bits.template shr<23>() & splatInt(0xFF);
        auto exponent = biased - splatInt(126) + (subnormal & splatInt(-23));
        auto mantissa =
            F::fromBits((bits & splatInt(0x007FFFFF)) | splatInt(0x3F000000));

        const auto belowSqrtHalf = mantissa.lessThan(splat(sqrtHalf));
        exponent = exponent + belowSqrtHalf;
        mantissa = F::select(belowSqrtHalf, mantissa + mantissa, mantissa);

        return {mantissa - splat(1.f), F::fromInt(exponent)};
    }

    // The log(1 + m) polynomial both logarithms share, without its m^3 factor.
    static F logPolynomial(F m)
    {
        return horner(m,
                      {7.0376836292e-2f,
                       -1.1514610310e-1f,
                       1.1676998740e-1f,
                       -1.2420140846e-1f,
                       1.4249322787e-1f,
                       -1.6668057665e-1f,
                       2.0000714765e-1f,
                       -2.4999993993e-1f,
                       3.3333331174e-1f});
    }

    static F withLogSpecials(F x, F y)
    {
        const auto zero = splat(0.f);
        y = F::select(x.equals(zero), splat(-infinity), y);
        y = F::select(x.lessThan(zero), splat(quietNaN), y);
        y = F::select(x.equals(splat(infinity)), x, y);
        return F::select(x.isNaN(), x, y);
    }
};

template <class B>
struct Exp
{
    using L = Lanes<B>;
    using F = typename L::F;

    static F apply(F x)
    {
        auto y = L::expOfClamped(L::clamp(x, expLowest, expHighest));
        y = L::flushSubnormal(y);
        y = F::select(L::splat(expHighest).lessThan(x), L::splat(infinity), y);
        return F::select(x.isNaN(), x, y);
    }
};

template <class B>
struct Exp2
{
    using L = Lanes<B>;
    using F = typename L::F;

    // 2^x = 2^n * 2^f with n = round(x) and |f| <= 1/2 exact, so an integer
    // x gives exactly 2^x.
    static F apply(F x)
    {
        const auto clamped = L::clamp(x, exp2Lowest, exp2Highest);
        const auto n = L::roundToNearest(clamped);
        const auto f = clamped - n;
        const auto p = L::horner(f,
                                 {1.535336188319500e-4f,
                                  1.339887440266574e-3f,
                                  9.618437357674640e-3f,
                                  5.550332471162809e-2f,
                                  2.402264791363012e-1f,
                                  6.931472028550421e-1f});
        const auto twoToF = f * p + L::splat(1.f);

        auto y = L::flushSubnormal(L::scaleByPowerOfTwo(twoToF, n.truncate()));
        y = F::select(L::splat(exp2Highest).lessThan(x), L::splat(infinity), y);
        return F::select(x.isNaN(), x, y);
    }
};

template <class B>
struct Log
{
    using L = Lanes<B>;
    using F = typename L::F;

    static F apply(F x)
    {
        const auto [m, e] = L::decompose(x);
        const auto m2 = m * m;

        auto y = L::logPolynomial(m) * m * m2;
        y = y + e * L::splat(ln2Lo);
        y = y - L::splat(0.5f) * m2;
        const auto result = (m + y) + e * L::splat(ln2Hi);

        return L::withLogSpecials(x, result);
    }
};

template <class B>
struct Log2
{
    using L = Lanes<B>;
    using F = typename L::F;

    // log2(1 + m) = log(1 + m) * log2(e), with log2(e) split as 1 + 0.4427 so
    // the large terms are added exactly; log2 of a power of two is exact.
    static F apply(F x)
    {
        const auto [m, e] = L::decompose(x);
        const auto m2 = m * m;

        auto y = m * (m2 * L::logPolynomial(m));
        y = y - L::splat(0.5f) * m2;

        auto result = y * L::splat(log2OfEMinusOne);
        result = result + m * L::splat(log2OfEMinusOne);
        result = result + y;
        result = result + m;
        result = result + e;

        return L::withLogSpecials(x, result);
    }
};

template <class B, bool Cosine>
struct SinCos
{
    using L = Lanes<B>;
    using F = typename L::F;
    using I = typename L::I;

    // x - j pi/2 as an unevaluated sum hi + lo. The first two chunk
    // subtractions are exact; each later one keeps its rounding error in lo
    // (Fast2Sum, error-free here because every j * chunk is exact), so r is
    // accurate both when it is near pi/4 and when it nearly cancels to zero.
    struct Reduced
    {
        F hi;
        F lo;
    };

    static Reduced reduce(F x, F j)
    {
        auto hi = (x - j * L::splat(halfPi1)) - j * L::splat(halfPi2);
        auto lo = L::splat(0.f);

        for (const auto chunk: {halfPi3, halfPi4, halfPi5})
        {
            const auto product = j * L::splat(chunk);
            const auto difference = hi - product;
            lo = lo + ((hi - difference) - product);
            hi = difference;
        }

        lo = lo - j * L::splat(halfPi6);
        const auto r = hi + lo;
        return {r, (hi - r) + lo};
    }

    // x = j pi/2 + r with |r| <= pi/4, then the sine or cosine polynomial of r
    // chosen and signed by the quadrant. The low part of r enters each
    // polynomial as its first-order term: sin(r + d) ~ sin r + d cos r and
    // cos(r + d) ~ cos r - d r.
    static F apply(F x)
    {
        const auto reducible = L::abs(x).lessEqual(L::splat(sinCosLargest));
        const auto reduced = F::select(reducible, x, L::splat(0.f));

        const auto j = L::roundToNearest(reduced * L::splat(twoOverPi));
        const auto [r, d] = reduce(reduced, j);

        auto quadrant = j.truncate();
        if constexpr (Cosine)
            quadrant = quadrant + L::splatInt(1);

        const auto z = r * r;
        const auto cosR = ((L::horner(z,
                                      {2.443315711809948e-5f,
                                       -1.388731625493765e-3f,
                                       4.166664568298827e-2f})
                                * z * z
                            - L::splat(0.5f) * z)
                           - d * r)
                          + L::splat(1.f);
        const auto sinR =
            r
            + (L::horner(z, {-1.9515295891e-4f, 8.3321608736e-3f, -1.6666654611e-1f})
                   * z * r
               + d * cosR);

        auto y = F::select(L::template maskOfBit<0>(quadrant), cosR, sinR);
        y = L::flipSign(y, L::template maskOfBit<1>(quadrant).template shl<31>());

        if constexpr (!Cosine)
            y = F::select(x.equals(L::splat(0.f)), x, y);

        y = F::select(L::abs(x).equals(L::splat(infinity)), L::splat(quietNaN), y);
        return F::select(x.isNaN(), x, y);
    }
};

template <class B>
using Sin = SinCos<B, false>;

template <class B>
using Cos = SinCos<B, true>;

template <class B>
struct Tanh
{
    using L = Lanes<B>;
    using F = typename L::F;

    // Odd, so computed on |x| and signed at the end (tanh(-0) = -0). A
    // polynomial below 0.625, 1 - 2 / (e^2|x| + 1) above it.
    static F apply(F x)
    {
        const auto ax = L::abs(x);
        const auto saturation = L::splat(tanhSaturation);
        const auto z = F::select(ax.lessThan(saturation), ax, saturation);

        const auto one = L::splat(1.f);
        const auto viaExp = one - L::splat(2.f) / (L::expOfClamped(z + z) + one);

        const auto z2 = z * z;
        const auto viaPolynomial = L::horner(z2,
                                             {-5.70498872745e-3f,
                                              2.06390887954e-2f,
                                              -5.37397155531e-2f,
                                              1.33314422036e-1f,
                                              -3.33332819422e-1f})
                                       * z2 * z
                                   + z;

        const auto small = z.lessThan(L::splat(tanhExpThreshold));
        const auto y = F::select(small, viaPolynomial, viaExp);
        return F::select(x.isNaN(), x, L::withSign(y, L::signOf(x)));
    }
};

// out[i] = Fn(in[i]). The main loop takes B::F32::lanes floats per step; the
// remainder goes through the Scalar backend, which is also the whole
// algorithm when B is Scalar. `out` may equal `in`.
template <template <class> class Fn, class B>
void mapImpl(const float* in, float* out, int count)
{
    using F = typename B::F32;

    auto i = 0;
    for (; i + F::lanes <= count; i += F::lanes)
        F::store(out + i, Fn<B>::apply(F::load(in + i)));

    if constexpr (F::lanes > 1)
        if (i < count)
            mapImpl<Fn, backend::Scalar>(in + i, out + i, count - i);
}

} // namespace
} // namespace esimd::kernels
