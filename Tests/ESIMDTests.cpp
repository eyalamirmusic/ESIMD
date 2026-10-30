#include <ESIMD/Backends.h>
#include <ESIMD/Ops.h>

#include <NanoTest/NanoTest.h>
#include <ea_data_structures/Structures/Array.h>
#include <ea_data_structures/Structures/Span.h>
#include <ea_data_structures/ea_data_structures.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace nano;

namespace
{
using Pixels = EA::Vector<std::uint8_t>;
using SwapFn = void (*)(const std::uint8_t*, std::uint8_t*, int);
using ResizeFn = void (*)(const std::uint8_t*, int, int, std::uint8_t*, int, int);

// Pixel counts chosen to straddle every backend's lane width (SSE2/NEON = 4,
// AVX2 = 8): zero, sub-lane, exact multiples, and odd remainders.
constexpr int kSwapSizes[] = {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 64, 1000};

Pixels makePixels(int pixelCount)
{
    auto data = Pixels(pixelCount * 4);
    for (int i = 0; i < data.size(); ++i)
        data[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
    return data;
}

// Independent reference: swap byte 0 and byte 2 of every 4-byte pixel.
Pixels swapReference(const Pixels& in)
{
    auto out = in;
    const auto pixelCount = out.size() / 4;
    for (int p = 0; p < pixelCount; ++p)
    {
        const auto red = out[p * 4 + 0];
        out[p * 4 + 0] = out[p * 4 + 2];
        out[p * 4 + 2] = red;
    }
    return out;
}

void checkSwapAgainstScalar(SwapFn fn)
{
    for (auto count: kSwapSizes)
    {
        const auto in = makePixels(count);
        auto got = Pixels(in.size());
        auto oracle = Pixels(in.size());
        fn(in.data(), got.data(), count);
        esimd::backends::swapRedBlue_scalar(in.data(), oracle.data(), count);
        check(got == oracle);
    }
}

struct ResizeCase
{
    int srcW, srcH, dstW, dstH;
};

// Up, down, identity, extreme aspect, and prime sizes to stress edge clamping
// and the per-pixel coordinate math in both directions.
constexpr ResizeCase kResizeCases[] = {
    {1, 1, 4, 4},
    {4, 4, 1, 1},
    {2, 2, 1, 1},
    {8, 8, 8, 8},
    {16, 9, 7, 13},
    {7, 13, 16, 9},
    {1, 10, 5, 3},
    {10, 1, 3, 5},
    {17, 17, 5, 5},
    {5, 5, 17, 17},
    {64, 48, 33, 21},
    {3, 3, 9, 9},
};

Pixels makeImage(int w, int h)
{
    auto data = Pixels(w * h * 4);
    for (int i = 0; i < data.size(); ++i)
        data[i] = static_cast<std::uint8_t>((i * 53 + (i / 4) * 7 + 19) & 0xFF);
    return data;
}

Pixels runResize(ResizeFn fn, const Pixels& src, const ResizeCase& c)
{
    auto dst = Pixels(c.dstW * c.dstH * 4);
    fn(src.data(), c.srcW, c.srcH, dst.data(), c.dstW, c.dstH);
    return dst;
}

void checkResizeAgainstScalar(ResizeFn fn)
{
    for (const auto& c: kResizeCases)
    {
        const auto src = makeImage(c.srcW, c.srcH);
        const auto got = runResize(fn, src, c);
        const auto oracle =
            runResize(&esimd::backends::resizeBilinear_scalar, src, c);
        check(got == oracle);
    }
}
} // namespace

auto tScalarSwapMatchesReference = test("ESIMD/scalarSwapMatchesReference") = []
{
    for (auto count: kSwapSizes)
    {
        const auto in = makePixels(count);
        auto got = Pixels(in.size());
        esimd::backends::swapRedBlue_scalar(in.data(), got.data(), count);
        check(got == swapReference(in));
    }
};

auto tBaselineSwapMatchesScalar = test("ESIMD/baselineSwapMatchesScalar") = []
{
#if defined(__x86_64__) || defined(_M_X64)
    checkSwapAgainstScalar(&esimd::backends::swapRedBlue_sse2);
#elif defined(__aarch64__) || defined(_M_ARM64)
    checkSwapAgainstScalar(&esimd::backends::swapRedBlue_neon);
#endif
};

auto tAvx2SwapMatchesScalar = test("ESIMD/avx2SwapMatchesScalar") = []
{
#if defined(ESIMD_HAS_AVX2)
    if (esimd::cpu::hasAvx2Fma())
        checkSwapAgainstScalar(&esimd::backends::swapRedBlue_avx2);
#endif
};

auto tSwapDispatchMatchesScalar = test("ESIMD/swapDispatchMatchesScalar") = []
{
    const auto in = makePixels(257);
    auto got = Pixels(in.size());
    auto oracle = Pixels(in.size());
    esimd::swapRedBlue(in.data(), got.data(), 257);
    esimd::backends::swapRedBlue_scalar(in.data(), oracle.data(), 257);
    check(got == oracle);
};

auto tSwapTwiceRestoresOriginal = test("ESIMD/swapTwiceRestoresOriginal") = []
{
    auto buffer = makePixels(123);
    const auto original = buffer;
    esimd::swapRedBlue(buffer.data(), buffer.data(), 123);
    esimd::swapRedBlue(buffer.data(), buffer.data(), 123);
    check(buffer == original);
};

auto tBaselineResizeMatchesScalar = test("ESIMD/baselineResizeMatchesScalar") = []
{
#if defined(__x86_64__) || defined(_M_X64)
    checkResizeAgainstScalar(&esimd::backends::resizeBilinear_sse2);
#elif defined(__aarch64__) || defined(_M_ARM64)
    checkResizeAgainstScalar(&esimd::backends::resizeBilinear_neon);
#endif
};

namespace
{
using WarpFn =
    void (*)(const std::uint8_t*, int, int, const float*, std::uint8_t*, int, int);

struct WarpCase
{
    int srcW, srcH, dstW, dstH;
    float m[6];
};

// Identity, scale+translate, rotate-ish, shear, a degenerate source width, and
// odd sizes -- exercising edge clamping and the affine coordinate math.
constexpr WarpCase kWarpCases[] = {
    {16, 16, 16, 16, {1.f, 0.f, 0.f, 0.f, 1.f, 0.f}},
    {16, 16, 20, 12, {0.7f, 0.f, 1.f, 0.f, 0.7f, 1.f}},
    {16, 16, 16, 16, {0.9f, -0.3f, 2.f, 0.3f, 0.9f, 1.f}},
    {16, 16, 24, 24, {0.5f, 0.2f, 0.f, 0.1f, 0.5f, 0.f}},
    {1, 16, 8, 8, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f}},
    {33, 17, 9, 21, {1.3f, 0.1f, -2.f, -0.2f, 1.1f, 3.f}},
};

Pixels runWarp(WarpFn fn, const Pixels& src, const WarpCase& c)
{
    auto dst = Pixels(c.dstW * c.dstH * 4);
    fn(src.data(), c.srcW, c.srcH, c.m, dst.data(), c.dstW, c.dstH);
    return dst;
}

void checkWarpAgainstScalar(WarpFn fn)
{
    for (const auto& c: kWarpCases)
    {
        const auto src = makeImage(c.srcW, c.srcH);
        const auto got = runWarp(fn, src, c);
        const auto oracle =
            runWarp(&esimd::backends::warpAffineInverse_scalar, src, c);
        check(got == oracle);
    }
}
} // namespace

auto tBaselineWarpMatchesScalar = test("ESIMD/baselineWarpMatchesScalar") = []
{
#if defined(__x86_64__) || defined(_M_X64)
    checkWarpAgainstScalar(&esimd::backends::warpAffineInverse_sse2);
#elif defined(__aarch64__) || defined(_M_ARM64)
    checkWarpAgainstScalar(&esimd::backends::warpAffineInverse_neon);
#endif
};

auto tResizeIdentityReturnsSource = test("ESIMD/resizeIdentityReturnsSource") = []
{
    // Same-size bilinear with half-pixel centers samples each pixel exactly, so
    // every backend must return the source untouched.
    const ResizeCase cases[] = {{5, 4, 5, 4}, {16, 9, 16, 9}};
    for (const auto& c: cases)
    {
        const auto src = makeImage(c.srcW, c.srcH);
        auto dst = Pixels(src.size());
        esimd::resizeBilinear(
            src.data(), c.srcW, c.srcH, dst.data(), c.dstW, c.dstH);
        check(dst == src);
    }
};

namespace
{
// A length that is not a multiple of any backend's vector width, to exercise the
// scalar tail. Integer-valued data keeps every result exact (so the comparison
// is fusion-agnostic for multiplyAdd).
constexpr int kArrayCount = 1003;

EA::Vector<float> ramp(int stride, int offset)
{
    auto v = EA::Vector<float>(kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        v[i] = static_cast<float>((i % stride) + offset);
    return v;
}
} // namespace

auto tArrayAddMatchesReference = test("ESIMD/arrayAddMatchesReference") = []
{
    const auto a = ramp(17, 1);
    const auto b = ramp(23, 0);
    auto out = EA::Vector<float>(kArrayCount);
    esimd::add(a.data(), b.data(), out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == a[i] + b[i]);
};

auto tArraySubtractMatchesReference =
    test("ESIMD/arraySubtractMatchesReference") = []
{
    const auto a = ramp(29, 5);
    const auto b = ramp(13, 0);
    auto out = EA::Vector<float>(kArrayCount);
    esimd::subtract(a.data(), b.data(), out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == a[i] - b[i]);
};

auto tArrayMultiplyMatchesReference =
    test("ESIMD/arrayMultiplyMatchesReference") = []
{
    const auto a = ramp(11, 0);
    const auto b = ramp(7, 1);
    auto out = EA::Vector<float>(kArrayCount);
    esimd::multiply(a.data(), b.data(), out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == a[i] * b[i]);
};

auto tArrayMultiplyByScalarMatches =
    test("ESIMD/arrayMultiplyByScalarMatchesReference") = []
{
    const auto a = ramp(19, 2);
    auto out = EA::Vector<float>(kArrayCount);
    esimd::multiplyByScalar(a.data(), 3.f, out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == a[i] * 3.f);
};

auto tArrayMultiplyAddMatchesReference =
    test("ESIMD/arrayMultiplyAddMatchesReference") = []
{
    const auto a = ramp(11, 0);
    const auto b = ramp(7, 1);
    const auto c = ramp(5, 0);
    auto out = EA::Vector<float>(kArrayCount);
    esimd::multiplyAdd(a.data(), b.data(), c.data(), out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == a[i] * b[i] + c[i]);
};

auto tArrayMultiplyAddScalarMatchesReference =
    test("ESIMD/arrayMultiplyAddScalarMatchesReference") = []
{
    const auto a = ramp(11, 0);
    const auto c = ramp(5, 0);
    auto out = EA::Vector<float>(kArrayCount);
    esimd::multiplyAdd(a.data(), 3.f, c.data(), out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == a[i] * 3.f + c[i]);
};

// out == c is the accumulate-in-place shape the docs promise (out[i] += a[i]*b).
auto tArrayMultiplyAddScalarAccumulatesInPlace =
    test("ESIMD/arrayMultiplyAddScalarAccumulatesInPlace") = []
{
    const auto a = ramp(11, 0);
    const auto before = ramp(5, 0);
    auto out = before;
    esimd::multiplyAdd(a.data(), 2.f, out.data(), out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == before[i] + a[i] * 2.f);
};

// t = 0.25 keeps every intermediate exactly representable, so the comparison
// stays exact whether or not the compiler contracts the multiply-add.
auto tArrayLerpMatchesReference = test("ESIMD/arrayLerpMatchesReference") = []
{
    const auto a = ramp(17, 1);
    const auto b = ramp(23, 0);
    auto out = EA::Vector<float>(kArrayCount);
    esimd::lerp(a.data(), b.data(), 0.25f, out.data(), kArrayCount);
    for (int i = 0; i < kArrayCount; ++i)
        check(out[i] == a[i] + 0.25f * (b[i] - a[i]));
};

// Integer-valued data keeps the double accumulation exact regardless of the
// four-lane interleave, so the comparison against a sequential sum is exact.
auto tArraySumOfSquaresMatchesReference =
    test("ESIMD/arraySumOfSquaresMatchesReference") = []
{
    const auto a = ramp(17, -8); // mixed signs
    auto expected = 0.0;
    for (int i = 0; i < kArrayCount; ++i)
        expected += (double) a[i] * (double) a[i];

    check(esimd::sumOfSquares(a.data(), kArrayCount) == expected);
    check(esimd::sumOfSquares(a.data(), 0) == 0.0);

    // Counts around the four-lane width exercise the tail loop.
    for (auto count: {1, 2, 3, 4, 5, 7, 8, 9})
    {
        auto partial = 0.0;
        for (int i = 0; i < count; ++i)
            partial += (double) a[i] * (double) a[i];
        check(esimd::sumOfSquares(a.data(), count) == partial);
    }
};

auto tArrayPeakAbsMatchesReference = test("ESIMD/arrayPeakAbsMatchesReference") = []
{
    auto a = ramp(29, -14); // mixed signs; peak is a negative value's magnitude
    check(esimd::peakAbs(a.data(), kArrayCount) == 14.f);
    check(esimd::peakAbs(a.data(), 0) == 0.f);

    // The peak can land in any lane, including the tail.
    a[kArrayCount - 1] = -99.f;
    check(esimd::peakAbs(a.data(), kArrayCount) == 99.f);
    a[2] = 200.f;
    check(esimd::peakAbs(a.data(), kArrayCount) == 200.f);
};

// The buffer-level helpers in Ops.h: each forwards to the raw primitive, in
// place on its first argument.
auto tOpsHelpersForwardToPrimitives =
    test("ESIMD/opsHelpersForwardToPrimitives") = []
{
    const auto a = ramp(17, 1);
    const auto b = ramp(23, 0);

    auto dst = a;
    esimd::add(dst, b);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == a[i] + b[i]);

    dst = a;
    esimd::subtract(dst, b);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == a[i] - b[i]);

    dst = a;
    esimd::multiply(dst, b);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == a[i] * b[i]);

    dst = a;
    esimd::multiply(dst, 3.f);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == a[i] * 3.f);

    dst = a;
    esimd::multiplyAdd(dst, b, 2.f);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == a[i] + b[i] * 2.f);

    dst = a;
    esimd::multiplyAdd(dst, a, b);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == a[i] + a[i] * b[i]);

    dst = a;
    esimd::lerp(dst, b, 0.25f);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == a[i] + 0.25f * (b[i] - a[i]));

    check(esimd::sumOfSquares(a) == esimd::sumOfSquares(a.data(), kArrayCount));
    check(esimd::peakAbs(a) == esimd::peakAbs(a.data(), kArrayCount));
};

// Mismatched sizes process the common prefix and never touch the tail.
auto tOpsHelpersStopAtShortestBuffer =
    test("ESIMD/opsHelpersStopAtShortestBuffer") = []
{
    const auto a = ramp(17, 1);
    auto shorter = EA::Vector<float>(kArrayCount / 2);
    for (int i = 0; i < shorter.size(); ++i)
        shorter[i] = 1.f;

    auto dst = a;
    esimd::add(dst, shorter);
    for (int i = 0; i < kArrayCount; ++i)
        check(dst[i] == (i < kArrayCount / 2 ? a[i] + 1.f : a[i]));
};

// Ops.h's concepts are container-agnostic: EA::Array (fixed size), EA::Span
// (a view, mutable and const) and a Span's subviews all satisfy them, and the
// int sizes they report drive the primitives directly.
auto tOpsHelpersAcceptArraysAndSpans =
    test("ESIMD/opsHelpersAcceptArraysAndSpans") = []
{
    auto gains = EA::Array<float, 4> {1.f, 2.f, 3.f, 4.f};
    const auto factors = EA::Array<float, 4> {2.f, 2.f, 4.f, 4.f};
    esimd::multiply(gains, factors);
    check(gains[0] == 2.f);
    check(gains[1] == 4.f);
    check(gains[2] == 12.f);
    check(gains[3] == 16.f);
    check(esimd::peakAbs(gains) == 16.f);

    auto storage = ramp(17, 1);
    const auto original = storage;
    const auto whole = EA::Span<float> {storage};
    check(esimd::peakAbs(whole) == esimd::peakAbs(storage));
    check(esimd::sumOfSquares(EA::Span<const float> {storage})
          == esimd::sumOfSquares(storage));

    // A subview bounds the work: only its own prefix of the buffer changes.
    auto head = whole.first(8);
    esimd::multiply(head, 2.f);
    for (int i = 0; i < storage.size(); ++i)
        check(storage[i] == (i < 8 ? original[i] * 2.f : original[i]));

    esimd::subtract(storage, EA::Span<const float> {head});
    for (int i = 0; i < storage.size(); ++i)
        check(storage[i] == (i < 8 ? 0.f : original[i]));
};

// --- Byte scanning ---

namespace
{
using ScanFn = int (*)(const std::uint8_t*, int, const esimd::ByteClass&);
using Bytes = EA::Vector<std::uint8_t>;

struct ScanBackend
{
    const char* name;
    ScanFn findFirst;
    ScanFn findFirstNot;
    ScanFn countOf;
};

// Every backend the running CPU can execute. The scalar one is the oracle the
// others are held to, and it is listed too so the test also runs it against
// the independent reference below.
EA::Vector<ScanBackend> scanBackends()
{
    auto backends = EA::Vector<ScanBackend> {};
    backends.add({"scalar",
                  &esimd::backends::findFirst_scalar,
                  &esimd::backends::findFirstNot_scalar,
                  &esimd::backends::countOf_scalar});
#if defined(__x86_64__) || defined(_M_X64)
    backends.add({"sse2",
                  &esimd::backends::findFirst_sse2,
                  &esimd::backends::findFirstNot_sse2,
                  &esimd::backends::countOf_sse2});
#if defined(ESIMD_HAS_AVX2)
    if (esimd::cpu::hasAvx2Fma())
        backends.add({"avx2",
                      &esimd::backends::findFirst_avx2,
                      &esimd::backends::findFirstNot_avx2,
                      &esimd::backends::countOf_avx2});
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    backends.add({"neon",
                  &esimd::backends::findFirst_neon,
                  &esimd::backends::findFirstNot_neon,
                  &esimd::backends::countOf_neon});
#endif
    return backends;
}

// Classes covering every feature of a ByteClass: single values, the full
// eight-value capacity, one and two ranges, a value/range mix, the open-ended
// builders, and the empty class.
EA::Vector<esimd::ByteClass> scanClasses()
{
    using namespace esimd;
    auto classes = EA::Vector<ByteClass> {};
    classes.add(ByteClass {});
    classes.add('"');
    classes.add(anyOf('"', '\\') | below(0x20));
    classes.add(anyOf(' ', '\t', '\n', '\r'));
    classes.add(anyOf(1, 2, 3, 4, 5, 6, 7, 8));
    classes.add(inRange('0', '9'));
    classes.add(inRange('a', 'z') | inRange('A', 'Z'));
    classes.add(inRange('0', '9') | anyOf('.', 'e', 'E', '+', '-'));
    classes.add(above(0x7F));
    classes.add(below(1));
    classes.add(above(254));
    classes.add(inRange(0, 255));
    return classes;
}

// The reference the scalar backend is held to: ByteClass::contains, byte by byte.
int referenceFindFirst(const Bytes& data, const esimd::ByteClass& cls, bool negate)
{
    for (int i = 0; i < data.size(); ++i)
        if (cls.contains(data[i]) != negate)
            return i;
    return data.size();
}

int referenceCountOf(const Bytes& data, const esimd::ByteClass& cls)
{
    auto total = 0;
    for (int i = 0; i < data.size(); ++i)
        total += cls.contains(data[i]) ? 1 : 0;
    return total;
}

// A buffer walking through every byte value, so each class sees members and
// non-members in every lane position.
Bytes makeScanBytes(int count)
{
    auto data = Bytes(count);
    for (int i = 0; i < count; ++i)
        data[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
    return data;
}

Bytes filledBytes(int count, int value)
{
    auto data = Bytes(count);
    for (int i = 0; i < count; ++i)
        data[i] = static_cast<std::uint8_t>(value);
    return data;
}

// Some byte in / not in the class, or -1 when there is none.
int someByte(const esimd::ByteClass& cls, bool inside)
{
    for (int b = 0; b < 256; ++b)
        if (cls.contains(static_cast<std::uint8_t>(b)) == inside)
            return b;
    return -1;
}

// Lengths that straddle every lane width (16 and 32) and the scalar tail.
constexpr int kScanSizes[] = {
    0, 1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 257};

void checkScanBackendAgainstScalar(const ScanBackend& backend)
{
    for (const auto& cls: scanClasses())
    {
        for (auto size: kScanSizes)
        {
            const auto data = makeScanBytes(size);
            check(backend.findFirst(data.data(), size, cls)
                  == esimd::backends::findFirst_scalar(data.data(), size, cls));
            check(backend.findFirstNot(data.data(), size, cls)
                  == esimd::backends::findFirstNot_scalar(data.data(), size, cls));
            check(backend.countOf(data.data(), size, cls)
                  == esimd::backends::countOf_scalar(data.data(), size, cls));
        }

        // A single hit planted at every position of a hit-free buffer (and no
        // hit at all): the answer must be exactly that position on every lane.
        const auto inside = someByte(cls, true);
        const auto outside = someByte(cls, false);
        if (inside < 0 || outside < 0)
            continue;

        constexpr auto size = 70;
        for (auto hit = 0; hit <= size; ++hit)
        {
            auto data = filledBytes(size, outside);
            if (hit < size)
                data[hit] = static_cast<std::uint8_t>(inside);
            check(backend.findFirst(data.data(), size, cls) == hit);
            check(backend.countOf(data.data(), size, cls) == (hit < size ? 1 : 0));

            auto inverse = filledBytes(size, inside);
            if (hit < size)
                inverse[hit] = static_cast<std::uint8_t>(outside);
            check(backend.findFirstNot(inverse.data(), size, cls) == hit);
        }
    }
}
} // namespace

auto tByteClassBuilders = test("ESIMD/byteClassBuilders") = []
{
    using namespace esimd;

    const auto quoteOrEscape = anyOf('"', '\\');
    check(quoteOrEscape.contains('"'));
    check(quoteOrEscape.contains('\\'));
    check(!quoteOrEscape.contains('a'));

    const ByteClass single = '"';
    check(single.contains('"'));
    check(!single.contains('\''));

    const auto control = below(0x20);
    check(control.contains(0));
    check(control.contains(0x1F));
    check(!control.contains(0x20));

    const auto nonAscii = above(0x7F);
    check(!nonAscii.contains(0x7F));
    check(nonAscii.contains(0x80));
    check(nonAscii.contains(0xFF));

    const auto digits = inRange('0', '9');
    check(digits.contains('0'));
    check(digits.contains('9'));
    check(!digits.contains('/'));
    check(!digits.contains(':'));

    // A union keeps every member of both sides, values and ranges alike.
    const auto stringSpecial = quoteOrEscape | control;
    check(stringSpecial.contains('"'));
    check(stringSpecial.contains('\n'));
    check(!stringSpecial.contains('x'));

    const auto letters = inRange('a', 'z') | inRange('A', 'Z');
    check(letters.contains('a'));
    check(letters.contains('Z'));
    check(!letters.contains('5'));

    // A high byte as a (possibly signed) char still lands on its byte value.
    const auto leadByte = anyOf('\xC4');
    check(leadByte.contains(0xC4));

    check(!ByteClass {}.contains(0));
    check(!ByteClass {}.contains(0xFF));
};

auto tScalarScanMatchesReference = test("ESIMD/scalarScanMatchesReference") = []
{
    for (const auto& cls: scanClasses())
    {
        for (auto size: kScanSizes)
        {
            const auto data = makeScanBytes(size);
            check(esimd::backends::findFirst_scalar(data.data(), size, cls)
                  == referenceFindFirst(data, cls, false));
            check(esimd::backends::findFirstNot_scalar(data.data(), size, cls)
                  == referenceFindFirst(data, cls, true));
            check(esimd::backends::countOf_scalar(data.data(), size, cls)
                  == referenceCountOf(data, cls));
        }
    }
};

auto tEveryScanBackendMatchesScalar =
    test("ESIMD/everyScanBackendMatchesScalar") = []
{
    for (const auto& backend: scanBackends())
        checkScanBackendAgainstScalar(backend);
};

auto tScanDispatchMatchesScalar = test("ESIMD/scanDispatchMatchesScalar") = []
{
    const auto data = makeScanBytes(1001);
    for (const auto& cls: scanClasses())
    {
        check(esimd::findFirst(data.data(), data.size(), cls)
              == esimd::backends::findFirst_scalar(data.data(), data.size(), cls));
        check(
            esimd::findFirstNot(data.data(), data.size(), cls)
            == esimd::backends::findFirstNot_scalar(data.data(), data.size(), cls));
        check(esimd::countOf(data.data(), data.size(), cls)
              == esimd::backends::countOf_scalar(data.data(), data.size(), cls));
    }
};

// The byte helpers in Ops.h accept any contiguous buffer of a byte-like type
// and report positions into it, with an optional start offset.
auto tOpsByteHelpersAcceptStringsAndBuffers =
    test("ESIMD/opsByteHelpersAcceptStringsAndBuffers") = []
{
    using namespace esimd;
    static constexpr auto stringSpecial = anyOf('"', '\\') | below(0x20);
    static constexpr auto whitespace = anyOf(' ', '\t', '\n', '\r');

    const auto text = std::string {"  \t\"hello\\world\"\n  tail"};
    const auto view = std::string_view {text};

    check(findFirst(text, '"') == 3);
    check(findFirst(view, stringSpecial) == 2); // the tab is a control char
    check(findFirst(view, stringSpecial, 4) == 9);
    check(findFirst(view, stringSpecial, 10) == 15);
    check(findFirst(view, stringSpecial, 16) == 16);
    check(findFirst(view, stringSpecial, 17) == view.size());
    check(findFirst(view, stringSpecial, 10'000) == view.size());
    check(findFirst(view, stringSpecial, -5) == 2);

    check(findFirstNot(view, whitespace) == 3);
    check(findFirstNot(view, whitespace, 16) == 19);
    check(findFirstNot(std::string_view {"   "}, whitespace) == 3);

    check(contains(view, '\\'));
    check(!contains(view, '!'));
    check(allOf(view, below(0x80)));
    check(!allOf(view, whitespace));
    check(allOf(std::string_view {}, whitespace));
    check(countOf(view, '"') == 2);
    check(countOf(view, whitespace) == 6);

    // Unsigned bytes, signed chars, std::byte and a Span all qualify.
    auto raw = EA::Vector<std::uint8_t> {0x41, 0xC4, 0xA0, 0x42};
    check(findFirst(raw, above(0x7F)) == 1);
    check(countOf(raw, above(0x7F)) == 2);
    check(!allOf(raw, below(0x80)));

    const auto chars = std::vector<char> {'a', 'b', '\n', 'c'};
    check(findFirst(chars, '\n') == 2);

    const auto bytes = std::vector<std::byte> {std::byte {1}, std::byte {9}};
    check(findFirst(bytes, 9) == 1);

    const auto span = EA::Span<const std::uint8_t> {raw};
    check(findFirst(span, 0x42) == 3);
    check(findFirst(span.first(2), 0x42) == 2);
};

// --- Math functions ---

namespace
{
using MathFn = void (*)(const float*, float*, int);
using Floats = EA::Vector<float>;

// One math function: the scalar oracle, every other backend the running CPU
// can execute (the dispatched entry point last), the double-precision
// reference and the ULP bound ESIMD.h documents.
struct MathFunction
{
    const char* name;
    MathFn scalar;
    double (*reference)(double);
    double ulpBound;
    EA::Vector<MathFn> backends;
};

void addBackend(EA::Vector<MathFunction>& functions,
                std::initializer_list<MathFn> perFunction)
{
    auto i = 0;
    for (auto fn: perFunction)
        functions[i++].backends.add(fn);
}

EA::Vector<MathFunction> mathFunctions()
{
    using namespace esimd::backends;

    auto functions = EA::Vector<MathFunction> {};
    functions.add({"exp", &exp_scalar, [](double x) { return std::exp(x); }, 1.0});
    functions.add(
        {"exp2", &exp2_scalar, [](double x) { return std::exp2(x); }, 1.25});
    functions.add({"log", &log_scalar, [](double x) { return std::log(x); }, 1.0});
    functions.add(
        {"log2", &log2_scalar, [](double x) { return std::log2(x); }, 1.5});
    functions.add({"sin", &sin_scalar, [](double x) { return std::sin(x); }, 1.3});
    functions.add({"cos", &cos_scalar, [](double x) { return std::cos(x); }, 1.3});
    functions.add(
        {"tanh", &tanh_scalar, [](double x) { return std::tanh(x); }, 1.35});

#if defined(__x86_64__) || defined(_M_X64)
    addBackend(functions,
               {&exp_sse2,
                &exp2_sse2,
                &log_sse2,
                &log2_sse2,
                &sin_sse2,
                &cos_sse2,
                &tanh_sse2});
#if defined(ESIMD_HAS_AVX2)
    if (esimd::cpu::hasAvx2Fma())
        addBackend(functions,
                   {&exp_avx2,
                    &exp2_avx2,
                    &log_avx2,
                    &log2_avx2,
                    &sin_avx2,
                    &cos_avx2,
                    &tanh_avx2});
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    addBackend(functions,
               {&exp_neon,
                &exp2_neon,
                &log_neon,
                &log2_neon,
                &sin_neon,
                &cos_neon,
                &tanh_neon});
#endif

    addBackend(functions,
               {&esimd::exp,
                &esimd::exp2,
                &esimd::log,
                &esimd::log2,
                &esimd::sin,
                &esimd::cos,
                &esimd::tanh});
    return functions;
}

// Integer-only generation, so the inputs are the same bits on every compiler,
// flag set and architecture.
struct BitSource
{
    std::uint32_t next()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }

    std::uint32_t state;
};

// Half arbitrary bit patterns (NaNs with payloads, infinities, subnormals,
// huge and tiny values), half finite values of magnitude 2^-16 to 2^16 where
// every function does real work.
Floats mathInputs(int count)
{
    auto bits = BitSource {12345u};
    auto data = Floats(count);
    for (int i = 0; i < count; ++i)
    {
        auto pattern = bits.next();
        if (i % 2 == 1)
            pattern = (pattern & 0x807FFFFFu) | ((111u + bits.next() % 32u) << 23);
        data[i] = std::bit_cast<float>(pattern);
    }
    return data;
}

Floats run(MathFn fn, const Floats& in)
{
    auto out = Floats(in.size());
    fn(in.data(), out.data(), in.size());
    return out;
}

bool sameBits(const Floats& a, const Floats& b)
{
    return a.size() == b.size()
           && std::memcmp(a.data(), b.data(), sizeof(float) * a.size()) == 0;
}

// Element counts straddling every lane width (4 and 8), the scalar tail and
// a few longer runs.
constexpr int kMathSizes[] = {
    0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1001};

// The distance from `got` to the exact value in units of the float spacing at
// the exact value (the subnormal spacing below FLT_MIN).
double ulpError(float got, double exact)
{
    const auto magnitude = std::fabs(exact);
    auto spacing = std::ldexp(1.0, -149);
    if (magnitude >= std::ldexp(1.0, -126))
    {
        auto exponent = 0;
        std::frexp(magnitude, &exponent);
        spacing = std::ldexp(1.0, exponent - 24);
    }
    return std::fabs(static_cast<double>(got) - exact) / spacing;
}

void addGrid(Floats& data, double low, double high, int steps)
{
    for (int k = 0; k <= steps; ++k)
        data.add(static_cast<float>(low + (high - low) * k / steps));
}

void addRandom(Floats& data, double low, double high, int count)
{
    auto bits = BitSource {777u};
    for (int k = 0; k < count; ++k)
        data.add(static_cast<float>(low + (high - low) * (bits.next() * 0x1p-32)));
}

// The floats nearest each of the first `count` multiples of pi/2, and two
// neighbours either side: where sin or cos nearly cancels to zero.
void addNearHalfPiMultiples(Floats& data, int count)
{
    for (int k = 1; k <= count; ++k)
    {
        auto x = static_cast<float>(k * 1.57079632679489661923);
        for (int step = 0; step < 2; ++step)
            x = std::nextafter(x, 0.f);
        for (int step = 0; step < 5; ++step, x = std::nextafter(x, 1e9f))
        {
            data.add(x);
            data.add(-x);
        }
    }
}

// Positive floats spread evenly over every binade, subnormals included.
void addEveryBinade(Floats& data, int count)
{
    auto bits = BitSource {4242u};
    for (int k = 0; k < count; ++k)
        data.add(std::bit_cast<float>(bits.next() % 0x7F800000u));
}

// The sweep each function's accuracy is measured over: its useful range,
// plus a dense grid of the places a polynomial or a reduction goes wrong.
Floats accuracyInputs(std::string_view name)
{
    auto data = Floats {};
    if (name == "exp")
    {
        addRandom(data, -87.3, 88.72, 100'000);
        addGrid(data, -1e-3, 1e-3, 20'000);
        addGrid(data, 88.0, 88.7228, 20'000);
        addGrid(data, -87.33, -86.0, 20'000);
    }
    else if (name == "exp2")
    {
        addRandom(data, -126.0, 127.99, 100'000);
        addGrid(data, -1e-3, 1e-3, 20'000);
        addGrid(data, -3.0, 3.0, 60'000);
        addGrid(data, 127.0, 127.999, 20'000);
    }
    else if (name == "log" || name == "log2")
    {
        addEveryBinade(data, 100'000);
        addGrid(data, 0.99, 1.01, 40'000);
        addGrid(data, 0.5, 2.0, 40'000);
    }
    else if (name == "sin" || name == "cos")
    {
        addRandom(data, -65536.0, 65536.0, 100'000);
        addGrid(data, -10.0, 10.0, 40'000);
        addGrid(data, -1e-3, 1e-3, 10'000);
        addNearHalfPiMultiples(data, 41'000);
    }
    else if (name == "tanh")
    {
        addRandom(data, -12.0, 12.0, 100'000);
        addGrid(data, -1e-3, 1e-3, 10'000);
        addGrid(data, 0.6, 0.65, 20'000);
        addGrid(data, -0.65, -0.6, 20'000);
        addGrid(data, 8.0, 10.0, 20'000);
    }
    return data;
}

// exp and exp2 flush results below FLT_MIN to zero, as documented.
bool flushedAsDocumented(float got, double exact)
{
    return got == 0.f && exact < std::ldexp(1.0, -126) * (1.0 + 0x1p-20);
}

double maxUlpError(const MathFunction& fn)
{
    const auto in = accuracyInputs(fn.name);
    const auto out = run(fn.scalar, in);
    auto worst = 0.0;
    for (int i = 0; i < in.size(); ++i)
    {
        const auto exact = fn.reference(static_cast<double>(in[i]));
        if (!flushedAsDocumented(out[i], exact))
            worst = std::max(worst, ulpError(out[i], exact));
    }
    return worst;
}

// FNV-1a over the output bits.
std::uint64_t hashOf(const Floats& values)
{
    auto hash = 14695981039346656037ull;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    for (std::size_t i = 0; i < sizeof(float) * values.size(); ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    return hash;
}

Floats filledFloats(int count, float value)
{
    auto data = Floats(count);
    for (int i = 0; i < count; ++i)
        data[i] = value;
    return data;
}

struct SpecialCase
{
    float in;
    float expected;
};

// Every lane of every backend gets the case: the value fills a buffer longer
// than the widest vector, and the odd length sends some through the tail.
void checkSpecialCases(const MathFunction& fn,
                       std::initializer_list<SpecialCase> cases)
{
    auto all = EA::Vector<MathFn> {fn.scalar};
    for (auto backend: fn.backends)
        all.add(backend);

    for (const auto& c: cases)
    {
        for (auto backend: all)
            for (auto y: run(backend, filledFloats(19, c.in)))
            {
                if (std::isnan(c.expected))
                    check(std::isnan(y));
                else
                    check(std::bit_cast<std::uint32_t>(y)
                          == std::bit_cast<std::uint32_t>(c.expected));
            }
    }
}

const MathFunction& mathFunction(const EA::Vector<MathFunction>& functions,
                                 std::string_view name)
{
    for (const auto& fn: functions)
        if (name == fn.name)
            return fn;
    return functions[0];
}

constexpr auto kInf = std::numeric_limits<float>::infinity();
constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr auto kMinNormal = std::numeric_limits<float>::min();

void checkInPlaceHelper(void (*helper)(Floats&), MathFn raw)
{
    const auto in = mathInputs(1001);
    auto buffer = in;
    helper(buffer);
    check(sameBits(buffer, run(raw, in)));
}
} // namespace

auto tEveryMathBackendMatchesScalar =
    test("ESIMD/everyMathBackendMatchesScalar") = []
{
    for (const auto& fn: mathFunctions())
        for (auto size: kMathSizes)
        {
            const auto in = mathInputs(size);
            const auto oracle = run(fn.scalar, in);
            for (auto backend: fn.backends)
                check(sameBits(run(backend, in), oracle));
        }
};

// The scalar oracle's bits for a fixed input set, recorded on arm64. Every
// backend is held to the oracle on the machine it runs on; this holds the
// oracle itself to the same bits on every architecture and compiler. A
// deliberate change to a math kernel changes these and must update them.
auto tMathOracleMatchesRecordedBits =
    test("ESIMD/mathOracleMatchesRecordedBits") = []
{
    const auto expected = EA::Vector<std::uint64_t> {0xa00bf52f21b4a3f6ull,
                                                     0xc39cbb49352db7c4ull,
                                                     0x94ab50f8a4b5d33aull,
                                                     0x90550e2e6e48f4b8ull,
                                                     0x8a424666c05d52c1ull,
                                                     0xe3a2683965e079acull,
                                                     0x56abf55924d86902ull};
    const auto in = mathInputs(4096);
    const auto functions = mathFunctions();
    for (int i = 0; i < functions.size(); ++i)
        check(hashOf(run(functions[i].scalar, in)) == expected[i]);
};

auto tMathAccuracyWithinDocumentedBound =
    test("ESIMD/mathAccuracyWithinDocumentedBound") = []
{
    for (const auto& fn: mathFunctions())
        check(maxUlpError(fn) <= fn.ulpBound, fn.name);
};

auto tMathSpecialValues = test("ESIMD/mathSpecialValues") = []
{
    const auto functions = mathFunctions();

    checkSpecialCases(mathFunction(functions, "exp"),
                      {{0.f, 1.f},
                       {-0.f, 1.f},
                       {kInf, kInf},
                       {-kInf, 0.f},
                       {kNaN, kNaN},
                       {88.7229f, kInf},
                       {1000.f, kInf},
                       {-88.f, 0.f},
                       {-1000.f, 0.f}});

    // The largest float whose exponential is finite still gives a finite
    // result, just below FLT_MAX.
    for (auto y: run(&esimd::exp, filledFloats(19, 0x1.62e42ep+6f)))
        check(std::isfinite(y) && y > 3.4e38f);

    checkSpecialCases(mathFunction(functions, "exp2"),
                      {{0.f, 1.f},
                       {1.f, 2.f},
                       {10.f, 1024.f},
                       {-1.f, 0.5f},
                       {-126.f, kMinNormal},
                       {-127.f, 0.f},
                       {127.f, 0x1p127f},
                       {128.f, kInf},
                       {kInf, kInf},
                       {-kInf, 0.f},
                       {kNaN, kNaN}});

    checkSpecialCases(mathFunction(functions, "log"),
                      {{1.f, 0.f},
                       {0.f, -kInf},
                       {-0.f, -kInf},
                       {-1.f, kNaN},
                       {-kInf, kNaN},
                       {kInf, kInf},
                       {kNaN, kNaN}});

    checkSpecialCases(mathFunction(functions, "log2"),
                      {{1.f, 0.f},
                       {2.f, 1.f},
                       {1024.f, 10.f},
                       {0.5f, -1.f},
                       {kMinNormal, -126.f},
                       {0x1p-149f, -149.f},
                       {0.f, -kInf},
                       {-2.f, kNaN},
                       {kInf, kInf},
                       {kNaN, kNaN}});

    checkSpecialCases(
        mathFunction(functions, "sin"),
        {{0.f, 0.f}, {-0.f, -0.f}, {kInf, kNaN}, {-kInf, kNaN}, {kNaN, kNaN}});

    checkSpecialCases(
        mathFunction(functions, "cos"),
        {{0.f, 1.f}, {-0.f, 1.f}, {kInf, kNaN}, {-kInf, kNaN}, {kNaN, kNaN}});

    checkSpecialCases(mathFunction(functions, "tanh"),
                      {{0.f, 0.f},
                       {-0.f, -0.f},
                       {kInf, 1.f},
                       {-kInf, -1.f},
                       {20.f, 1.f},
                       {-20.f, -1.f},
                       {1e30f, 1.f},
                       {kNaN, kNaN}});
};

// A NaN goes through untouched, payload and all, on every backend.
auto tMathPassesNaNThrough = test("ESIMD/mathPassesNaNThrough") = []
{
    auto in = Floats(19);
    for (int i = 0; i < in.size(); ++i)
        in[i] = std::bit_cast<float>(0x7FC00000u + static_cast<std::uint32_t>(i));

    for (const auto& fn: mathFunctions())
    {
        check(sameBits(run(fn.scalar, in), in), fn.name);
        for (auto backend: fn.backends)
            check(sameBits(run(backend, in), in), fn.name);
    }
};

// Past the documented full-accuracy range sin and cos degrade, but never
// leave [-1, 1]; past 2^22 they are not reduced at all.
auto tSinCosStayBoundedBeyondTheirRange =
    test("ESIMD/sinCosStayBoundedBeyondTheirRange") = []
{
    auto in = Floats {};
    addRandom(in, 65536.0, 4194304.0, 20'000);
    addRandom(in, -4194304.0, -65536.0, 20'000);

    const auto functions = mathFunctions();
    for (const auto& name: {"sin", "cos"})
    {
        const auto& fn = mathFunction(functions, name);
        for (auto y: run(fn.scalar, in))
            check(std::fabs(y) <= 1.f);
    }

    const auto huge = Floats {0x1p23f, -1e30f, 3e38f};
    for (auto y: run(&esimd::sin, huge))
        check(y == 0.f);
    for (auto y: run(&esimd::cos, huge))
        check(y == 1.f);
};

auto tMathInPlaceMatchesSeparateOutput =
    test("ESIMD/mathInPlaceMatchesSeparateOutput") = []
{
    const auto in = mathInputs(1001);
    for (const auto& fn: mathFunctions())
    {
        auto all = fn.backends;
        all.add(fn.scalar);
        for (auto backend: all)
        {
            auto buffer = in;
            backend(buffer.data(), buffer.data(), buffer.size());
            check(sameBits(buffer, run(backend, in)), fn.name);
        }
    }
};

auto tOpsExpInPlace = test("ESIMD/opsExpInPlace") = []
{ checkInPlaceHelper([](Floats& v) { esimd::exp(v); }, &esimd::exp); };

auto tOpsExp2InPlace = test("ESIMD/opsExp2InPlace") = []
{ checkInPlaceHelper([](Floats& v) { esimd::exp2(v); }, &esimd::exp2); };

auto tOpsLogInPlace = test("ESIMD/opsLogInPlace") = []
{ checkInPlaceHelper([](Floats& v) { esimd::log(v); }, &esimd::log); };

auto tOpsLog2InPlace = test("ESIMD/opsLog2InPlace") = []
{ checkInPlaceHelper([](Floats& v) { esimd::log2(v); }, &esimd::log2); };

auto tOpsSinInPlace = test("ESIMD/opsSinInPlace") = []
{ checkInPlaceHelper([](Floats& v) { esimd::sin(v); }, &esimd::sin); };

auto tOpsCosInPlace = test("ESIMD/opsCosInPlace") = []
{ checkInPlaceHelper([](Floats& v) { esimd::cos(v); }, &esimd::cos); };

auto tOpsTanhInPlace = test("ESIMD/opsTanhInPlace") = []
{ checkInPlaceHelper([](Floats& v) { esimd::tanh(v); }, &esimd::tanh); };

// The helpers take any contiguous float buffer, a Span's subview included,
// and touch only what it covers.
auto tOpsMathHelpersAcceptSpans = test("ESIMD/opsMathHelpersAcceptSpans") = []
{
    auto storage = Floats {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    const auto original = storage;
    auto head = EA::Span<float> {storage}.first(5);
    esimd::exp(head);

    auto expected = original;
    esimd::exp(expected.data(), expected.data(), 5);
    check(sameBits(storage, expected));

    auto fixed = EA::Array<float, 4> {1.f, 2.f, 4.f, 8.f};
    esimd::log2(fixed);
    check(fixed[0] == 0.f && fixed[1] == 1.f && fixed[2] == 2.f && fixed[3] == 3.f);
};
