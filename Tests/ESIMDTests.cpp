#include <ESIMD/Backends.h>
#include <ESIMD/Ops.h>

#include <NanoTest/NanoTest.h>
#include <ea_data_structures/Structures/Array.h>
#include <ea_data_structures/Structures/Span.h>
#include <ea_data_structures/ea_data_structures.h>

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
