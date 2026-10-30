#include <ESIMD/Backends.h>
#include <ESIMD/ESIMD.h>

#include <ea_data_structures/ea_data_structures.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>

// Micro-benchmark comparing the scalar reference against the active SIMD backend
// for the ESIMD kernels, plus the float-array primitives (auto-vectorized vs
// a non-vectorized scalar reference). ESIMD and this harness are always built
// -O3, so any build config is fine.

// Disable auto-vectorization for the array primitives' scalar reference, so the
// comparison shows the vectorization win rather than vectorized-vs-vectorized.
#if defined(__clang__)
#define ESIMD_NO_VECTORIZE                                                          \
    _Pragma("clang loop vectorize(disable) interleave(disable)")
#elif defined(_MSC_VER)
#define ESIMD_NO_VECTORIZE __pragma(loop(no_vector))
#else
#define ESIMD_NO_VECTORIZE
#endif

namespace
{
using Pixels = EA::Vector<std::uint8_t>;
using Floats = EA::Vector<float>;
using ResizeFn = void (*)(const std::uint8_t*, int, int, std::uint8_t*, int, int);
using SwapFn = void (*)(const std::uint8_t*, std::uint8_t*, int);
using WarpFn =
    void (*)(const std::uint8_t*, int, int, const float*, std::uint8_t*, int, int);

// A volatile sink that consumes one output byte per iteration so the optimizer
// cannot elide the calls. Combined with perturbing an input byte per iteration
// (below) it also prevents hoisting the loop-invariant call out of the loop.
volatile std::uint64_t gSink = 0;

Pixels makeBuffer(int byteCount)
{
    auto data = Pixels(byteCount);
    for (int i = 0; i < data.size(); ++i)
        data[i] = static_cast<std::uint8_t>((i * 53 + (i / 4) * 7 + 19) & 0xFF);
    return data;
}

Floats makeFloats(int count, int stride, int offset)
{
    auto data = Floats(count);
    for (int i = 0; i < count; ++i)
        data[i] = static_cast<float>((i % stride) + offset);
    return data;
}

std::uint8_t* bytes(Floats& v)
{
    return reinterpret_cast<std::uint8_t*>(v.data());
}

// Returns milliseconds per iteration. One input byte is perturbed each iteration
// so the call is not loop-invariant; one output byte is observed so the stores
// are not dead. Both are O(1) and negligible against a whole-buffer op.
template <class Run>
double timeMs(int iters,
              std::uint8_t* perturb,
              int perturbBytes,
              const std::uint8_t* observe,
              Run&& run)
{
    run(); // warm up caches and the dispatch pointer

    std::uint64_t sink = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        perturb[i % perturbBytes] ^= static_cast<std::uint8_t>(i + 1);
        run();
        sink += observe[0];
    }
    const auto end = std::chrono::steady_clock::now();

    gSink ^= sink;
    return std::chrono::duration<double, std::milli>(end - start).count()
           / static_cast<double>(iters);
}

double perSec(long long items, double ms)
{
    return static_cast<double>(items) / (ms * 1000.0);
}

void report(
    const char* name, long long items, double ms, double scalarMs, const char* unit)
{
    std::printf(" | %-5s %8.3f ms %7.1f %s %5.2fx",
                name,
                ms,
                perSec(items, ms),
                unit,
                scalarMs / ms);
}

// Non-vectorized scalar references for the array primitives.
void scalarAdd(const float* a, const float* b, float* out, int n)
{
    ESIMD_NO_VECTORIZE
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

void scalarSubtract(const float* a, const float* b, float* out, int n)
{
    ESIMD_NO_VECTORIZE
    for (int i = 0; i < n; ++i)
        out[i] = a[i] - b[i];
}

void scalarMultiply(const float* a, const float* b, float* out, int n)
{
    ESIMD_NO_VECTORIZE
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
}

void scalarMultiplyByScalar(const float* a, float s, float* out, int n)
{
    ESIMD_NO_VECTORIZE
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * s;
}

void scalarMultiplyAdd(
    const float* a, const float* b, const float* c, float* out, int n)
{
    ESIMD_NO_VECTORIZE
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * b[i] + c[i];
}

const char* baselineName()
{
#if defined(__x86_64__) || defined(_M_X64)
    return "sse2";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "neon";
#else
    return "scalar";
#endif
}

ResizeFn baselineResize()
{
#if defined(__x86_64__) || defined(_M_X64)
    return &esimd::backends::resizeBilinear_sse2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &esimd::backends::resizeBilinear_neon;
#else
    return &esimd::backends::resizeBilinear_scalar;
#endif
}

SwapFn baselineSwap()
{
#if defined(__x86_64__) || defined(_M_X64)
    return &esimd::backends::swapRedBlue_sse2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &esimd::backends::swapRedBlue_neon;
#else
    return &esimd::backends::swapRedBlue_scalar;
#endif
}

WarpFn baselineWarp()
{
#if defined(__x86_64__) || defined(_M_X64)
    return &esimd::backends::warpAffineInverse_sse2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &esimd::backends::warpAffineInverse_neon;
#else
    return &esimd::backends::warpAffineInverse_scalar;
#endif
}

using ScanFn = int (*)(const std::uint8_t*, int, const esimd::ByteClass&);

ScanFn baselineFindFirst()
{
#if defined(__x86_64__) || defined(_M_X64)
    return &esimd::backends::findFirst_sse2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &esimd::backends::findFirst_neon;
#else
    return &esimd::backends::findFirst_scalar;
#endif
}

struct ResizeConfig
{
    int srcW, srcH, dstW, dstH;
};

void benchResize(const ResizeConfig& c)
{
    auto src = makeBuffer(c.srcW * c.srcH * 4);
    auto dst = Pixels(c.dstW * c.dstH * 4);
    const long long outPixels = static_cast<long long>(c.dstW) * c.dstH;
    const int iters =
        static_cast<int>(std::max<long long>(20, 80'000'000LL / outPixels));

    const auto scalar = &esimd::backends::resizeBilinear_scalar;
    const auto simd = baselineResize();

    const auto scalarMs = timeMs(
        iters,
        src.data(),
        src.size(),
        dst.data(),
        [&] { scalar(src.data(), c.srcW, c.srcH, dst.data(), c.dstW, c.dstH); });
    const auto simdMs = timeMs(
        iters,
        src.data(),
        src.size(),
        dst.data(),
        [&] { simd(src.data(), c.srcW, c.srcH, dst.data(), c.dstW, c.dstH); });

    std::printf("  %4dx%-4d -> %4dx%-4d | scalar %8.3f ms %7.1f Mpix/s",
                c.srcW,
                c.srcH,
                c.dstW,
                c.dstH,
                scalarMs,
                perSec(outPixels, scalarMs));
    report(baselineName(), outPixels, simdMs, scalarMs, "Mpix/s");
    std::printf("\n");
}

void benchWarp(const ResizeConfig& c)
{
    auto src = makeBuffer(c.srcW * c.srcH * 4);
    auto dst = Pixels(c.dstW * c.dstH * 4);
    // A rotate + scale + translate inverse matrix (the values don't affect timing).
    const float m[6] = {0.8f, -0.2f, 8.f, 0.2f, 0.8f, 5.f};
    const long long outPixels = static_cast<long long>(c.dstW) * c.dstH;
    const int iters =
        static_cast<int>(std::max<long long>(20, 80'000'000LL / outPixels));

    const auto scalar = &esimd::backends::warpAffineInverse_scalar;
    const auto simd = baselineWarp();

    const auto scalarMs = timeMs(
        iters,
        src.data(),
        src.size(),
        dst.data(),
        [&] { scalar(src.data(), c.srcW, c.srcH, m, dst.data(), c.dstW, c.dstH); });
    const auto simdMs = timeMs(
        iters,
        src.data(),
        src.size(),
        dst.data(),
        [&] { simd(src.data(), c.srcW, c.srcH, m, dst.data(), c.dstW, c.dstH); });

    std::printf("  %4dx%-4d -> %4dx%-4d | scalar %8.3f ms %7.1f Mpix/s",
                c.srcW,
                c.srcH,
                c.dstW,
                c.dstH,
                scalarMs,
                perSec(outPixels, scalarMs));
    report(baselineName(), outPixels, simdMs, scalarMs, "Mpix/s");
    std::printf("\n");
}

void benchSwap(long long pixels)
{
    auto src = makeBuffer(static_cast<int>(pixels) * 4);
    auto dst = Pixels(static_cast<int>(pixels) * 4);
    const auto count = static_cast<int>(pixels);
    const int iters =
        static_cast<int>(std::max<long long>(20, 400'000'000LL / pixels));

    const auto scalarMs = timeMs(
        iters,
        src.data(),
        src.size(),
        dst.data(),
        [&] { esimd::backends::swapRedBlue_scalar(src.data(), dst.data(), count); });
    const auto simd = baselineSwap();
    const auto simdMs = timeMs(iters,
                               src.data(),
                               src.size(),
                               dst.data(),
                               [&] { simd(src.data(), dst.data(), count); });

    std::printf("  %5.1f Mpix | scalar %8.3f ms %7.1f Mpix/s",
                static_cast<double>(pixels) / 1'000'000.0,
                scalarMs,
                perSec(pixels, scalarMs));
    report(baselineName(), pixels, simdMs, scalarMs, "Mpix/s");

#if defined(ESIMD_HAS_AVX2)
    if (esimd::cpu::hasAvx2Fma())
    {
        const auto avx2Ms = timeMs(
            iters,
            src.data(),
            src.size(),
            dst.data(),
            [&]
            { esimd::backends::swapRedBlue_avx2(src.data(), dst.data(), count); });
        report("avx2", pixels, avx2Ms, scalarMs, "Mpix/s");
    }
#endif
    std::printf("\n");
}

void benchArrayOps(int count)
{
    auto a = makeFloats(count, 17, 1);
    auto b = makeFloats(count, 23, 1);
    auto c = makeFloats(count, 5, 1);
    auto out = Floats(count);
    const int iters = std::max(20, 400'000'000 / count);

    std::printf("  %d elements:\n", count);

    const auto row = [&](const char* name, auto scalarRun, auto simdRun)
    {
        const auto scalarMs =
            timeMs(iters, bytes(a), a.size() * 4, bytes(out), scalarRun);
        const auto simdMs =
            timeMs(iters, bytes(a), a.size() * 4, bytes(out), simdRun);
        std::printf("    %-16s | scalar %8.3f ms %7.1f Melem/s",
                    name,
                    scalarMs,
                    perSec(count, scalarMs));
        report(baselineName(), count, simdMs, scalarMs, "Melem/s");
        std::printf("\n");
    };

    row(
        "add",
        [&] { scalarAdd(a.data(), b.data(), out.data(), count); },
        [&] { esimd::add(a.data(), b.data(), out.data(), count); });
    row(
        "subtract",
        [&] { scalarSubtract(a.data(), b.data(), out.data(), count); },
        [&] { esimd::subtract(a.data(), b.data(), out.data(), count); });
    row(
        "multiply",
        [&] { scalarMultiply(a.data(), b.data(), out.data(), count); },
        [&] { esimd::multiply(a.data(), b.data(), out.data(), count); });
    row(
        "multiplyByScalar",
        [&] { scalarMultiplyByScalar(a.data(), 3.f, out.data(), count); },
        [&] { esimd::multiplyByScalar(a.data(), 3.f, out.data(), count); });
    row(
        "multiplyAdd",
        [&] { scalarMultiplyAdd(a.data(), b.data(), c.data(), out.data(), count); },
        [&]
        { esimd::multiplyAdd(a.data(), b.data(), c.data(), out.data(), count); });
}

// Non-vectorized std:: references for the math functions.
template <float (*Fn)(float)>
void scalarMath(const float* a, float* out, int n)
{
    ESIMD_NO_VECTORIZE
    for (int i = 0; i < n; ++i)
        out[i] = Fn(a[i]);
}

float stdExp(float x)
{
    return std::exp(x);
}
float stdExp2(float x)
{
    return std::exp2(x);
}
float stdLog(float x)
{
    return std::log(x);
}
float stdLog2(float x)
{
    return std::log2(x);
}
float stdSin(float x)
{
    return std::sin(x);
}
float stdCos(float x)
{
    return std::cos(x);
}
float stdTanh(float x)
{
    return std::tanh(x);
}

using MathFn = void (*)(const float*, float*, int);

struct MathRow
{
    const char* name;
    MathFn reference;
    MathFn scalar;
    MathFn baseline;
    MathFn avx2;
};

EA::Vector<MathRow> mathRows()
{
    using namespace esimd::backends;
    const char* names[] = {"exp", "exp2", "log", "log2", "sin", "cos", "tanh"};
    const MathFn references[] = {&scalarMath<stdExp>,
                                 &scalarMath<stdExp2>,
                                 &scalarMath<stdLog>,
                                 &scalarMath<stdLog2>,
                                 &scalarMath<stdSin>,
                                 &scalarMath<stdCos>,
                                 &scalarMath<stdTanh>};
    const MathFn scalars[] = {&exp_scalar,
                              &exp2_scalar,
                              &log_scalar,
                              &log2_scalar,
                              &sin_scalar,
                              &cos_scalar,
                              &tanh_scalar};
#if defined(__x86_64__) || defined(_M_X64)
    const MathFn baselines[] = {&exp_sse2,
                                &exp2_sse2,
                                &log_sse2,
                                &log2_sse2,
                                &sin_sse2,
                                &cos_sse2,
                                &tanh_sse2};
#elif defined(__aarch64__) || defined(_M_ARM64)
    const MathFn baselines[] = {&exp_neon,
                                &exp2_neon,
                                &log_neon,
                                &log2_neon,
                                &sin_neon,
                                &cos_neon,
                                &tanh_neon};
#else
    const auto& baselines = scalars;
#endif
    MathFn avx2[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
#if defined(ESIMD_HAS_AVX2)
    if (esimd::cpu::hasAvx2Fma())
    {
        const MathFn available[] = {&exp_avx2,
                                    &exp2_avx2,
                                    &log_avx2,
                                    &log2_avx2,
                                    &sin_avx2,
                                    &cos_avx2,
                                    &tanh_avx2};
        std::copy(std::begin(available), std::end(available), avx2);
    }
#endif

    auto rows = EA::Vector<MathRow> {};
    for (int i = 0; i < 7; ++i)
        rows.add({names[i], references[i], scalars[i], baselines[i], avx2[i]});
    return rows;
}

// The std:: function (scalar, not vectorized) against ESIMD's scalar oracle,
// its baseline SIMD backend and, where the CPU has it, AVX2. The inputs lie
// in (0, 8), inside every function's domain.
void benchMath(int count)
{
    auto a = Floats(count);
    for (int i = 0; i < count; ++i)
        a[i] = static_cast<float>((i * 37) % 1000 + 1) * 0.008f;
    auto out = Floats(count);
    const int iters = std::max(20, 40'000'000 / count);

    for (const auto& row: mathRows())
    {
        const auto time = [&](MathFn fn)
        {
            return timeMs(iters,
                          bytes(a),
                          a.size() * 4,
                          bytes(out),
                          [&] { fn(a.data(), out.data(), count); });
        };

        const auto referenceMs = time(row.reference);
        std::printf("    %-5s | std    %8.3f ms %7.1f Melem/s",
                    row.name,
                    referenceMs,
                    perSec(count, referenceMs));
        report("esimd", count, time(row.scalar), referenceMs, "Melem/s");
        report(baselineName(), count, time(row.baseline), referenceMs, "Melem/s");
        if (row.avx2 != nullptr)
            report("avx2", count, time(row.avx2), referenceMs, "Melem/s");
        std::printf("\n");
    }
}

// A text scan: a buffer of plain string bytes with the one special byte at the
// very end, so every backend walks the whole thing. The input is left alone
// (perturbing it would plant an early hit); the result is what the sink
// observes instead.
void benchFindFirst(int count)
{
    static constexpr auto stringSpecial =
        esimd::anyOf('"', '\\') | esimd::below(0x20);

    auto text = Pixels(count);
    for (int i = 0; i < count; ++i)
        text[i] = 'a';
    text[count - 1] = '"';
    std::uint8_t scratch[1] = {0};
    std::uint8_t result[4] = {0, 0, 0, 0};
    const int iters = std::max(20, 400'000'000 / count);

    const auto run = [&](auto fn)
    {
        return [&, fn]
        {
            const auto index = fn(text.data(), count, stringSpecial);
            std::memcpy(result, &index, sizeof(int));
        };
    };

    const auto scalarMs =
        timeMs(iters, scratch, 1, result, run(&esimd::backends::findFirst_scalar));
    const auto simdMs = timeMs(iters, scratch, 1, result, run(baselineFindFirst()));

    std::printf("  %9d bytes | scalar %8.3f ms %7.1f MB/s",
                count,
                scalarMs,
                perSec(count, scalarMs));
    report(baselineName(), count, simdMs, scalarMs, "MB/s");

#if defined(ESIMD_HAS_AVX2)
    if (esimd::cpu::hasAvx2Fma())
    {
        const auto avx2Ms =
            timeMs(iters, scratch, 1, result, run(&esimd::backends::findFirst_avx2));
        report("avx2", count, avx2Ms, scalarMs, "MB/s");
    }
#endif
    std::printf("\n");
}
} // namespace

int main()
{
#if defined(_MSC_VER) && !defined(NDEBUG)
    std::printf("Note: MSVC Debug -- the kernels are /O2 but this benchmark "
                "harness is /Od; build Release for the cleanest numbers.\n\n");
#endif

    std::printf("ESIMD benchmark | baseline SIMD = %s", baselineName());
#if defined(ESIMD_HAS_AVX2)
    std::printf(", avx2 %s", esimd::cpu::hasAvx2Fma() ? "available" : "absent");
#endif
    std::printf("\n\nresizeBilinear:\n");

    const ResizeConfig resizes[] = {
        {1920, 1080, 640, 360}, // 3x downscale
        {1280, 720, 1280, 720}, // identity
        {640, 360, 1920, 1080}, // 3x upscale
        {1280, 720, 213, 120}, // thumbnail
        {512, 512, 1024, 1024}, // 2x upscale
    };
    for (const auto& cfg: resizes)
        benchResize(cfg);

    std::printf("\nwarpAffineInverse:\n");
    for (const auto& cfg: resizes)
        benchWarp(cfg);

    std::printf("\nswapRedBlue:\n");
    for (long long pixels: {1'000'000LL, 8'000'000LL})
        benchSwap(pixels);

    std::printf("\nfindFirst (quote, backslash or control char at the end):\n");
    for (int count: {64, 4096, 1'000'000})
        benchFindFirst(count);

    std::printf("\narray primitives (non-vectorized scalar vs %s):\n",
                baselineName());
    benchArrayOps(4096); // cache-resident: shows the vectorization win
    benchArrayOps(4'000'000); // larger than cache: memory-bandwidth-bound

    std::printf("\nmath functions on 4096 elements (std:: scalar vs ESIMD):\n");
    benchMath(4096);

    std::printf("\nchecksum %llu\n", static_cast<unsigned long long>(gSink));
    return 0;
}
