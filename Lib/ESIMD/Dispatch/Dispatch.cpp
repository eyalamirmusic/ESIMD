#include "../Backends.h"
#include "../ESIMD.h"

// Public entry points of the dispatched kernels. Each resolves the best available backend
// once (CPUID on x86-64, fixed on every other architecture) and calls it through
// a function pointer -- a deliberate, non-inlinable boundary between baseline and
// AVX2 code. (The float-array primitives live in Tu/ArrayOps.cpp; being
// memory-bound and auto-vectorized, they need no runtime dispatch.)
namespace esimd
{
namespace
{

using SwapFn = void (*)(const std::uint8_t*, std::uint8_t*, int);

SwapFn pickSwapRedBlue() noexcept
{
#if defined(__x86_64__) || defined(_M_X64)
#if defined(ESIMD_HAS_AVX2)
    if (cpu::hasAvx2Fma())
        return &backends::swapRedBlue_avx2;
#endif
    return &backends::swapRedBlue_sse2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &backends::swapRedBlue_neon;
#else
    return &backends::swapRedBlue_scalar;
#endif
}

using ResizeFn = void (*)(const std::uint8_t*, int, int, std::uint8_t*, int, int);

ResizeFn pickResizeBilinear() noexcept
{
    // Bilinear blends one pixel's four channels per 128-bit step, so AVX2's
    // 256-bit width adds nothing yet; x86 uses SSE2 until a 2-pixel AVX2 path
    // lands.
#if defined(__x86_64__) || defined(_M_X64)
    return &backends::resizeBilinear_sse2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &backends::resizeBilinear_neon;
#else
    return &backends::resizeBilinear_scalar;
#endif
}

using WarpFn =
    void (*)(const std::uint8_t*, int, int, const float*, std::uint8_t*, int, int);

WarpFn pickWarpAffineInverse() noexcept
{
    // Same 128-bit per-pixel blend as resizeBilinear, so AVX2 adds nothing yet.
#if defined(__x86_64__) || defined(_M_X64)
    return &backends::warpAffineInverse_sse2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &backends::warpAffineInverse_neon;
#else
    return &backends::warpAffineInverse_scalar;
#endif
}

using ScanFn = int (*)(const std::uint8_t*, int, const ByteClass&);

// The three scans share one selection: a byte-at-a-time classification is
// compute-bound, so the 32-lane AVX2 path is worth its dispatch.
struct ScanBackend
{
    ScanFn findFirst;
    ScanFn findFirstNot;
    ScanFn countOf;
};

ScanBackend pickScanBackend() noexcept
{
#if defined(__x86_64__) || defined(_M_X64)
#if defined(ESIMD_HAS_AVX2)
    if (cpu::hasAvx2Fma())
        return {&backends::findFirst_avx2,
                &backends::findFirstNot_avx2,
                &backends::countOf_avx2};
#endif
    return {&backends::findFirst_sse2,
            &backends::findFirstNot_sse2,
            &backends::countOf_sse2};
#elif defined(__aarch64__) || defined(_M_ARM64)
    return {&backends::findFirst_neon,
            &backends::findFirstNot_neon,
            &backends::countOf_neon};
#else
    return {&backends::findFirst_scalar,
            &backends::findFirstNot_scalar,
            &backends::countOf_scalar};
#endif
}

const ScanBackend& scanBackend() noexcept
{
    static const ScanBackend backend = pickScanBackend();
    return backend;
}

using MathFn = void (*)(const float*, float*, int);

// The math functions share one selection too: a polynomial per element is
// compute-bound, so the 8-lane AVX2 path is worth its dispatch (unlike the
// memory-bound float-array primitives).
struct MathBackend
{
    MathFn exp;
    MathFn exp2;
    MathFn log;
    MathFn log2;
    MathFn sin;
    MathFn cos;
    MathFn tanh;
};

MathBackend pickMathBackend() noexcept
{
#if defined(__x86_64__) || defined(_M_X64)
#if defined(ESIMD_HAS_AVX2)
    if (cpu::hasAvx2Fma())
        return {&backends::exp_avx2,
                &backends::exp2_avx2,
                &backends::log_avx2,
                &backends::log2_avx2,
                &backends::sin_avx2,
                &backends::cos_avx2,
                &backends::tanh_avx2};
#endif
    return {&backends::exp_sse2,
            &backends::exp2_sse2,
            &backends::log_sse2,
            &backends::log2_sse2,
            &backends::sin_sse2,
            &backends::cos_sse2,
            &backends::tanh_sse2};
#elif defined(__aarch64__) || defined(_M_ARM64)
    return {&backends::exp_neon,
            &backends::exp2_neon,
            &backends::log_neon,
            &backends::log2_neon,
            &backends::sin_neon,
            &backends::cos_neon,
            &backends::tanh_neon};
#else
    return {&backends::exp_scalar,
            &backends::exp2_scalar,
            &backends::log_scalar,
            &backends::log2_scalar,
            &backends::sin_scalar,
            &backends::cos_scalar,
            &backends::tanh_scalar};
#endif
}

const MathBackend& mathBackend() noexcept
{
    static const MathBackend backend = pickMathBackend();
    return backend;
}

} // namespace

void swapRedBlue(const std::uint8_t* in, std::uint8_t* out, int pixelCount)
{
    static const SwapFn fn = pickSwapRedBlue();
    fn(in, out, pixelCount);
}

void convertBgraToRgba(const std::uint8_t* src,
                       int srcBytesPerRow,
                       std::uint8_t* dst,
                       int width,
                       int height)
{
    if (src == nullptr || dst == nullptr || width <= 0 || height <= 0)
        return;

    const auto tightRowBytes = width * 4;

    // Tightly-packed frames swap in a single pass; padded rows go one by one.
    // Either way the per-pixel work runs through the dispatched swapRedBlue, and
    // this whole routine is compiled at the SIMD module's forced optimization.
    if (srcBytesPerRow == tightRowBytes)
    {
        swapRedBlue(src, dst, width * height);
    }
    else
    {
        for (auto y = 0; y < height; ++y)
            swapRedBlue(src + static_cast<std::ptrdiff_t>(y) * srcBytesPerRow,
                        dst + static_cast<std::ptrdiff_t>(y) * tightRowBytes,
                        width);
    }
}

void resizeBilinear(const std::uint8_t* src,
                    int srcWidth,
                    int srcHeight,
                    std::uint8_t* dst,
                    int dstWidth,
                    int dstHeight)
{
    static const ResizeFn fn = pickResizeBilinear();
    fn(src, srcWidth, srcHeight, dst, dstWidth, dstHeight);
}

void warpAffineInverse(const std::uint8_t* src,
                       int srcWidth,
                       int srcHeight,
                       const float* inverse2x3,
                       std::uint8_t* dst,
                       int dstWidth,
                       int dstHeight)
{
    static const WarpFn fn = pickWarpAffineInverse();
    fn(src, srcWidth, srcHeight, inverse2x3, dst, dstWidth, dstHeight);
}

int findFirst(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return scanBackend().findFirst(data, count, cls);
}

int findFirstNot(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return scanBackend().findFirstNot(data, count, cls);
}

int countOf(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return scanBackend().countOf(data, count, cls);
}

void exp(const float* a, float* out, int count)
{
    mathBackend().exp(a, out, count);
}

void exp2(const float* a, float* out, int count)
{
    mathBackend().exp2(a, out, count);
}

void log(const float* a, float* out, int count)
{
    mathBackend().log(a, out, count);
}

void log2(const float* a, float* out, int count)
{
    mathBackend().log2(a, out, count);
}

void sin(const float* a, float* out, int count)
{
    mathBackend().sin(a, out, count);
}

void cos(const float* a, float* out, int count)
{
    mathBackend().cos(a, out, count);
}

void tanh(const float* a, float* out, int count)
{
    mathBackend().tanh(a, out, count);
}

} // namespace esimd
