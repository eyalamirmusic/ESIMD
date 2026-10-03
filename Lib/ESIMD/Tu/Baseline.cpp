#include "../Backends.h"
#include "../Kernels/FindByte.h"
#include "../Kernels/ResizeBilinear.h"
#include "../Kernels/SwapRedBlue.h"
#include "../Kernels/WarpAffine.h"

// The per-architecture baseline backend, reached without a special compile
// flag: SSE2 on x86-64 (guaranteed by the ABI), NEON on AArch64 (mandatory).
// Arch-guarded so a wrong-architecture slice of a multi-arch build is a no-op.

#if defined(__x86_64__) || defined(_M_X64)

#include "../Backend/Sse2.h"

namespace esimd::backends
{

void swapRedBlue_sse2(const std::uint8_t* in, std::uint8_t* out, int pixelCount)
{
    kernels::swapRedBlueImpl<backend::Sse2>(in, out, pixelCount);
}

void resizeBilinear_sse2(const std::uint8_t* src,
                         int srcWidth,
                         int srcHeight,
                         std::uint8_t* dst,
                         int dstWidth,
                         int dstHeight)
{
    kernels::resizeBilinearImpl<backend::Sse2>(
        src, srcWidth, srcHeight, dst, dstWidth, dstHeight);
}

void warpAffineInverse_sse2(const std::uint8_t* src,
                            int srcWidth,
                            int srcHeight,
                            const float* inverse2x3,
                            std::uint8_t* dst,
                            int dstWidth,
                            int dstHeight)
{
    kernels::warpAffineInverseImpl<backend::Sse2>(
        src, srcWidth, srcHeight, inverse2x3, dst, dstWidth, dstHeight);
}

int findFirst_sse2(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::findFirstImpl<backend::Sse2, false>(data, count, cls);
}

int findFirstNot_sse2(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::findFirstImpl<backend::Sse2, true>(data, count, cls);
}

int countOf_sse2(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::countOfImpl<backend::Sse2>(data, count, cls);
}

} // namespace esimd::backends

#elif defined(__aarch64__) || defined(_M_ARM64)

#include "../Backend/Neon.h"

namespace esimd::backends
{

void swapRedBlue_neon(const std::uint8_t* in, std::uint8_t* out, int pixelCount)
{
    kernels::swapRedBlueImpl<backend::Neon>(in, out, pixelCount);
}

void resizeBilinear_neon(const std::uint8_t* src,
                         int srcWidth,
                         int srcHeight,
                         std::uint8_t* dst,
                         int dstWidth,
                         int dstHeight)
{
    kernels::resizeBilinearImpl<backend::Neon>(
        src, srcWidth, srcHeight, dst, dstWidth, dstHeight);
}

void warpAffineInverse_neon(const std::uint8_t* src,
                            int srcWidth,
                            int srcHeight,
                            const float* inverse2x3,
                            std::uint8_t* dst,
                            int dstWidth,
                            int dstHeight)
{
    kernels::warpAffineInverseImpl<backend::Neon>(
        src, srcWidth, srcHeight, inverse2x3, dst, dstWidth, dstHeight);
}

int findFirst_neon(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::findFirstImpl<backend::Neon, false>(data, count, cls);
}

int findFirstNot_neon(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::findFirstImpl<backend::Neon, true>(data, count, cls);
}

int countOf_neon(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::countOfImpl<backend::Neon>(data, count, cls);
}

} // namespace esimd::backends

#endif
