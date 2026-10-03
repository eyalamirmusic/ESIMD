#include "../Backend/Scalar.h"
#include "../Backends.h"
#include "../Kernels/FindByte.h"
#include "../Kernels/ResizeBilinear.h"
#include "../Kernels/SwapRedBlue.h"
#include "../Kernels/WarpAffine.h"

// Scalar backend entry points. Always compiled, on every architecture: it is
// the correctness oracle and the SIMD kernels' tail fallback.
namespace esimd::backends
{

void swapRedBlue_scalar(const std::uint8_t* in, std::uint8_t* out, int pixelCount)
{
    kernels::swapRedBlueImpl<backend::Scalar>(in, out, pixelCount);
}

void resizeBilinear_scalar(const std::uint8_t* src,
                           int srcWidth,
                           int srcHeight,
                           std::uint8_t* dst,
                           int dstWidth,
                           int dstHeight)
{
    kernels::resizeBilinearImpl<backend::Scalar>(
        src, srcWidth, srcHeight, dst, dstWidth, dstHeight);
}

void warpAffineInverse_scalar(const std::uint8_t* src,
                              int srcWidth,
                              int srcHeight,
                              const float* inverse2x3,
                              std::uint8_t* dst,
                              int dstWidth,
                              int dstHeight)
{
    kernels::warpAffineInverseImpl<backend::Scalar>(
        src, srcWidth, srcHeight, inverse2x3, dst, dstWidth, dstHeight);
}

int findFirst_scalar(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::findFirstImpl<backend::Scalar, false>(data, count, cls);
}

int findFirstNot_scalar(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::findFirstImpl<backend::Scalar, true>(data, count, cls);
}

int countOf_scalar(const std::uint8_t* data, int count, const ByteClass& cls)
{
    return kernels::countOfImpl<backend::Scalar>(data, count, cls);
}

} // namespace esimd::backends
