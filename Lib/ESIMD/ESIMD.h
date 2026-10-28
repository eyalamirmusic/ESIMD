#pragma once

#include "ByteClass.h"

#include <cstdint>

// ESIMD: a small, self-contained portable SIMD layer.
//
// This is the public, portable surface. It carries NO architecture or feature
// conditionals -- every one of those lives in the implementation (and in the
// internal <ESIMD/Backends.h>). Each call selects the fastest backend
// available on the host at runtime, once, behind a function pointer.
namespace esimd
{

// --- Image kernels ---

// Swap the red and blue channels of `pixelCount` tightly-packed 8-bit RGBA
// pixels (RGBA <-> BGRA), writing to `out`. `in` and `out` may be equal
// (in place) but must not otherwise overlap.
void swapRedBlue(const std::uint8_t* in, std::uint8_t* out, int pixelCount);

// Convert a BGRA8 camera frame to tightly-packed RGBA8 in one pass: swap
// red/blue and drop any trailing row padding. `src` rows are `srcBytesPerRow`
// apart (>= width*4); `dst` holds width*height*4 bytes with no padding. This
// lives in ESIMD so its per-byte work is always built at the platform
// optimization maximum, independent of the (possibly unoptimized) caller.
void convertBgraToRgba(const std::uint8_t* src,
                       int srcBytesPerRow,
                       std::uint8_t* dst,
                       int width,
                       int height);

// Resize a tightly-packed RGBA8 image with bilinear sampling, half-pixel-center
// mapping and edge clamping (OpenCV semantics). `dst` holds dstW*dstH*4 bytes.
// Caller guarantees src/dst dimensions are positive and buffers correctly sized.
void resizeBilinear(const std::uint8_t* src,
                    int srcWidth,
                    int srcHeight,
                    std::uint8_t* dst,
                    int dstWidth,
                    int dstHeight);

// Inverse affine warp of a tightly-packed RGBA8 image with bilinear sampling and
// edge clamping. `inverse2x3` points to the six row-major coefficients of the
// 2x3 inverse matrix ([m0 m1 m2; m3 m4 m5]): each destination pixel (dx, dy)
// samples the source at (m0*dx + m1*dy + m2, m3*dx + m4*dy + m5). `dst` holds
// dstW*dstH*4 bytes.
void warpAffineInverse(const std::uint8_t* src,
                       int srcWidth,
                       int srcHeight,
                       const float* inverse2x3,
                       std::uint8_t* dst,
                       int dstWidth,
                       int dstHeight);

// Crop a width x height region at (x, y) of a tightly-packed RGBA8 image and
// mirror it horizontally in one pass. `dst` holds width*height*4 bytes. The
// caller guarantees the crop region lies within the source.
void mirroredCrop(const std::uint8_t* src,
                  int srcWidth,
                  int x,
                  int y,
                  int width,
                  int height,
                  std::uint8_t* dst);

// --- Elementwise float-array primitives ---
//
// Each processes `count` floats. `out` may alias an input (safe, elementwise).
// SIMD-accelerated and deterministic: being elementwise, the result is identical
// at any vector width and on every build.

// out[i] = a[i] + b[i]
void add(const float* a, const float* b, float* out, int count);

// out[i] = a[i] - b[i]
void subtract(const float* a, const float* b, float* out, int count);

// out[i] = a[i] * b[i]
void multiply(const float* a, const float* b, float* out, int count);

// out[i] = a[i] * scalar
void multiplyByScalar(const float* a, float scalar, float* out, int count);

// out[i] = a[i] * b[i] + c[i]
void multiplyAdd(
    const float* a, const float* b, const float* c, float* out, int count);

// out[i] = a[i] * b + c[i] -- the axpy shape; with out == c it accumulates a
// scaled array in place (out[i] += a[i] * b).
void multiplyAdd(const float* a, float b, const float* c, float* out, int count);

// out[i] = a[i] + t * (b[i] - a[i])
void lerp(const float* a, const float* b, float t, float* out, int count);

// --- Elementwise math functions ---
//
// Each processes `count` floats; `out` may alias `a`. SIMD-accelerated (AVX2,
// SSE2, NEON) and deterministic: every backend, vector width and build gives
// bit-identical results, on every architecture. They are built only from
// exactly rounded operations -- no FMA, no hardware estimates -- so they are
// not correctly rounded, but the error against the exact result, measured over
// every float input, is at most:
//
//   exp 1.0 ULP, exp2 1.25, log 1.0, log2 1.5, sin 1.3, cos 1.3, tanh 1.35
//
// sin and cos hold that bound for |x| <= 65536 (2^16). Beyond it the error
// grows with |x| (about |x| * 2^-24 absolute) but results stay within [-1, 1];
// beyond 2^22, where adjacent floats are half a radian apart, the argument is
// not reduced at all and sin returns 0, cos returns 1.
//
// Special values: a NaN input is returned unchanged. exp(+inf) = +inf,
// exp(-inf) = 0, and exp/exp2 overflow to +inf; results that would be below
// FLT_MIN (x < ~-87.34 for exp, x < -126 for exp2) are flushed to +0 instead
// of returned as subnormals. exp2 of an integer, and log2 of a power of two,
// are exact. log(1) = 0, log(+-0) = -inf, log(+inf) = +inf, log of a negative
// number is NaN; subnormal inputs are handled. sin(+-0) = +-0, cos(0) = 1,
// sin and cos of +-inf are NaN. tanh(+-0) = +-0 and tanh saturates to exactly
// +-1 (at |x| >= ~9, so tanh(+-inf) = +-1). Results assume the default
// floating-point environment: a caller that turns on flush-to-zero sees
// subnormal inputs treated as zero.

// out[i] = e^a[i]
void exp(const float* a, float* out, int count);

// out[i] = 2^a[i]
void exp2(const float* a, float* out, int count);

// out[i] = ln(a[i])
void log(const float* a, float* out, int count);

// out[i] = log2(a[i])
void log2(const float* a, float* out, int count);

// out[i] = sin(a[i])
void sin(const float* a, float* out, int count);

// out[i] = cos(a[i])
void cos(const float* a, float* out, int count);

// out[i] = tanh(a[i])
void tanh(const float* a, float* out, int count);

// --- Float-array reductions ---
//
// Unlike the elementwise primitives, a reduction has an accumulation order.
// These use a fixed four-lane interleave (explicitly reassociated so the
// compiler can vectorize them without fast-math), so the result is still
// deterministic — identical on every build and architecture — but it is NOT
// bit-equal to a naive sequential loop over the same data.

// Returns sum(a[i]^2), accumulated in double for precision. 0.0 when count == 0.
double sumOfSquares(const float* a, int count);

// Returns max(|a[i]|), 0.f when count == 0. Max is order-independent, so this
// one matches a sequential loop exactly.
float peakAbs(const float* a, int count);

// --- Byte scanning ---
//
// A ByteClass (ByteClass.h) names a set of bytes: anyOf('"', '\\') | below(0x20)
// is "a quote, a backslash or a control character". The scans below test
// sixteen or thirty-two bytes per step against it. They are what a text parser
// spends most of its time doing: run to the end of a plain string, skip
// whitespace, count the newlines before an error position. All three are exact
// and deterministic on every backend.

// Index of the first byte in [data, data + count) that is in `cls`, or `count`
// when there is none.
int findFirst(const std::uint8_t* data, int count, const ByteClass& cls);

// Index of the first byte in [data, data + count) that is NOT in `cls`, or
// `count` when every byte is.
int findFirstNot(const std::uint8_t* data, int count, const ByteClass& cls);

// How many bytes in [data, data + count) are in `cls`.
int countOf(const std::uint8_t* data, int count, const ByteClass& cls);

} // namespace esimd
