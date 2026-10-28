#pragma once

#include "ByteClass.h"
#include "Common.h"

// Internal header: the per-backend entry points for the kernels, used by
// the runtime dispatcher, the unit tests and the benchmark. This is NOT the
// public ESIMD API -- include <ESIMD/ESIMD.h> for that. It is the one
// header where per-architecture / per-feature conditionals are allowed; the
// public interface stays free of them.

#include "Dispatch/Cpu.h"

namespace esimd::backends
{

void swapRedBlue_scalar(const std::uint8_t* in, std::uint8_t* out, int pixelCount);

void resizeBilinear_scalar(const std::uint8_t* src,
                           int srcWidth,
                           int srcHeight,
                           std::uint8_t* dst,
                           int dstWidth,
                           int dstHeight);

void warpAffineInverse_scalar(const std::uint8_t* src,
                              int srcWidth,
                              int srcHeight,
                              const float* inverse2x3,
                              std::uint8_t* dst,
                              int dstWidth,
                              int dstHeight);

int findFirst_scalar(const std::uint8_t* data, int count, const ByteClass& cls);
int findFirstNot_scalar(const std::uint8_t* data, int count, const ByteClass& cls);
int countOf_scalar(const std::uint8_t* data, int count, const ByteClass& cls);

void exp_scalar(const float* a, float* out, int count);
void exp2_scalar(const float* a, float* out, int count);
void log_scalar(const float* a, float* out, int count);
void log2_scalar(const float* a, float* out, int count);
void sin_scalar(const float* a, float* out, int count);
void cos_scalar(const float* a, float* out, int count);
void tanh_scalar(const float* a, float* out, int count);

#if defined(__x86_64__) || defined(_M_X64)
void swapRedBlue_sse2(const std::uint8_t* in, std::uint8_t* out, int pixelCount);
void resizeBilinear_sse2(const std::uint8_t* src,
                         int srcWidth,
                         int srcHeight,
                         std::uint8_t* dst,
                         int dstWidth,
                         int dstHeight);
void warpAffineInverse_sse2(const std::uint8_t* src,
                            int srcWidth,
                            int srcHeight,
                            const float* inverse2x3,
                            std::uint8_t* dst,
                            int dstWidth,
                            int dstHeight);
int findFirst_sse2(const std::uint8_t* data, int count, const ByteClass& cls);
int findFirstNot_sse2(const std::uint8_t* data, int count, const ByteClass& cls);
int countOf_sse2(const std::uint8_t* data, int count, const ByteClass& cls);
void exp_sse2(const float* a, float* out, int count);
void exp2_sse2(const float* a, float* out, int count);
void log_sse2(const float* a, float* out, int count);
void log2_sse2(const float* a, float* out, int count);
void sin_sse2(const float* a, float* out, int count);
void cos_sse2(const float* a, float* out, int count);
void tanh_sse2(const float* a, float* out, int count);
#if defined(ESIMD_HAS_AVX2)
void swapRedBlue_avx2(const std::uint8_t* in, std::uint8_t* out, int pixelCount);
int findFirst_avx2(const std::uint8_t* data, int count, const ByteClass& cls);
int findFirstNot_avx2(const std::uint8_t* data, int count, const ByteClass& cls);
int countOf_avx2(const std::uint8_t* data, int count, const ByteClass& cls);
void exp_avx2(const float* a, float* out, int count);
void exp2_avx2(const float* a, float* out, int count);
void log_avx2(const float* a, float* out, int count);
void log2_avx2(const float* a, float* out, int count);
void sin_avx2(const float* a, float* out, int count);
void cos_avx2(const float* a, float* out, int count);
void tanh_avx2(const float* a, float* out, int count);
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
void swapRedBlue_neon(const std::uint8_t* in, std::uint8_t* out, int pixelCount);
void resizeBilinear_neon(const std::uint8_t* src,
                         int srcWidth,
                         int srcHeight,
                         std::uint8_t* dst,
                         int dstWidth,
                         int dstHeight);
void warpAffineInverse_neon(const std::uint8_t* src,
                            int srcWidth,
                            int srcHeight,
                            const float* inverse2x3,
                            std::uint8_t* dst,
                            int dstWidth,
                            int dstHeight);
int findFirst_neon(const std::uint8_t* data, int count, const ByteClass& cls);
int findFirstNot_neon(const std::uint8_t* data, int count, const ByteClass& cls);
int countOf_neon(const std::uint8_t* data, int count, const ByteClass& cls);
void exp_neon(const float* a, float* out, int count);
void exp2_neon(const float* a, float* out, int count);
void log_neon(const float* a, float* out, int count);
void log2_neon(const float* a, float* out, int count);
void sin_neon(const float* a, float* out, int count);
void cos_neon(const float* a, float* out, int count);
void tanh_neon(const float* a, float* out, int count);
#endif

} // namespace esimd::backends
