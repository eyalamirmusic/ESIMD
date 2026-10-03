#pragma once

#include "ESIMD.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <type_traits>

// Header-only, buffer-level conveniences over the raw primitives in ESIMD.h.
// Anything contiguous with data()/size() qualifies (EA::Vector, EA::Array,
// EA::Span, std::vector, std::string, std::string_view, std::span, ...), so
// call sites can write `multiply(buffer, gain)` or `findFirst(text, '"')`
// instead of spelling out pointers and counts.
//
// The float helpers are elementwise and in place on their first argument. When
// the buffers disagree on size, the common prefix is processed -- never past
// the end of the shortest one.
//
// The byte helpers take any buffer of char, unsigned char, signed char,
// std::byte or char8_t, and report positions as int indices into it.
namespace esimd
{

template <typename B>
concept FloatBuffer = requires(const B& b) {
    { b.data() } -> std::convertible_to<const float*>;
    { b.size() } -> std::convertible_to<int>;
};

template <typename B>
concept MutableFloatBuffer = FloatBuffer<B> && requires(B& b) {
    { b.data() } -> std::convertible_to<float*>;
};

template <FloatBuffer... Buffers>
int commonCount(const Buffers&... buffers)
{
    return std::min({(int) buffers.size()...});
}

// dst[i] += src[i]
template <MutableFloatBuffer Dst, FloatBuffer Src>
void add(Dst& dst, const Src& src)
{
    add(dst.data(), src.data(), dst.data(), commonCount(dst, src));
}

// dst[i] -= src[i]
template <MutableFloatBuffer Dst, FloatBuffer Src>
void subtract(Dst& dst, const Src& src)
{
    subtract(dst.data(), src.data(), dst.data(), commonCount(dst, src));
}

// dst[i] *= src[i]
template <MutableFloatBuffer Dst, FloatBuffer Src>
void multiply(Dst& dst, const Src& src)
{
    multiply(dst.data(), src.data(), dst.data(), commonCount(dst, src));
}

// dst[i] *= value
template <MutableFloatBuffer Dst>
void multiply(Dst& dst, float value)
{
    multiplyByScalar(dst.data(), value, dst.data(), (int) dst.size());
}

// dst[i] += a[i] * b[i]
template <MutableFloatBuffer Dst, FloatBuffer A, FloatBuffer B>
void multiplyAdd(Dst& dst, const A& a, const B& b)
{
    multiplyAdd(a.data(), b.data(), dst.data(), dst.data(), commonCount(dst, a, b));
}

// dst[i] += src[i] * value
template <MutableFloatBuffer Dst, FloatBuffer Src>
void multiplyAdd(Dst& dst, const Src& src, float value)
{
    multiplyAdd(src.data(), value, dst.data(), dst.data(), commonCount(dst, src));
}

// dst[i] += t * (target[i] - dst[i])
template <MutableFloatBuffer Dst, FloatBuffer Target>
void lerp(Dst& dst, const Target& target, float t)
{
    lerp(dst.data(), target.data(), t, dst.data(), commonCount(dst, target));
}

// sum(src[i]^2)
template <FloatBuffer Src>
double sumOfSquares(const Src& src)
{
    return sumOfSquares(src.data(), (int) src.size());
}

// max(|src[i]|)
template <FloatBuffer Src>
float peakAbs(const Src& src)
{
    return peakAbs(src.data(), (int) src.size());
}

// --- Byte scanning over containers ---

template <typename T>
concept ByteLike = std::same_as<std::remove_cv_t<T>, char>
                   || std::same_as<std::remove_cv_t<T>, unsigned char>
                   || std::same_as<std::remove_cv_t<T>, signed char>
                   || std::same_as<std::remove_cv_t<T>, std::byte>
                   || std::same_as<std::remove_cv_t<T>, char8_t>;

template <typename B>
concept ByteBuffer = requires(const B& b) {
    { b.size() } -> std::convertible_to<int>;
    requires ByteLike<std::remove_pointer_t<decltype(b.data())>>;
};

template <ByteBuffer Buffer>
const std::uint8_t* byteData(const Buffer& buffer)
{
    return reinterpret_cast<const std::uint8_t*>(buffer.data());
}

template <ByteBuffer Buffer>
int byteCount(const Buffer& buffer)
{
    return static_cast<int>(buffer.size());
}

// Index of the first byte at or after `from` that is in `cls`, or the
// buffer's size when there is none. `from` is clamped to the buffer, so a
// parser can advance with `pos = findFirst(text, cls, pos)` and stop at size.
template <ByteBuffer Buffer>
int findFirst(const Buffer& buffer, const ByteClass& cls, int from = 0)
{
    const auto size = byteCount(buffer);
    from = std::clamp(from, 0, size);
    return from + findFirst(byteData(buffer) + from, size - from, cls);
}

// Index of the first byte at or after `from` that is NOT in `cls`, or the
// buffer's size when every byte from there on is.
template <ByteBuffer Buffer>
int findFirstNot(const Buffer& buffer, const ByteClass& cls, int from = 0)
{
    const auto size = byteCount(buffer);
    from = std::clamp(from, 0, size);
    return from + findFirstNot(byteData(buffer) + from, size - from, cls);
}

// Whether any byte of the buffer is in `cls`.
template <ByteBuffer Buffer>
bool contains(const Buffer& buffer, const ByteClass& cls)
{
    return findFirst(buffer, cls) < byteCount(buffer);
}

// Whether every byte of the buffer is in `cls` (true for an empty buffer), so
// `allOf(text, below(0x80))` asks "is this pure ASCII?".
template <ByteBuffer Buffer>
bool allOf(const Buffer& buffer, const ByteClass& cls)
{
    return findFirstNot(buffer, cls) == byteCount(buffer);
}

// How many bytes of the buffer are in `cls`.
template <ByteBuffer Buffer>
int countOf(const Buffer& buffer, const ByteClass& cls)
{
    return countOf(byteData(buffer), byteCount(buffer), cls);
}

} // namespace esimd
