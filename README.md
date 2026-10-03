# ESIMD

A small, self-contained portable SIMD library in C++20. It has no dependencies,
and it is built so that its results are bit-identical on every build,
configuration, architecture and instruction set.

It has two halves:

- **Array-at-a-time kernels** (`ESIMD/ESIMD.h`). Image kernels over tightly
  packed RGBA8 — red/blue swap, BGRA-to-RGBA conversion, bilinear resize, inverse
  affine warp, mirrored crop — and elementwise float-array primitives (`add`,
  `subtract`, `multiply`, `multiplyByScalar`, `multiplyAdd`, `lerp`) with two
  reductions (`sumOfSquares`, `peakAbs`). Each call selects the widest backend
  the running CPU has, once, behind a function pointer: SSE2 and AVX2 on x86-64,
  NEON on arm64, scalar everywhere. The library compiles its own code at full
  optimization in every configuration, with FMA contraction and fast-math off,
  so a Debug build of your app still runs the kernels at speed and gets the
  same bits as a Release one.
- **A register-level vector type** (`ESIMD/Vector.h`). `esimd::F32` and friends
  wrap one register of whatever width the translation unit is compiled for.
  It is header-only on purpose: it compiles at *your* settings, so a kernel
  that wants FMA contraction turns it on for its own target.

`ESIMD/Ops.h` adds buffer-level conveniences over the array primitives for
anything with `data()` and `size()` — `std::vector`, `std::span`, a fixed array —
so a call site reads `esimd::multiply(buffer, gain)`.

## Using it

With [CPM](https://github.com/cpm-cmake/CPM.cmake):

```cmake
CPMAddPackage(
        NAME ESIMD
        GITHUB_REPOSITORY eyalamirmusic/ESIMD
        GIT_TAG main)

target_link_libraries(MyApp PRIVATE ESIMD)
```

Or as a submodule or vendored copy, with `add_subdirectory(ESIMD)`. The target is
`ESIMD`, the include root is `Lib/`, and everything is in `namespace esimd`:

```cpp
#include <ESIMD/ESIMD.h>
#include <ESIMD/Ops.h>

esimd::swapRedBlue(pixels.data(), pixels.data(), pixelCount);
esimd::multiplyAdd(mix, dry, 0.5f);
```

To work against a local checkout while developing both sides, pass
`-DCPM_ESIMD_SOURCE=$HOME/Code/ESIMD` at configure time.

### Options

| Option | Default | Effect |
| --- | --- | --- |
| `ESIMD_UNITY_BUILD` | `OFF` | Compile the library as a CMake unity build. The AVX2 translation unit is always excluded from the blob. |
| `ESIMD_BUILD_TESTS` | `OFF` (`ON` when top-level) | Build the test suite and the benchmark. They fetch NanoTest and ea_data_structures through CPM; the library itself fetches nothing. |

## Determinism

The elementwise primitives and the image kernels are bit-identical at every
vector width and on every build: the library is compiled with FP contraction
off and no fast-math, and the SIMD main loop hands the remainder to the scalar
backend. The reductions use a fixed four-lane interleave, so they are
deterministic across builds and architectures too, but not bit-equal to a
naive sequential loop.

AVX2 is a separately compiled translation unit chosen at runtime, and the
library opts itself out of LTO so the optimizer can never hoist AVX2 code into
a baseline caller. An Apple universal (`arm64;x86_64`) build has no AVX2 unit,
because a per-source `-mavx2` cannot be expressed per slice; its x86-64 slice
runs on the SSE2 baseline.

## Layout

```
Lib/ESIMD/
  ESIMD.h       The public array-at-a-time API
  Ops.h         Buffer-level helpers over it
  Vector.h      The register-level vector type
  Backends.h    Per-backend entry points (internal; the tests and bench use it)
  Backend/      Scalar, SSE2, AVX2 and NEON lane types
  Kernels/      The image kernels, written once over a backend
  Dispatch/     CPU feature detection and the runtime selection
  Tu/           One translation unit per ISA
Tests/
  ESIMDTests.cpp   Every backend against the scalar oracle
  ESIMDBench.cpp   Scalar reference against the active backend
```

## Building the tests

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/Tests/ESIMDBench
```

CI builds and tests macOS (universal and arm64), Linux (GCC and Clang) and
Windows (x64 and ARM64, MSVC and clang-cl).

## License

MIT. See `LICENSE`.
