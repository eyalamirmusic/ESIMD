# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Rules

Claude must never commit or push without explicit permission from the user in
the current conversation.

## What is ESIMD

A small, self-contained portable SIMD library in C++20 with no dependencies.
It began life as the `SIMD` module of [eacp](https://github.com/eyalamirmusic/eacp)
and became a repo of its own so other projects can take it without eacp; eacp
now fetches it through CPM like any other dependency. `README.md` is written
for users of the library — interface, guarantees and usage — and stays free of
build-system and implementation detail; that goes here.

Everything is in `namespace esimd`. The target is `ESIMD`, the include root is
`Lib/`, and the public headers are `<ESIMD/ESIMD.h>` (the array-at-a-time
kernels), `<ESIMD/Ops.h>` (buffer-level helpers over them) and
`<ESIMD/Vector.h>` (the register-level vector type). `<ESIMD/Backends.h>` is
internal: the per-backend entry points the dispatcher, the tests and the
benchmark call directly.

## Build & Test Commands

```bash
# Configure and build (tests and benchmark are on when this is the top-level
# project; a consumer gets the library alone unless it sets ESIMD_BUILD_TESTS)
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run every test, or one by name
ctest --test-dir build --output-on-failure
./build/Tests/ESIMDTests --test ESIMD/baselineSwapMatchesScalar
./build/Tests/ESIMDTests --list-tests

# Scalar reference against the active backend
./build/Tests/ESIMDBench

# What CI runs on every lane: Debug plus the unity build
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DESIMD_UNITY_BUILD=ON

# A universal build has no AVX2 translation unit (see below)
cmake -G Ninja -B build-universal -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"

# Format edited files
clang-format -i <files>
```

Debug is the configuration that matters: the library forces its own
optimization level in every configuration, and a Debug lane is where that has
to hold. CI (`.github/workflows/ci.yml`) builds and tests macOS universal and
arm64, Linux GCC and Clang, and Windows x64 and ARM64 on MSVC and clang-cl.
An Apple Silicon Mac without Rosetta can compile an x86_64 build but not run
it, so the AVX2 path is first exercised by CI's x64 lanes.

### Options

- `ESIMD_UNITY_BUILD` (default `OFF`): compiles the library as a CMake unity
  build. The AVX2 translation unit is always kept out of the blob
  (`SKIP_UNITY_BUILD_INCLUSION`), since its ISA flag must reach no other file.
  Off by default because a unity build collapses the per-file entries in
  `compile_commands.json` that LSP tooling reads; Claude keeps it off.
- `ESIMD_BUILD_TESTS` (default `OFF`, on when top-level): builds `ESIMDTests`
  and `ESIMDBench`, which fetch NanoTest and ea_data_structures via CPM
  (`CMake/FindNanoTest.cmake`, `CMake/FindEADataStructures.cmake`). The
  library itself fetches nothing and links nothing.

### Co-developing with a consumer

A consumer that fetches ESIMD through CPM builds against a local checkout with
`-DCPM_ESIMD_SOURCE=$HOME/Code/ESIMD`. Use `$HOME`, not `~`: CMake does not
expand it and tilde expansion is suppressed inside quotes. A plain configure
fetches `main` from GitHub, so a change must be pushed before a consumer's CI
sees it.

## Architecture

Two halves with opposite compilation rules, and the distinction is the point:

- **The array-at-a-time half** (`ESIMD.h`, implemented under `Dispatch/` and
  `Tu/`) is compiled at the *library's* settings — always `-O3`, FP contraction
  off, no fast-math — so every result is bit-identical on every build,
  configuration and architecture. It walks whole buffers and is what a Debug
  build of an app calls to get Release-speed kernels.
- **The register-level half** (`Vector.h`) is header-only on purpose so it
  compiles at the *consumer's* settings. A microkernel keeps accumulators in
  registers across a reduction and may want FMA contraction; it turns that on
  for its own target without touching the array half's guarantee. The width
  is whatever the including translation unit is compiled for; nothing in it is
  runtime-dispatched.

The array half, bottom up:

- `Common.h` — the shared standard-library includes.
- `Backend/` — `Scalar`, `Sse2`, `Avx2` and `Neon`: each a struct of lane types
  (`U32`, `F4`) with the same operation set. `Scalar` is the reference the
  SIMD backends mirror lane-for-lane and the oracle the tests validate against.
- `Kernels/` — the image kernels (`SwapRedBlue`, `Bilinear`, `ResizeBilinear`,
  `WarpAffine`), each written once as a template over a backend `B`. The main
  loop processes `B::U32::lanes` pixels at a time and the remainder runs through
  `Scalar`, which is what keeps every backend bit-identical to the oracle.
- `Tu/` — one translation unit per ISA, instantiating the kernels for its
  backend: `Scalar.cpp`, `Baseline.cpp` (SSE2 on x86-64, NEON on arm64, the
  ISA the whole target is compiled for) and `Avx2.cpp`, which alone is
  compiled with `-mavx2 -mfma` / `/arch:AVX2`. `ArrayOps.cpp` holds the
  float-array primitives and reductions: memory-bound and auto-vectorized, so
  they need no backend and no dispatch. `MirroredCrop.cpp` is a byte copy that
  lives here purely to be built `-O3`.
- `Dispatch/` — `Cpu.{h,cpp}` is CPUID feature detection (`hasAvx2Fma`);
  `Dispatch.cpp` is the public entry points, each resolving its backend once
  into a function pointer. That pointer is a deliberate, non-inlinable
  boundary between baseline and AVX2 code.
- `Backends.h` — the per-backend entry points, and the one header where
  per-architecture conditionals are allowed. The public interface carries
  none.

### The rules the CMake encodes, and why

`Lib/ESIMD/CMakeLists.txt` and `CMake/ESIMDTargetSetup.cmake` carry
correctness rules, not conveniences. Do not relax them for a build that
happens to be inconvenient:

- **Forced optimization in every configuration**
  (`esimd_force_optimization`): at `-O0`/`/Od` the intrinsics spill and the
  library is pointless. The helper adds `-O3 -ffp-contract=off` (clang-cl:
  `/clang:-O3 /clang:-ffp-contract=off`, since a bare GCC-style flag is
  silently dropped and a dropped contraction flag lets clang-cl fuse FMA and
  break bit-exactness; cl: `/O2`, which never contracts under `/fp:precise`).
  On MSVC it also strips `/RTC1` and `/Od` from the *directory-scoped* Debug
  flags, which is why the library and the benchmark each sit in a directory
  of their own and `ESIMDBench` is only forced outside Debug — forcing it there
  would strip the runtime checks off `ESIMDTests` beside it.
- **No IPO/LTO, whatever the enclosing project sets**: under LTO the optimizer
  could hoist AVX2 code into a baseline caller (on MSVC `/LTCG` there is no
  per-function target attribute to stop it) and fault on a pre-AVX2 CPU. The
  dispatch boundary is non-inlinable by design, so nothing is lost.
- **AVX2 only for a single-arch x86-64 target**: a per-source `-mavx2` cannot
  be expressed per slice of an Apple universal compile, so a universal build
  has no `Avx2.cpp` and its x86-64 slice runs the SSE2 baseline.
  `ESIMD_HAS_AVX2` is a PUBLIC define so every consumer sees the same
  declarations in `Backends.h`.
- **Shared precompiled headers**: a consumer that shares one image across its
  targets must leave `ESIMD` out of it; the flags above are not the ones the
  image was created under, and MSVC rejects the mismatch outright. eacp's PCH
  walk skips CPM-fetched packages for this reason.

### Determinism contract

The elementwise primitives and the image kernels are bit-identical at every
vector width and on every build. The reductions (`sumOfSquares`, `peakAbs`)
use a fixed four-lane interleave so they are deterministic across builds and
architectures, but not bit-equal to a naive sequential loop. Any change to a
kernel must keep its test against the scalar oracle passing on every backend,
and a new backend operation is added to `Scalar` first.

### Tests

`Tests/ESIMDTests.cpp` (NanoTest, prefix `ESIMD/`) checks every backend
against the scalar oracle at pixel counts chosen to straddle every lane width —
zero, sub-lane, exact multiples and odd remainders — and the dispatched entry
points against the oracle again. An AVX2 case self-skips on a CPU without it.
`Tests/ESIMDBench.cpp` times the scalar reference against the active backend
and disables auto-vectorization on its reference loops so the comparison shows
the SIMD win. Both use `EA::Vector`/`EA::Array`/`EA::Span` from
ea_data_structures as the buffer types `Ops.h` is exercised with.

## Code Style

Always use the most modern C++ and RAII practices.
Use auto for variables and whenever possible.
Don't use auto for functions and member functions.

Don't use comments unless absolutely needed. Use named functions to make code
self documenting. The comments that are here explain a non-obvious rule — a
flag, a boundary, a determinism guarantee — and should stay.

Enforced via `.clang-format`:
- Allman brace style
- 85 column limit
- 4-space indentation (no tabs)
- Pointer alignment: left (`int* ptr`)
- Break constructor initializers before comma

Always run clang-format for edited code files.

New source files are added to `Lib/ESIMD/CMakeLists.txt` under
`target_sources(ESIMD PRIVATE ...)`; a new ISA gets a translation unit of its
own under `Tu/` with its flag applied per source, never target-wide.
