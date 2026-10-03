function(esimd_set_warnings target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif ()

    # A 64-bit count assigned into an int is silent under -Wall -Wextra, so the
    # one warning that catches it is on wherever it exists. Clang only: GCC has
    # no equivalent short of -Wconversion, which is a far larger and much
    # noisier set, and MSVC already reports it as C4244 under /W4.
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
        target_compile_options(${target} PRIVATE -Wshorten-64-to-32)
    endif ()
endfunction()

# Force a target to compile at the platform's maximum *speed* optimization in
# EVERY configuration -- including Debug -- without enabling fast-math: IEEE FP
# semantics and bit-exact results are preserved (no multiply-add contraction).
# Intended for hot numeric / SIMD kernels that are pointless at -O0 / -Od. Debug
# info (-g / /Zi) is left untouched, so the target stays debuggable, just fast.
#
# Removing cl's Debug-only /RTC1 and /Od edits the *directory-scoped* Debug flags
# (hence PARENT_SCOPE), so keep each force-optimized target in its own
# subdirectory -- the usual case. The rest of the project keeps the defaults.
#
# A consumer that shares a precompiled header across its targets must leave
# this one out: the flags below are not the ones such an image was created
# under, and MSVC rejects the mismatch outright.
function(esimd_force_optimization target)
    if (MSVC)
        # The cl/clang-cl Debug defaults fight optimization: /RTC1 is a hard
        # error under any /O level (D8016) and /Od warns when overridden by /O2
        # (D9025). Strip both before forcing the level below.
        foreach (lang C CXX)
            string(REGEX REPLACE "/RTC[1csu]+" "" CMAKE_${lang}_FLAGS_DEBUG
                    "${CMAKE_${lang}_FLAGS_DEBUG}")
            string(REGEX REPLACE "/Od" "" CMAKE_${lang}_FLAGS_DEBUG
                    "${CMAKE_${lang}_FLAGS_DEBUG}")
            set(CMAKE_${lang}_FLAGS_DEBUG "${CMAKE_${lang}_FLAGS_DEBUG}"
                    PARENT_SCOPE)
        endforeach ()

        # clang-cl reports CXX_COMPILER_ID==Clang but parses the MSVC-style
        # driver, so its GCC-style flags must tunnel through /clang: or they are
        # silently dropped -- and a dropped -ffp-contract=off lets clang-cl fuse
        # FMA, breaking bit-exactness. Real cl has no /O3 (tops out at /O2) and
        # never contracts FP by default (/fp:precise).
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE
                    /clang:-O3 /clang:-ffp-contract=off)
        else ()
            target_compile_options(${target} PRIVATE /O2)
        endif ()
    else ()
        target_compile_options(${target} PRIVATE -O3 -ffp-contract=off)
    endif ()
endfunction()
