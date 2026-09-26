/*
 * matrix.h -- the kernel matrix, in one place.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * There are algorithms x ISAs x stream counts kernels. Each one needs an
 * instantiation, a forward declaration, and a registry row, and all three are
 * mechanically derivable from the table below. Writing them out by hand meant
 * over a hundred entries kept in sync across three files, which is exactly the
 * kind of bookkeeping that rots.
 *
 * So the table is the source of truth and the preprocessor expands it. Adding
 * an ISA is one block here plus its OPS_* macros; adding an algorithm is one
 * line in VB_FOR_ALGS plus its template. Neither touches registry.c.
 *
 * Expansion is layered rather than recursive, since the C preprocessor will not
 * re-expand a macro inside its own expansion: streams, then algorithms, then
 * ISAs, each a separate fixed macro.
 *
 *
 * WHY THE MATRIX IS ISA-MAJOR, AND MUST STAY SO
 * =============================================
 *
 * Each ISA gets one translation unit carrying every algorithm, rather than each
 * algorithm getting one carrying every ISA. That is not a style choice: a
 * translation unit can only be compiled with one set of -m flags, so an
 * algorithm-major layout could not give AVX-512 code -mavx512f while keeping it
 * out of the SSE2 build. Do not "fix" this.
 */

#ifndef VALUBENCH_KERNEL_MATRIX_H
#define VALUBENCH_KERNEL_MATRIX_H

/* ---- token plumbing ----------------------------------------------------- */

#define VB_STR_(x)      #x
#define VB_STR(x)       VB_STR_(x)
#define VB_CAT3_(a,b,c) a##b##c
#define VB_CAT3(a,b,c)  VB_CAT3_(a,b,c)

/* Entry point symbol for one kernel: vb_<alg>_<isa>_s<streams>. */
#define VB_KSYM(alg, isa, st) VB_CAT3(vb_##alg##_, isa, _s##st)

/* Human-facing name: "<alg>/<isa>-s<streams>", e.g. "sha1/avx2-s3". */
#define VB_KNAME(alg, isa, st) VB_STR(alg) "/" VB_STR(isa) "-s" VB_STR(st)

/* ---- the matrix --------------------------------------------------------- */
/*
 * Every visitor macro M is called as:
 *
 *   M(alg, isa, streams, isa_display_name, algorithm_id, available_fn,
 *     lanes, lanes_fn)
 *
 * `lanes` is per (isa, algorithm) because a register holds half as many 64-bit
 * messages as 32-bit ones -- SHA-512 gets the narrow count.
 *
 * `lanes_fn` is NULL for every fixed-width ISA. A vector-length-agnostic ISA
 * does not know its lane count until it runs, so it leaves `lanes` at 0 and
 * names a function instead; src/registry.c calls it once and writes the answer
 * into the row before anything reads it.
 */

#define VB_FOR_STREAMS(M, alg, isa, isaname, algid, avail, lanes, lfn) \
    M(alg, isa, 1, isaname, algid, avail, lanes, lfn)                  \
    M(alg, isa, 2, isaname, algid, avail, lanes, lfn)                  \
    M(alg, isa, 3, isaname, algid, avail, lanes, lfn)                  \
    M(alg, isa, 4, isaname, algid, avail, lanes, lfn)                  \
    M(alg, isa, 6, isaname, algid, avail, lanes, lfn)                  \
    M(alg, isa, 8, isaname, algid, avail, lanes, lfn)

/* SHA-NI is a fixed-function unit with its own template, which implements one
   to four streams and is not part of the stream-count question. */
#define VB_FOR_STREAMS4(M, alg, isa, isaname, algid, avail, lanes, lfn) \
    M(alg, isa, 1, isaname, algid, avail, lanes, lfn)                   \
    M(alg, isa, 2, isaname, algid, avail, lanes, lfn)                   \
    M(alg, isa, 3, isaname, algid, avail, lanes, lfn)                   \
    M(alg, isa, 4, isaname, algid, avail, lanes, lfn)

/* Add an algorithm here and in instantiate_all.h; nothing else changes. */
#define VB_FOR_ALGS(M, isa, isaname, avail, l32, l64, f32, f64)             \
    VB_FOR_STREAMS(M, md5,    isa, isaname, VB_ALG_MD5,    avail, l32, f32) \
    VB_FOR_STREAMS(M, sha1,   isa, isaname, VB_ALG_SHA1,   avail, l32, f32) \
    VB_FOR_STREAMS(M, sha512, isa, isaname, VB_ALG_SHA512, avail, l64, f64)

/*
 * One block per ISA. The guards are here rather than inside a visitor because
 * #if cannot appear in a macro expansion; an ISA the toolchain cannot build
 * simply expands to nothing.
 *
 * Columns: ISA token, display name, availability predicate, 32-bit lanes,
 * 64-bit lanes, 32-bit lane function, 64-bit lane function. The two lane
 * functions are NULL for every fixed-width ISA; only a vector-length-agnostic
 * one (SVE, SVE2) registers lanes of 0 and names them instead.
 */
#define VB_ISA_SCALAR(M) VB_FOR_ALGS(M, scalar, "scalar", always, 1, 1, NULL, NULL)

#if VB_HAVE_SSE2
#  define VB_ISA_SSE2(M) VB_FOR_ALGS(M, sse2, "SSE2", vb_cpu_has_sse2, 4, 2, NULL, NULL)
#else
#  define VB_ISA_SSE2(M)
#endif

#if VB_HAVE_AVX2
#  define VB_ISA_AVX2(M) VB_FOR_ALGS(M, avx2, "AVX2", vb_cpu_has_avx2, 8, 4, NULL, NULL)
#else
#  define VB_ISA_AVX2(M)
#endif

#if VB_HAVE_AVX512
#  define VB_ISA_AVX512(M) \
       VB_FOR_ALGS(M, avx512, "AVX512", vb_cpu_has_avx512f, 16, 8, NULL, NULL)
#else
#  define VB_ISA_AVX512(M)
#endif

/*
 * SHA-NI is not a vector-width variation on the others, so it does not use
 * VB_FOR_ALGS. It implements SHA-1 in fixed-function hardware and cannot do MD5
 * or SHA-512 at all, and its unit of work is one message rather than a vector
 * of them -- hence one algorithm and lanes = 1. See shani.c beside it.
 */
#if VB_HAVE_SHANI
#  define VB_ISA_SHANI(M) \
       VB_FOR_STREAMS4(M, sha1, shani, "SHA-NI", VB_ALG_SHA1, vb_cpu_has_sha_ni, 1, NULL)
#else
#  define VB_ISA_SHANI(M)
#endif

/*
 * NEON is 128-bit like SSE2, and is here to be compared with it: same width,
 * different operation set. It needs no -m flag, since Advanced SIMD is
 * mandatory on AArch64, and no arch guard of its own -- VB_HAVE_NEON is only
 * defined on targets where the build probed it successfully.
 */
#if VB_HAVE_NEON
#  define VB_ISA_NEON(M) VB_FOR_ALGS(M, neon, "NEON", vb_cpu_has_neon, 4, 2, NULL, NULL)
#else
#  define VB_ISA_NEON(M)
#endif

/*
 * SVE and SVE2 are the only ISAs whose lane count is not known here: the
 * vector length is 128 to 2048 bits and the hardware chooses. They register
 * lanes = 0 and name a function, and src/registry.c resolves it once.
 *
 * They are separate ISAs rather than one with a feature bit because for this
 * workload they are separate instruction sets. SVE has no three-input bitwise
 * select, no three-way XOR, no fused xor-rotate and no shift-right-insert;
 * SVE2 has all four. Neoverse V1 offers SVE at 256 bits, V2 offers SVE2 at
 * 128, so the comparison runs both ways on rentable hardware.
 */
#if VB_HAVE_SVE
#  define VB_ISA_SVE(M) \
       VB_FOR_ALGS(M, sve, "SVE", vb_cpu_has_sve, 0, 0, \
                   vb_sve_lanes32, vb_sve_lanes64)
#else
#  define VB_ISA_SVE(M)
#endif

#if VB_HAVE_SVE2
#  define VB_ISA_SVE2(M) \
       VB_FOR_ALGS(M, sve2, "SVE2", vb_cpu_has_sve2, 0, 0, \
                   vb_sve_lanes32, vb_sve_lanes64)
#else
#  define VB_ISA_SVE2(M)
#endif

/* ---- device kernels ------------------------------------------------------ */
/*
 * Device kernels are not an ISA matrix -- there is one OpenCL backend and only
 * the algorithm varies -- so they get their own list rather than going through
 * VB_FOR_ALGS. `lanes` is the corpus interleave width rather than a register
 * width, which is why it is one constant here and a per-ISA column above.
 *
 * Two files expand this: src/registry.c for its rows and src/opencl/backend.c
 * for its program table. They used to be maintained by hand, and registry.c
 * carried a comment asking the reader to keep them in step -- which is the kind
 * of instruction that survives exactly as long as the person who read it.
 *
 *   M(alg, ALG, algorithm_id, entry_point)
 *
 * ALG is the same token uppercased, because the embedded source symbol is
 * VB_OCL_<ALG>_SOURCE and the preprocessor cannot change case.
 */
#define VB_OCL_LANES 64

#define VB_FOR_EACH_DEVICE_ALG(M)                  \
    M(md5,    MD5,    VB_ALG_MD5,    "vb_md5")     \
    M(sha1,   SHA1,   VB_ALG_SHA1,   "vb_sha1")    \
    M(sha512, SHA512, VB_ALG_SHA512, "vb_sha512")

/* Add a new ISA to this list and give it a block above. */
#define VB_FOR_EACH_KERNEL(M) \
    VB_ISA_SCALAR(M)          \
    VB_ISA_SSE2(M)            \
    VB_ISA_AVX2(M)            \
    VB_ISA_AVX512(M)          \
    VB_ISA_SHANI(M)           \
    VB_ISA_NEON(M)            \
    VB_ISA_SVE(M)             \
    VB_ISA_SVE2(M)

#endif /* VALUBENCH_KERNEL_MATRIX_H */
