/*
 * sha1_device_impl.h -- the SHA-1 device kernel, in no particular dialect.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Same construction as md5_device_impl.h, which describes how a program is
 * assembled: a dialect header, device_primitives.h, then this. LANES and
 * STREAMS arrive as -D so one source specialises into every variant, and the
 * constants and all 80 steps are written out below from FIPS 180-4. The work
 * decomposition is identical to the CPU template, so the XOR checksum matches
 * it bit for bit.
 *
 * The one structural difference from MD5 is the message schedule, and it costs
 * registers. MD5 permutes sixteen message words, so md5_device_impl.h can
 * leave them in global memory and read each one as the step needs it. SHA-1
 * *expands* sixteen words into eighty, so the rolling window has to be
 * materialised: sixteen live words per stream, on top of the five state words.
 * At STREAMS=4 that is 84 private words per work-item, which on most devices
 * spills or costs occupancy.
 *
 * That is a real property of the algorithm on this hardware, not something to
 * engineer around -- it is why the harness sweeps the stream count and picks by
 * measurement instead of assuming more streams is better.
 */

/* Ch and Maj are primitives, spelled per target in device_primitives.h;
   parity is left plain for the compiler to fuse. */
#define S1CH(x, y, z)  VB_CH32((x), (y), (z))
#define S1PAR(x, y, z) ((x) ^ (y) ^ (z))
#define S1MAJ(x, y, z) VB_MAJ32((x), (y), (z))

/*
 * In-place round. The five state variables are renamed rather than moved, so
 * the host generator rotates the argument names one position per step and no
 * copies are emitted. `b` is rotated in place because the next step reads it as
 * its `c`.
 */
#define STEP(f, a, b, c, d, e, k, t, KC)                        \
    e[k] += VB_ROTL32(a[k], 5) + f(b[k], c[k], d[k]) + KC       \
          + w[k][(t) & 15];                                     \
    b[k] = VB_ROTL32(b[k], 30);

/* W[t] = ROTL1(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]), the indices reduced into
   the sixteen-entry rolling window so the update is in place. */
#define EXP1(k, t)                                              \
    w[k][(t) & 15] = VB_ROTL32(w[k][((t) -  3) & 15] ^          \
                               w[k][((t) -  8) & 15] ^          \
                               w[k][((t) - 14) & 15] ^          \
                               w[k][((t) - 16) & 15], 1);

/* Streams are expanded by macro, never by a loop -- if a loop failed to unroll
   the kernel would collapse to one dependency chain and silently under-report
   the device. Same reasoning as the CPU template. */
#if STREAMS == 1
#define EACH(f, a, b, c, d, e, t, KC) STEP(f,a,b,c,d,e,0,t,KC)
#define EXPAND(t) EXP1(0,t)
#elif STREAMS == 2
#define EACH(f, a, b, c, d, e, t, KC) STEP(f,a,b,c,d,e,0,t,KC) STEP(f,a,b,c,d,e,1,t,KC)
#define EXPAND(t) EXP1(0,t) EXP1(1,t)
#elif STREAMS == 3
#define EACH(f, a, b, c, d, e, t, KC) STEP(f,a,b,c,d,e,0,t,KC) STEP(f,a,b,c,d,e,1,t,KC) STEP(f,a,b,c,d,e,2,t,KC)
#define EXPAND(t) EXP1(0,t) EXP1(1,t) EXP1(2,t)
#elif STREAMS == 4
#define EACH(f, a, b, c, d, e, t, KC) STEP(f,a,b,c,d,e,0,t,KC) STEP(f,a,b,c,d,e,1,t,KC) STEP(f,a,b,c,d,e,2,t,KC) STEP(f,a,b,c,d,e,3,t,KC)
#define EXPAND(t) EXP1(0,t) EXP1(1,t) EXP1(2,t) EXP1(3,t)
#else
#error "STREAMS must be 1..4"
#endif

/* Sixteen explicit loads rather than a loop: the window only stays in registers
   if every index is a compile-time constant, and nothing here may depend on a
   compiler choosing to unroll. */
#define LOADW(k)                                            \
    w[k][ 0] = wp[k][ 0 * LANES]; w[k][ 1] = wp[k][ 1 * LANES]; \
    w[k][ 2] = wp[k][ 2 * LANES]; w[k][ 3] = wp[k][ 3 * LANES]; \
    w[k][ 4] = wp[k][ 4 * LANES]; w[k][ 5] = wp[k][ 5 * LANES]; \
    w[k][ 6] = wp[k][ 6 * LANES]; w[k][ 7] = wp[k][ 7 * LANES]; \
    w[k][ 8] = wp[k][ 8 * LANES]; w[k][ 9] = wp[k][ 9 * LANES]; \
    w[k][10] = wp[k][10 * LANES]; w[k][11] = wp[k][11 * LANES]; \
    w[k][12] = wp[k][12 * LANES]; w[k][13] = wp[k][13 * LANES]; \
    w[k][14] = wp[k][14 * LANES]; w[k][15] = wp[k][15 * LANES];

#define DIGEST_WORDS 5

/*
 * The four round constants, FIPS 180-4 section 4.2.1. The CPU path holds the
 * same values in src/reference/sha1.c and in its kernel templates; `make check`
 * validates every device kernel against the scalar reference, so the two cannot
 * drift apart unnoticed.
 */
VB_CONST_TABLE vb_u32 K[4] = {
    0x5a827999u, 0x6ed9eba1u, 0x8f1bbcdcu, 0xca62c1d6u,
};

VB_KERNEL vb_sha1(VB_GLOBAL const vb_u32 *corpus,
                  const vb_u32 blocks,
                  const vb_u32 iterations,
                  const vb_u64 n_groups,
                  const vb_u32 repeats,
                  VB_GLOBAL vb_u32 *partials
                  VB_SCRATCH_PARAM(vb_u32, scratch))
{
    VB_SCRATCH_DECL(vb_u32, scratch)

    const size_t gid  = VB_GLOBAL_ID();
    const size_t lid  = VB_LOCAL_ID();
    const size_t lane = gid % LANES;

    const size_t slot_words  = (size_t)blocks * 16 * LANES;
    const size_t block_words = 16 * LANES;

    const size_t g_stride = VB_GLOBAL_SIZE() / LANES;

    vb_u32 acc[DIGEST_WORDS];
    for (int j = 0; j < DIGEST_WORDS; j++)
        acc[j] = 0;

    /* Sweep the corpus `repeats` times, always an odd count so XORing every
       pass leaves the single-pass checksum intact. See md5_device_impl.h. */
    for (vb_u32 rep = 0; rep < repeats; rep++)
    for (size_t g = gid / LANES; g < n_groups; g += g_stride) {
        VB_GLOBAL const vb_u32 *slot[STREAMS];
        VB_GLOBAL const vb_u32 *wp[STREAMS];
        vb_u32 w[STREAMS][16];
        vb_u32 fb[STREAMS][DIGEST_WORDS];
        vb_u32 h[STREAMS][DIGEST_WORDS];
        vb_u32 A[STREAMS], B[STREAMS], C[STREAMS], D[STREAMS], E[STREAMS];

        for (int k = 0; k < STREAMS; k++) {
            slot[k] = corpus + (g * STREAMS + k) * slot_words + lane;
            fb[k][0] = slot[k][0 * LANES];
            fb[k][1] = slot[k][1 * LANES];
            fb[k][2] = slot[k][2 * LANES];
            fb[k][3] = slot[k][3 * LANES];
            fb[k][4] = slot[k][4 * LANES];
        }

        for (vb_u32 it = 0; it < iterations; it++) {
            for (int k = 0; k < STREAMS; k++) {
                h[k][0] = 0x67452301u; h[k][1] = 0xefcdab89u;
                h[k][2] = 0x98badcfeu; h[k][3] = 0x10325476u;
                h[k][4] = 0xc3d2e1f0u;
            }

            for (vb_u32 b = 0; b < blocks; b++) {
                for (int k = 0; k < STREAMS; k++) {
                    wp[k] = slot[k] + (size_t)b * block_words;
                    LOADW(k)

                    /* Block 0 carries the previous digest over the head of the
                       message; every other block is corpus data unchanged. */
                    if (b == 0) {
                        w[k][0] = fb[k][0]; w[k][1] = fb[k][1];
                        w[k][2] = fb[k][2]; w[k][3] = fb[k][3];
                        w[k][4] = fb[k][4];
                    }

                    A[k] = h[k][0]; B[k] = h[k][1]; C[k] = h[k][2];
                    D[k] = h[k][3]; E[k] = h[k][4];
                }

                EACH(S1CH, A, B, C, D, E, 0, K[0])
                EACH(S1CH, E, A, B, C, D, 1, K[0])
                EACH(S1CH, D, E, A, B, C, 2, K[0])
                EACH(S1CH, C, D, E, A, B, 3, K[0])
                EACH(S1CH, B, C, D, E, A, 4, K[0])
                EACH(S1CH, A, B, C, D, E, 5, K[0])
                EACH(S1CH, E, A, B, C, D, 6, K[0])
                EACH(S1CH, D, E, A, B, C, 7, K[0])
                EACH(S1CH, C, D, E, A, B, 8, K[0])
                EACH(S1CH, B, C, D, E, A, 9, K[0])
                EACH(S1CH, A, B, C, D, E, 10, K[0])
                EACH(S1CH, E, A, B, C, D, 11, K[0])
                EACH(S1CH, D, E, A, B, C, 12, K[0])
                EACH(S1CH, C, D, E, A, B, 13, K[0])
                EACH(S1CH, B, C, D, E, A, 14, K[0])
                EACH(S1CH, A, B, C, D, E, 15, K[0])
                EXPAND(16)
                EACH(S1CH, E, A, B, C, D, 16, K[0])
                EXPAND(17)
                EACH(S1CH, D, E, A, B, C, 17, K[0])
                EXPAND(18)
                EACH(S1CH, C, D, E, A, B, 18, K[0])
                EXPAND(19)
                EACH(S1CH, B, C, D, E, A, 19, K[0])
                EXPAND(20)
                EACH(S1PAR, A, B, C, D, E, 20, K[1])
                EXPAND(21)
                EACH(S1PAR, E, A, B, C, D, 21, K[1])
                EXPAND(22)
                EACH(S1PAR, D, E, A, B, C, 22, K[1])
                EXPAND(23)
                EACH(S1PAR, C, D, E, A, B, 23, K[1])
                EXPAND(24)
                EACH(S1PAR, B, C, D, E, A, 24, K[1])
                EXPAND(25)
                EACH(S1PAR, A, B, C, D, E, 25, K[1])
                EXPAND(26)
                EACH(S1PAR, E, A, B, C, D, 26, K[1])
                EXPAND(27)
                EACH(S1PAR, D, E, A, B, C, 27, K[1])
                EXPAND(28)
                EACH(S1PAR, C, D, E, A, B, 28, K[1])
                EXPAND(29)
                EACH(S1PAR, B, C, D, E, A, 29, K[1])
                EXPAND(30)
                EACH(S1PAR, A, B, C, D, E, 30, K[1])
                EXPAND(31)
                EACH(S1PAR, E, A, B, C, D, 31, K[1])
                EXPAND(32)
                EACH(S1PAR, D, E, A, B, C, 32, K[1])
                EXPAND(33)
                EACH(S1PAR, C, D, E, A, B, 33, K[1])
                EXPAND(34)
                EACH(S1PAR, B, C, D, E, A, 34, K[1])
                EXPAND(35)
                EACH(S1PAR, A, B, C, D, E, 35, K[1])
                EXPAND(36)
                EACH(S1PAR, E, A, B, C, D, 36, K[1])
                EXPAND(37)
                EACH(S1PAR, D, E, A, B, C, 37, K[1])
                EXPAND(38)
                EACH(S1PAR, C, D, E, A, B, 38, K[1])
                EXPAND(39)
                EACH(S1PAR, B, C, D, E, A, 39, K[1])
                EXPAND(40)
                EACH(S1MAJ, A, B, C, D, E, 40, K[2])
                EXPAND(41)
                EACH(S1MAJ, E, A, B, C, D, 41, K[2])
                EXPAND(42)
                EACH(S1MAJ, D, E, A, B, C, 42, K[2])
                EXPAND(43)
                EACH(S1MAJ, C, D, E, A, B, 43, K[2])
                EXPAND(44)
                EACH(S1MAJ, B, C, D, E, A, 44, K[2])
                EXPAND(45)
                EACH(S1MAJ, A, B, C, D, E, 45, K[2])
                EXPAND(46)
                EACH(S1MAJ, E, A, B, C, D, 46, K[2])
                EXPAND(47)
                EACH(S1MAJ, D, E, A, B, C, 47, K[2])
                EXPAND(48)
                EACH(S1MAJ, C, D, E, A, B, 48, K[2])
                EXPAND(49)
                EACH(S1MAJ, B, C, D, E, A, 49, K[2])
                EXPAND(50)
                EACH(S1MAJ, A, B, C, D, E, 50, K[2])
                EXPAND(51)
                EACH(S1MAJ, E, A, B, C, D, 51, K[2])
                EXPAND(52)
                EACH(S1MAJ, D, E, A, B, C, 52, K[2])
                EXPAND(53)
                EACH(S1MAJ, C, D, E, A, B, 53, K[2])
                EXPAND(54)
                EACH(S1MAJ, B, C, D, E, A, 54, K[2])
                EXPAND(55)
                EACH(S1MAJ, A, B, C, D, E, 55, K[2])
                EXPAND(56)
                EACH(S1MAJ, E, A, B, C, D, 56, K[2])
                EXPAND(57)
                EACH(S1MAJ, D, E, A, B, C, 57, K[2])
                EXPAND(58)
                EACH(S1MAJ, C, D, E, A, B, 58, K[2])
                EXPAND(59)
                EACH(S1MAJ, B, C, D, E, A, 59, K[2])
                EXPAND(60)
                EACH(S1PAR, A, B, C, D, E, 60, K[3])
                EXPAND(61)
                EACH(S1PAR, E, A, B, C, D, 61, K[3])
                EXPAND(62)
                EACH(S1PAR, D, E, A, B, C, 62, K[3])
                EXPAND(63)
                EACH(S1PAR, C, D, E, A, B, 63, K[3])
                EXPAND(64)
                EACH(S1PAR, B, C, D, E, A, 64, K[3])
                EXPAND(65)
                EACH(S1PAR, A, B, C, D, E, 65, K[3])
                EXPAND(66)
                EACH(S1PAR, E, A, B, C, D, 66, K[3])
                EXPAND(67)
                EACH(S1PAR, D, E, A, B, C, 67, K[3])
                EXPAND(68)
                EACH(S1PAR, C, D, E, A, B, 68, K[3])
                EXPAND(69)
                EACH(S1PAR, B, C, D, E, A, 69, K[3])
                EXPAND(70)
                EACH(S1PAR, A, B, C, D, E, 70, K[3])
                EXPAND(71)
                EACH(S1PAR, E, A, B, C, D, 71, K[3])
                EXPAND(72)
                EACH(S1PAR, D, E, A, B, C, 72, K[3])
                EXPAND(73)
                EACH(S1PAR, C, D, E, A, B, 73, K[3])
                EXPAND(74)
                EACH(S1PAR, B, C, D, E, A, 74, K[3])
                EXPAND(75)
                EACH(S1PAR, A, B, C, D, E, 75, K[3])
                EXPAND(76)
                EACH(S1PAR, E, A, B, C, D, 76, K[3])
                EXPAND(77)
                EACH(S1PAR, D, E, A, B, C, 77, K[3])
                EXPAND(78)
                EACH(S1PAR, C, D, E, A, B, 78, K[3])
                EXPAND(79)
                EACH(S1PAR, B, C, D, E, A, 79, K[3])


                for (int k = 0; k < STREAMS; k++) {
                    h[k][0] += A[k]; h[k][1] += B[k]; h[k][2] += C[k];
                    h[k][3] += D[k]; h[k][4] += E[k];
                }
            }

            for (int k = 0; k < STREAMS; k++) {
                fb[k][0] = h[k][0]; fb[k][1] = h[k][1]; fb[k][2] = h[k][2];
                fb[k][3] = h[k][3]; fb[k][4] = h[k][4];
            }
        }

        for (int k = 0; k < STREAMS; k++)
            for (int j = 0; j < DIGEST_WORDS; j++)
                acc[j] ^= h[k][j];
    }

    /* Reduce within the work-group so only DIGEST_WORDS per group cross the
       bus. A partial per work-item would put megabytes of readback inside the
       timed region and corrupt the PCIe measurement this exists to make. */
    for (int j = 0; j < DIGEST_WORDS; j++)
        scratch[lid * DIGEST_WORDS + j] = acc[j];
    VB_BARRIER();

    for (size_t s = VB_LOCAL_SIZE() / 2; s > 0; s >>= 1) {
        if (lid < s)
            for (int j = 0; j < DIGEST_WORDS; j++)
                scratch[lid * DIGEST_WORDS + j] ^=
                    scratch[(lid + s) * DIGEST_WORDS + j];
        VB_BARRIER();
    }

    if (lid == 0)
        for (int j = 0; j < DIGEST_WORDS; j++)
            partials[VB_GROUP_ID() * DIGEST_WORDS + j] = scratch[j];
}
