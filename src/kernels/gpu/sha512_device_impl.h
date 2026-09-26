/*
 * sha512_device_impl.h -- the SHA-512 device kernel, in no particular dialect.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Same construction as md5_device_impl.h and sha1_device_impl.h -- assembled
 * after a dialect header and device_primitives.h, LANES and STREAMS arriving
 * as -D, and the constant tables and all 80 steps written out below from FIPS
 * 180-4. The work decomposition is identical to the CPU template, so the XOR
 * checksum matches it bit for bit.
 *
 * This is the one that stresses the device rather than the harness, in two
 * ways that are worth stating before the code.
 *
 * SIXTY-FOUR-BIT INTEGER ARITHMETIC. Every operation here is on 64-bit words. That
 * is core in both dialects and always compiles, but on most consumer GPUs the hardware ALUs
 * are 32-bit and the compiler emulates 64-bit work: an add becomes an add plus a
 * carry, and a 64-bit rotate becomes a funnel shift built from several 32-bit
 * ops. Whatever this kernel measures is therefore as much a statement about the
 * device's 64-bit emulation as about SHA-512, which is precisely why it is
 * interesting -- an accelerator that looks strong on MD5 can be structurally
 * poor here, and a buyer should know that before committing.
 *
 * REGISTER PRESSURE, WORSE THAN SHA-1'S. SHA-1 needed sixteen 32-bit schedule
 * words live per stream; SHA-512 needs sixteen 64-bit ones, plus eight state
 * words, plus eight of chaining state, plus eight held for the digest feedback.
 * That is forty 64-bit values -- eighty 32-bit registers -- per stream before
 * any temporaries. `sha1/ocl` already fell off a 10x cliff between three streams
 * and four for exactly this reason -- a device saturates on its own work
 * decomposition long before four streams; this kernel should
 * be expected to fall off it earlier. The harness sweeps and measures rather
 * than assuming, so where the cliff lands is a result, not a problem.
 */

/* Every rotate in FIPS 180-4's SHA-512 is to the right, and `n` is always a
   literal. How it becomes instructions is device_primitives.h's business --
   on 32-bit device ALUs it is the operation that matters most here. */
#define ROTR(x, n) VB_ROTR64((x), (n))

/* Ch and Maj, spelled per target in device_primitives.h. */
#define S5CH(x, y, z)  VB_CH64((x), (y), (z))
#define S5MAJ(x, y, z) VB_MAJ64((x), (y), (z))

/* FIPS 180-4 section 4.1.3. */
#define BSIG0(x) (ROTR(x, 28) ^ ROTR(x, 34) ^ ROTR(x, 39))
#define BSIG1(x) (ROTR(x, 14) ^ ROTR(x, 18) ^ ROTR(x, 41))
#define SSIG0(x) (ROTR(x,  1) ^ ROTR(x,  8) ^ ((x) >> 7))
#define SSIG1(x) (ROTR(x, 19) ^ ROTR(x, 61) ^ ((x) >> 6))

/*
 * In-place round. The eight state variables shift one position per step, so the
 * host generator rotates the argument names rather than emitting copies: the
 * new E lands in the variable named `d` and the new A in the one named `h`,
 * which is exactly the rename the specification's assignments describe.
 */
#define STEP(k, a, b, c, d, e, f, g, h, t, KC)                  \
    {                                                           \
        vb_u64 t1 = h[k] + BSIG1(e[k]) + S5CH(e[k], f[k], g[k]) \
                 + KC + w[k][(t) & 15];                         \
        vb_u64 t2 = BSIG0(a[k]) + S5MAJ(a[k], b[k], c[k]);      \
        d[k] += t1;                                             \
        h[k] = t1 + t2;                                         \
    }

/* W[t] = SSIG1(W[t-2]) + W[t-7] + SSIG0(W[t-15]) + W[t-16], with the indices
   reduced into the rolling window -- w[t & 15] still holds W[t-16] when this
   runs, so the update accumulates in place. Unlike SHA-1's, this recurrence is
   arithmetic, so the schedule carries real adds as well as rotates. */
#define EXP1(k, t)                                              \
    w[k][(t) & 15] += SSIG1(w[k][((t) -  2) & 15])              \
                    + w[k][((t) -  7) & 15]                     \
                    + SSIG0(w[k][((t) - 15) & 15]);

/* Streams are expanded by macro, never by a loop -- if a loop failed to unroll
   the kernel would collapse to one dependency chain and silently under-report
   the device. Same reasoning as the CPU template. */
#if STREAMS == 1
#define EACH(a,b,c,d,e,f,g,h,t,KC) STEP(0,a,b,c,d,e,f,g,h,t,KC)
#define EXPAND(t) EXP1(0,t)
#elif STREAMS == 2
#define EACH(a,b,c,d,e,f,g,h,t,KC) STEP(0,a,b,c,d,e,f,g,h,t,KC) STEP(1,a,b,c,d,e,f,g,h,t,KC)
#define EXPAND(t) EXP1(0,t) EXP1(1,t)
#elif STREAMS == 3
#define EACH(a,b,c,d,e,f,g,h,t,KC) STEP(0,a,b,c,d,e,f,g,h,t,KC) STEP(1,a,b,c,d,e,f,g,h,t,KC) STEP(2,a,b,c,d,e,f,g,h,t,KC)
#define EXPAND(t) EXP1(0,t) EXP1(1,t) EXP1(2,t)
#elif STREAMS == 4
#define EACH(a,b,c,d,e,f,g,h,t,KC) STEP(0,a,b,c,d,e,f,g,h,t,KC) STEP(1,a,b,c,d,e,f,g,h,t,KC) STEP(2,a,b,c,d,e,f,g,h,t,KC) STEP(3,a,b,c,d,e,f,g,h,t,KC)
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

#define DIGEST_WORDS 8

/*
 * The eighty round constants and the initial state, FIPS 180-4 sections 4.2.3
 * and 5.3.5 -- the first 64 bits of the fractional parts of the cube roots of
 * the first eighty primes, and of the square roots of the first eight. The CPU
 * path holds the same values in include/sha512_const.h; `make check` validates
 * every device kernel against the scalar reference, so the two cannot drift
 * apart unnoticed.
 */
VB_CONST_TABLE vb_u64 K[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL,
};
VB_CONST_TABLE vb_u64 IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL,
};

VB_KERNEL vb_sha512(VB_GLOBAL const vb_u64 *corpus,
                    const vb_u32 blocks,
                    const vb_u32 iterations,
                    const vb_u64 n_groups,
                    const vb_u32 repeats,
                    VB_GLOBAL vb_u64 *partials
                    VB_SCRATCH_PARAM(vb_u64, scratch))
{
    VB_SCRATCH_DECL(vb_u64, scratch)

    const size_t gid  = VB_GLOBAL_ID();
    const size_t lid  = VB_LOCAL_ID();
    const size_t lane = gid % LANES;

    /* Sixteen words per block holds whatever the word width -- SHA-512's block
       is 1024 bits because its words are 64. See include/algorithm.h. */
    const size_t slot_words  = (size_t)blocks * 16 * LANES;
    const size_t block_words = 16 * LANES;

    const size_t g_stride = VB_GLOBAL_SIZE() / LANES;

    vb_u64 acc[DIGEST_WORDS];
    for (int j = 0; j < DIGEST_WORDS; j++)
        acc[j] = 0;

    /* Sweep the corpus `repeats` times, always an odd count so XORing every
       pass leaves the single-pass checksum intact. See md5.cl. */
    for (vb_u32 rep = 0; rep < repeats; rep++)
    for (size_t g = gid / LANES; g < n_groups; g += g_stride) {
        VB_GLOBAL const vb_u64 *slot[STREAMS];
        VB_GLOBAL const vb_u64 *wp[STREAMS];
        vb_u64 w[STREAMS][16];
        vb_u64 fb[STREAMS][DIGEST_WORDS];
        vb_u64 h[STREAMS][DIGEST_WORDS];
        vb_u64 A[STREAMS], B[STREAMS], C[STREAMS], D[STREAMS];
        vb_u64 E[STREAMS], F[STREAMS], G[STREAMS], H[STREAMS];

        for (int k = 0; k < STREAMS; k++) {
            slot[k] = corpus + (g * STREAMS + k) * slot_words + lane;
            for (int j = 0; j < DIGEST_WORDS; j++)
                fb[k][j] = slot[k][j * LANES];
        }

        for (vb_u32 it = 0; it < iterations; it++) {
            for (int k = 0; k < STREAMS; k++) {
                h[k][0] = IV[0]; h[k][1] = IV[1];
                h[k][2] = IV[2]; h[k][3] = IV[3];
                h[k][4] = IV[4]; h[k][5] = IV[5];
                h[k][6] = IV[6]; h[k][7] = IV[7];
            }

            for (vb_u32 b = 0; b < blocks; b++) {
                for (int k = 0; k < STREAMS; k++) {
                    wp[k] = slot[k] + (size_t)b * block_words;
                    LOADW(k)

                    /* Block 0 carries the previous digest over the head of the
                       message; every other block is corpus data unchanged. A
                       SHA-512 digest is eight words -- exactly half a block. */
                    if (b == 0) {
                        w[k][0] = fb[k][0]; w[k][1] = fb[k][1];
                        w[k][2] = fb[k][2]; w[k][3] = fb[k][3];
                        w[k][4] = fb[k][4]; w[k][5] = fb[k][5];
                        w[k][6] = fb[k][6]; w[k][7] = fb[k][7];
                    }

                    A[k] = h[k][0]; B[k] = h[k][1];
                    C[k] = h[k][2]; D[k] = h[k][3];
                    E[k] = h[k][4]; F[k] = h[k][5];
                    G[k] = h[k][6]; H[k] = h[k][7];
                }

                EACH(A,B,C,D,E,F,G,H, 0, K[0])
                EACH(H,A,B,C,D,E,F,G, 1, K[1])
                EACH(G,H,A,B,C,D,E,F, 2, K[2])
                EACH(F,G,H,A,B,C,D,E, 3, K[3])
                EACH(E,F,G,H,A,B,C,D, 4, K[4])
                EACH(D,E,F,G,H,A,B,C, 5, K[5])
                EACH(C,D,E,F,G,H,A,B, 6, K[6])
                EACH(B,C,D,E,F,G,H,A, 7, K[7])
                EACH(A,B,C,D,E,F,G,H, 8, K[8])
                EACH(H,A,B,C,D,E,F,G, 9, K[9])
                EACH(G,H,A,B,C,D,E,F, 10, K[10])
                EACH(F,G,H,A,B,C,D,E, 11, K[11])
                EACH(E,F,G,H,A,B,C,D, 12, K[12])
                EACH(D,E,F,G,H,A,B,C, 13, K[13])
                EACH(C,D,E,F,G,H,A,B, 14, K[14])
                EACH(B,C,D,E,F,G,H,A, 15, K[15])
                EXPAND(16)
                EACH(A,B,C,D,E,F,G,H, 16, K[16])
                EXPAND(17)
                EACH(H,A,B,C,D,E,F,G, 17, K[17])
                EXPAND(18)
                EACH(G,H,A,B,C,D,E,F, 18, K[18])
                EXPAND(19)
                EACH(F,G,H,A,B,C,D,E, 19, K[19])
                EXPAND(20)
                EACH(E,F,G,H,A,B,C,D, 20, K[20])
                EXPAND(21)
                EACH(D,E,F,G,H,A,B,C, 21, K[21])
                EXPAND(22)
                EACH(C,D,E,F,G,H,A,B, 22, K[22])
                EXPAND(23)
                EACH(B,C,D,E,F,G,H,A, 23, K[23])
                EXPAND(24)
                EACH(A,B,C,D,E,F,G,H, 24, K[24])
                EXPAND(25)
                EACH(H,A,B,C,D,E,F,G, 25, K[25])
                EXPAND(26)
                EACH(G,H,A,B,C,D,E,F, 26, K[26])
                EXPAND(27)
                EACH(F,G,H,A,B,C,D,E, 27, K[27])
                EXPAND(28)
                EACH(E,F,G,H,A,B,C,D, 28, K[28])
                EXPAND(29)
                EACH(D,E,F,G,H,A,B,C, 29, K[29])
                EXPAND(30)
                EACH(C,D,E,F,G,H,A,B, 30, K[30])
                EXPAND(31)
                EACH(B,C,D,E,F,G,H,A, 31, K[31])
                EXPAND(32)
                EACH(A,B,C,D,E,F,G,H, 32, K[32])
                EXPAND(33)
                EACH(H,A,B,C,D,E,F,G, 33, K[33])
                EXPAND(34)
                EACH(G,H,A,B,C,D,E,F, 34, K[34])
                EXPAND(35)
                EACH(F,G,H,A,B,C,D,E, 35, K[35])
                EXPAND(36)
                EACH(E,F,G,H,A,B,C,D, 36, K[36])
                EXPAND(37)
                EACH(D,E,F,G,H,A,B,C, 37, K[37])
                EXPAND(38)
                EACH(C,D,E,F,G,H,A,B, 38, K[38])
                EXPAND(39)
                EACH(B,C,D,E,F,G,H,A, 39, K[39])
                EXPAND(40)
                EACH(A,B,C,D,E,F,G,H, 40, K[40])
                EXPAND(41)
                EACH(H,A,B,C,D,E,F,G, 41, K[41])
                EXPAND(42)
                EACH(G,H,A,B,C,D,E,F, 42, K[42])
                EXPAND(43)
                EACH(F,G,H,A,B,C,D,E, 43, K[43])
                EXPAND(44)
                EACH(E,F,G,H,A,B,C,D, 44, K[44])
                EXPAND(45)
                EACH(D,E,F,G,H,A,B,C, 45, K[45])
                EXPAND(46)
                EACH(C,D,E,F,G,H,A,B, 46, K[46])
                EXPAND(47)
                EACH(B,C,D,E,F,G,H,A, 47, K[47])
                EXPAND(48)
                EACH(A,B,C,D,E,F,G,H, 48, K[48])
                EXPAND(49)
                EACH(H,A,B,C,D,E,F,G, 49, K[49])
                EXPAND(50)
                EACH(G,H,A,B,C,D,E,F, 50, K[50])
                EXPAND(51)
                EACH(F,G,H,A,B,C,D,E, 51, K[51])
                EXPAND(52)
                EACH(E,F,G,H,A,B,C,D, 52, K[52])
                EXPAND(53)
                EACH(D,E,F,G,H,A,B,C, 53, K[53])
                EXPAND(54)
                EACH(C,D,E,F,G,H,A,B, 54, K[54])
                EXPAND(55)
                EACH(B,C,D,E,F,G,H,A, 55, K[55])
                EXPAND(56)
                EACH(A,B,C,D,E,F,G,H, 56, K[56])
                EXPAND(57)
                EACH(H,A,B,C,D,E,F,G, 57, K[57])
                EXPAND(58)
                EACH(G,H,A,B,C,D,E,F, 58, K[58])
                EXPAND(59)
                EACH(F,G,H,A,B,C,D,E, 59, K[59])
                EXPAND(60)
                EACH(E,F,G,H,A,B,C,D, 60, K[60])
                EXPAND(61)
                EACH(D,E,F,G,H,A,B,C, 61, K[61])
                EXPAND(62)
                EACH(C,D,E,F,G,H,A,B, 62, K[62])
                EXPAND(63)
                EACH(B,C,D,E,F,G,H,A, 63, K[63])
                EXPAND(64)
                EACH(A,B,C,D,E,F,G,H, 64, K[64])
                EXPAND(65)
                EACH(H,A,B,C,D,E,F,G, 65, K[65])
                EXPAND(66)
                EACH(G,H,A,B,C,D,E,F, 66, K[66])
                EXPAND(67)
                EACH(F,G,H,A,B,C,D,E, 67, K[67])
                EXPAND(68)
                EACH(E,F,G,H,A,B,C,D, 68, K[68])
                EXPAND(69)
                EACH(D,E,F,G,H,A,B,C, 69, K[69])
                EXPAND(70)
                EACH(C,D,E,F,G,H,A,B, 70, K[70])
                EXPAND(71)
                EACH(B,C,D,E,F,G,H,A, 71, K[71])
                EXPAND(72)
                EACH(A,B,C,D,E,F,G,H, 72, K[72])
                EXPAND(73)
                EACH(H,A,B,C,D,E,F,G, 73, K[73])
                EXPAND(74)
                EACH(G,H,A,B,C,D,E,F, 74, K[74])
                EXPAND(75)
                EACH(F,G,H,A,B,C,D,E, 75, K[75])
                EXPAND(76)
                EACH(E,F,G,H,A,B,C,D, 76, K[76])
                EXPAND(77)
                EACH(D,E,F,G,H,A,B,C, 77, K[77])
                EXPAND(78)
                EACH(C,D,E,F,G,H,A,B, 78, K[78])
                EXPAND(79)
                EACH(B,C,D,E,F,G,H,A, 79, K[79])


                for (int k = 0; k < STREAMS; k++) {
                    h[k][0] += A[k]; h[k][1] += B[k];
                    h[k][2] += C[k]; h[k][3] += D[k];
                    h[k][4] += E[k]; h[k][5] += F[k];
                    h[k][6] += G[k]; h[k][7] += H[k];
                }
            }

            for (int k = 0; k < STREAMS; k++)
                for (int j = 0; j < DIGEST_WORDS; j++)
                    fb[k][j] = h[k][j];
        }

        for (int k = 0; k < STREAMS; k++)
            for (int j = 0; j < DIGEST_WORDS; j++)
                acc[j] ^= h[k][j];
    }

    /* Reduce within the work-group so only one digest per group crosses the
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
