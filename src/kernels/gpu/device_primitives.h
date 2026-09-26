/*
 * device_primitives.h -- how each hash primitive is spelled on the device.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Included after a dialect header and before an algorithm core. The cores use
 * only the macros at the bottom of this file -- VB_ROTL32, VB_CH32 and the
 * rest -- and never a builtin directly, so how a primitive becomes an
 * instruction is decided here, in one place, per primitive.
 *
 * STEERED BY HARDWARE, NOT BY API
 * ===============================
 *
 * The default spelling of every primitive is plain C, the form the
 * specifications write it in. A *steer* replaces it with a spelling aimed at
 * one vendor's hardware -- an NVIDIA funnel shift, an AMD alignbit -- and the
 * host chooses steers by the device's vendor, never by the API compiling it.
 * So CUDA and OpenCL on one NVIDIA card compile identical text, and a
 * difference between them is a difference between toolchains.
 *
 * This deliberately inverts the CPU side's rule (AGENTS.md: round functions
 * are per ISA). There, each instruction set gets its own round functions
 * because the instruction sets differ. Here the instruction set is the same
 * under every API, so one steering per target, the same under every API, is
 * what keeps the APIs comparable.
 *
 * Three rules keep steering honest:
 *
 *   - A steer earns its place by measurement: it matches or beats the plain
 *     spelling on its target, and the host's steer table cites the idiom
 *     probe (tools/idiom_probe.py) result that shows it.
 *   - The result records which steers were active, in the JSON and the CSV.
 *   - Builtins are preferred to inline assembly where both exist, since the
 *     optimiser can see through a builtin and cannot see through asm.
 *
 * `--primitives neutral` passes no steers at all, so every primitive is plain
 * C: what the code loses when nothing steers it.
 *
 * THE OPENCL BUILTIN SPELLING
 * ===========================
 *
 * The kernels were written with OpenCL's rotate() and bitselect() before this
 * file existed. That spelling is kept as VB_SPELL_CLB for OpenCL targets that
 * no probe has examined -- changing what an unmeasured compiler sees is a
 * regression risk with nothing measured to set against it. It exists in one
 * dialect only, so it must never be a steer on a target that a second API can
 * also reach; the host's table enforces that.
 *
 * Each steer is selected by defining VB_STEER_<PRIMITIVE> to a spelling id
 * below, as a -D at compile time. Undefined means plain.
 */

#define VB_SPELL_PLAIN   0   /* plain C, the specification's form */
#define VB_SPELL_CLB     1   /* OpenCL C builtins; OpenCL dialect only */
#define VB_SPELL_PTX     2   /* NVIDIA inline PTX; same text in every dialect */
#define VB_SPELL_AMDGCN  3   /* AMDGPU builtins; same text in every dialect */

#ifndef VB_STEER_ROTL32
#define VB_STEER_ROTL32 VB_SPELL_PLAIN
#endif
#ifndef VB_STEER_ROTR64
#define VB_STEER_ROTR64 VB_SPELL_PLAIN
#endif
#ifndef VB_STEER_CH
#define VB_STEER_CH VB_SPELL_PLAIN
#endif
#ifndef VB_STEER_MAJ
#define VB_STEER_MAJ VB_SPELL_PLAIN
#endif

#if !defined(VB_DIALECT_OPENCL) && \
    (VB_STEER_ROTL32 == VB_SPELL_CLB || VB_STEER_ROTR64 == VB_SPELL_CLB || \
     VB_STEER_CH == VB_SPELL_CLB || VB_STEER_MAJ == VB_SPELL_CLB)
#error "the OpenCL builtin spelling was steered into a non-OpenCL dialect"
#endif

/* ---- NVIDIA inline PTX --------------------------------------------------- */
/*
 * Functions rather than macros only because asm is a statement. The rotate
 * counts are literals at every call site, so the n >= 32 branch folds away.
 */
#if VB_STEER_ROTL32 == VB_SPELL_PTX
VB_INLINE vb_u32 vb_rotl32_ptx(vb_u32 x, vb_u32 n)
{
    vb_u32 r;
    asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(x), "r"(n));
    return r;
}
#endif

#if VB_STEER_ROTR64 == VB_SPELL_PTX
/* Two funnel shifts over the halves: {hi:lo} >> n takes its low word from
   shf.r(lo, hi) and its high word from shf.r(hi, lo). */
VB_INLINE vb_u64 vb_rotr64_ptx(vb_u64 x, vb_u32 n)
{
    vb_u32 lo, hi, rlo, rhi;
    vb_u64 r;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "l"(x));
    if (n >= 32) {
        vb_u32 t = lo;
        lo = hi;
        hi = t;
        n -= 32;
    }
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(rlo) : "r"(lo), "r"(hi), "r"(n));
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(rhi) : "r"(hi), "r"(lo), "r"(n));
    asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "r"(rlo), "r"(rhi));
    return r;
}
#endif

#if VB_STEER_CH == VB_SPELL_PTX || VB_STEER_MAJ == VB_SPELL_PTX
/* One LOP3 with the truth table as an immediate: 0xca is x ? y : z and 0xe8
   is the majority, over the operand masks 0xf0, 0xcc and 0xaa. */
VB_INLINE vb_u32 vb_lop3_0xca(vb_u32 a, vb_u32 b, vb_u32 c)
{
    vb_u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
VB_INLINE vb_u32 vb_lop3_0xe8(vb_u32 a, vb_u32 b, vb_u32 c)
{
    vb_u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
VB_INLINE vb_u64 vb_lop3_64_0xca(vb_u64 a, vb_u64 b, vb_u64 c)
{
    return ((vb_u64) vb_lop3_0xca((vb_u32) (a >> 32), (vb_u32) (b >> 32),
                                  (vb_u32) (c >> 32)) << 32)
         | vb_lop3_0xca((vb_u32) a, (vb_u32) b, (vb_u32) c);
}
VB_INLINE vb_u64 vb_lop3_64_0xe8(vb_u64 a, vb_u64 b, vb_u64 c)
{
    return ((vb_u64) vb_lop3_0xe8((vb_u32) (a >> 32), (vb_u32) (b >> 32),
                                  (vb_u32) (c >> 32)) << 32)
         | vb_lop3_0xe8((vb_u32) a, (vb_u32) b, (vb_u32) c);
}
#endif

/* ---- AMDGPU builtins ----------------------------------------------------- */
/*
 * alignbit(a, b, s) is the low word of {a:b} >> s, so a rotate right by s is
 * alignbit(x, x, s) and a 64-bit rotate is two of them over the halves.
 */
#if VB_STEER_ROTR64 == VB_SPELL_AMDGCN
VB_INLINE vb_u64 vb_rotr64_amdgcn(vb_u64 x, vb_u32 n)
{
    vb_u32 lo = (vb_u32) x, hi = (vb_u32) (x >> 32);
    if (n >= 32) {
        vb_u32 t = lo;
        lo = hi;
        hi = t;
        n -= 32;
    }
    vb_u32 rlo = __builtin_amdgcn_alignbit(hi, lo, n);
    vb_u32 rhi = __builtin_amdgcn_alignbit(lo, hi, n);
    return ((vb_u64) rhi << 32) | rlo;
}
#endif

/* ---- the primitives the cores use --------------------------------------- */

#if VB_STEER_ROTL32 == VB_SPELL_CLB
#  define VB_ROTL32(x, n) rotate((vb_u32) (x), (vb_u32) (n))
#elif VB_STEER_ROTL32 == VB_SPELL_PTX
#  define VB_ROTL32(x, n) vb_rotl32_ptx((x), (n))
#elif VB_STEER_ROTL32 == VB_SPELL_AMDGCN
#  define VB_ROTL32(x, n) __builtin_amdgcn_alignbit((x), (x), 32u - (n))
#else
#  define VB_ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#endif

/* Right, because every SHA-512 rotate in FIPS 180-4 is. */
#if VB_STEER_ROTR64 == VB_SPELL_CLB
#  define VB_ROTR64(x, n) rotate((vb_u64) (x), (vb_u64) (64 - (n)))
#elif VB_STEER_ROTR64 == VB_SPELL_PTX
#  define VB_ROTR64(x, n) vb_rotr64_ptx((x), (n))
#elif VB_STEER_ROTR64 == VB_SPELL_AMDGCN
#  define VB_ROTR64(x, n) vb_rotr64_amdgcn((x), (n))
#else
#  define VB_ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#endif

/*
 * Ch(x, y, z): y where x has a 1 bit, z where it has 0. MD5's F is Ch(x, y, z)
 * and its G is Ch(z, x, y). bitselect(a, b, c) takes b where c is 1, so the
 * builtin spelling is bitselect(z, y, x) -- one BFI_INT on AMD and one LOP3 on
 * NVIDIA, the GPU counterpart of the vpternlogd win on AVX-512.
 */
#if VB_STEER_CH == VB_SPELL_CLB
#  define VB_CH32(x, y, z) bitselect((z), (y), (x))
#  define VB_CH64(x, y, z) bitselect((z), (y), (x))
#elif VB_STEER_CH == VB_SPELL_PTX
#  define VB_CH32(x, y, z) vb_lop3_0xca((x), (y), (z))
#  define VB_CH64(x, y, z) vb_lop3_64_0xca((x), (y), (z))
#else
#  define VB_CH32(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#  define VB_CH64(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#endif

/* Maj(x, y, z), the majority of three. bitselect(x, y, x ^ z) is x where x and
   z agree and y where they do not, which is the majority. */
#if VB_STEER_MAJ == VB_SPELL_CLB
#  define VB_MAJ32(x, y, z) bitselect((x), (y), (x) ^ (z))
#  define VB_MAJ64(x, y, z) bitselect((x), (y), (x) ^ (z))
#elif VB_STEER_MAJ == VB_SPELL_PTX
#  define VB_MAJ32(x, y, z) vb_lop3_0xe8((x), (y), (z))
#  define VB_MAJ64(x, y, z) vb_lop3_64_0xe8((x), (y), (z))
#else
#  define VB_MAJ32(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#  define VB_MAJ64(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#endif
