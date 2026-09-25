/*
 * cpu_features.c -- runtime ISA detection via CPUID.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Checking the CPUID feature bit alone is not enough for AVX and AVX-512: the
 * OS must also have enabled the wider register state via XCR0, or the
 * instructions fault despite the CPU supporting them. Both are checked here.
 *
 * AArch64 asks the kernel rather than the CPU: getauxval(AT_HWCAP) reports what
 * the OS has enabled, which is the same question XCR0 answers on x86 and the
 * reason neither architecture can be probed by feature bit alone.
 */

#include "cpu_features.h"

#include <string.h>

#if defined(__x86_64__) || defined(__i386__)

#include <cpuid.h>
#include <stdint.h>

static uint64_t read_xcr0(void)
{
    uint32_t lo, hi;
    __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t) hi << 32) | lo;
}

static int os_saves_ymm(void)
{
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d))
        return 0;
    if (!(c & (1u << 27)))        /* OSXSAVE */
        return 0;
    /* XCR0 bit 1 = SSE state, bit 2 = AVX (YMM upper) state. */
    return (read_xcr0() & 0x6) == 0x6;
}

static int os_saves_zmm(void)
{
    if (!os_saves_ymm())
        return 0;
    /* bit 5 = opmask, bit 6 = ZMM_Hi256, bit 7 = Hi16_ZMM. */
    return (read_xcr0() & 0xe0) == 0xe0;
}

int vb_cpu_has_sse2(void)
{
#if defined(__x86_64__)
    return 1;                     /* guaranteed by the x86-64 baseline */
#else
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d))
        return 0;
    return (d & (1u << 26)) != 0;
#endif
}

int vb_cpu_has_avx2(void)
{
    unsigned a, b, c, d;
    if (!os_saves_ymm())
        return 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
        return 0;
    return (b & (1u << 5)) != 0;  /* AVX2 */
}

int vb_cpu_has_avx512f(void)
{
    unsigned a, b, c, d;
    if (!os_saves_zmm())
        return 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
        return 0;
    return (b & (1u << 16)) != 0; /* AVX512F */
}

/*
 * SHA-NI needs no XCR0 check: it operates on xmm registers, whose state the OS
 * has always saved. That is unlike AVX and AVX-512, where the CPUID bit alone
 * is not enough.
 */
int vb_cpu_has_sha_ni(void)
{
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
        return 0;
    return (b & (1u << 29)) != 0; /* SHA */
}

const char *vb_cpu_brand(void)
{
    static char brand[49];
    static int done = 0;
    unsigned regs[12];

    if (done)
        return brand[0] ? brand : "unknown";
    done = 1;

    unsigned a, b, c, d;
    if (!__get_cpuid(0x80000000u, &a, &b, &c, &d) || a < 0x80000004u)
        return "unknown";

    for (unsigned i = 0; i < 3; i++) {
        if (!__get_cpuid(0x80000002u + i, &regs[i * 4], &regs[i * 4 + 1],
                         &regs[i * 4 + 2], &regs[i * 4 + 3]))
            return "unknown";
    }

    memcpy(brand, regs, 48);
    brand[48] = '\0';

    /* Trim leading spaces some vendors pad with. */
    char *p = brand;
    while (*p == ' ')
        p++;
    if (p != brand)
        memmove(brand, p, strlen(p) + 1);

    return brand[0] ? brand : "unknown";
}

int vb_cpu_has_neon(void)     { return 0; }
int vb_cpu_has_sve(void)      { return 0; }
int vb_cpu_has_sve2(void)     { return 0; }
unsigned vb_sve_lanes32(void) { return 0; }
unsigned vb_sve_lanes64(void) { return 0; }

#elif defined(__aarch64__)

#include <stdio.h>

#if defined(__APPLE__)

/*
 * Darwin has no auxiliary vector. The hw.optional sysctls answer the same
 * question getauxval(AT_HWCAP) does -- what the OS has enabled, not merely
 * what the silicon implements -- so this stays a real query rather than a
 * hardcoded answer.
 */
#include <sys/sysctl.h>

static int darwin_hw_optional(const char *name, int if_absent)
{
    int val = 0;
    size_t len = sizeof val;

    /* An absent key means this kernel has never heard of the feature, which is
       not the same as the feature being off; the caller says which answer that
       should produce. */
    if (sysctlbyname(name, &val, &len, NULL, 0) != 0)
        return if_absent;
    return val != 0;
}

#else
#include <sys/auxv.h>
#endif

int vb_cpu_has_sse2(void)     { return 0; }
int vb_cpu_has_avx2(void)     { return 0; }
int vb_cpu_has_avx512f(void)  { return 0; }

/*
 * ARM has SHA-1 acceleration too, but its instructions decompose the rounds
 * differently from x86's and would need their own kernel template, which does
 * not exist. Reporting 0 keeps the SHA-NI kernels unregistered rather than
 * offering a path that cannot run.
 */
int vb_cpu_has_sha_ni(void)   { return 0; }

/*
 * Advanced SIMD is mandatory on AArch64 and every kernel here is built for it
 * unconditionally, so this is true by construction. HWCAP_ASIMD is still worth
 * consulting rather than returning 1: a kernel that has disabled SIMD for a
 * given process reports it here, and faulting on the first vector instruction
 * is a worse way to find out.
 */
int vb_cpu_has_neon(void)
{
#if defined(__APPLE__)
    return darwin_hw_optional("hw.optional.neon", 1);
#elif defined(HWCAP_ASIMD)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#else
    return 1;
#endif
}

/*
 * SVE is optional on AArch64 and SVE2 is an Armv9 addition, so both are
 * genuine run-time questions -- unlike Advanced SIMD, which is mandatory.
 * Graviton3 (Neoverse V1) has SVE at 256 bits and no SVE2; Graviton4
 * (Neoverse V2) has SVE2 at 128.
 *
 * A build whose headers predate SVE cannot define HWCAP_SVE, and a binary
 * built without the SVE kernels has nothing to gate anyway, so the fallback
 * is 0 rather than a guess.
 */
int vb_cpu_has_sve(void)
{
#if !VB_HAVE_SVE
    return 0;
#elif defined(__APPLE__)
    /* Apple silicon has no SVE through M4, but the toolchain compiles the
       kernels anyway, so this gate is what keeps them from being selected. */
    return darwin_hw_optional("hw.optional.arm.FEAT_SVE", 0);
#elif defined(HWCAP_SVE)
    return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
#else
    return 0;
#endif
}

int vb_cpu_has_sve2(void)
{
#if !VB_HAVE_SVE2
    return 0;
#elif defined(__APPLE__)
    /* FEAT_SVE2 implies FEAT_SVE, and Apple silicon has neither through M4;
       the sysctl keeps this a real query rather than a hardcoded 0. */
    return darwin_hw_optional("hw.optional.arm.FEAT_SVE2", 0);
#elif defined(HWCAP2_SVE2)
    /* FEAT_SVE2 implies FEAT_SVE, so requiring both changes nothing on real
       silicon -- but the pair is not always reported consistently. qemu's
       `-cpu max,sve=off` clears HWCAP_SVE and leaves HWCAP2_SVE2 set, and
       trusting the second bit alone puts an svcntw() on a machine that will
       not execute one. The SVE2 kernels use the SVE base instructions too,
       so the weaker of the two claims is the one to believe. */
    if (!vb_cpu_has_sve())
        return 0;
    return (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
#else
    return 0;
#endif
}

/*
 * There is no CPUID equivalent. MIDR_EL1 holds implementer and part numbers but
 * is privileged, so the kernel's /proc/cpuinfo summary is the portable source,
 * and on many ARM server parts it carries no model name field at all.
 */
const char *vb_cpu_brand(void)
{
    static char brand[128];
    static int done;

    if (done)
        return brand[0] ? brand : "unknown";
    done = 1;

#if defined(__APPLE__)

    /* Darwin publishes the marketing name directly -- "Apple M4 Pro" -- which
       is the one thing /proc/cpuinfo cannot supply on most ARM parts. */
    size_t len = sizeof brand;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0) != 0)
        brand[0] = '\0';

    return brand[0] ? brand : "unknown";

#else

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f)
        return "unknown";

    char line[256];
    while (fgets(line, sizeof line, f)) {
        const char *key = NULL;
        if (!strncmp(line, "model name", 10))
            key = "model name";
        else if (!strncmp(line, "CPU implementer", 15))
            key = "CPU implementer";
        if (!key)
            continue;

        char *colon = strchr(line, ':');
        if (!colon)
            continue;
        colon++;
        while (*colon == ' ' || *colon == '\t')
            colon++;
        colon[strcspn(colon, "\r\n")] = '\0';

        if (!strcmp(key, "model name")) {          /* the better answer; stop */
            snprintf(brand, sizeof brand, "%s", colon);
            break;
        }
        /* Fall back to the implementer/part codes, which are always present. */
        snprintf(brand, sizeof brand, "AArch64 implementer %s", colon);
    }
    fclose(f);

    return brand[0] ? brand : "unknown";

#endif
}

#else  /* neither x86 nor AArch64 */

int vb_cpu_has_sse2(void)     { return 0; }
int vb_cpu_has_avx2(void)     { return 0; }
int vb_cpu_has_avx512f(void)  { return 0; }
int vb_cpu_has_sha_ni(void)   { return 0; }
int vb_cpu_has_neon(void)     { return 0; }
int vb_cpu_has_sve(void)      { return 0; }
int vb_cpu_has_sve2(void)     { return 0; }
unsigned vb_sve_lanes32(void) { return 0; }
unsigned vb_sve_lanes64(void) { return 0; }
const char *vb_cpu_brand(void) { return "unknown"; }

#endif
