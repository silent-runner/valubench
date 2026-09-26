/*
 * test_device_layer.c -- the device layer's pure parts, with no device.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Three things every device run leans on that need no hardware to be wrong:
 * how a slice of the corpus is cut into chunks, how many chunks the overlap
 * pipeline takes when it is not told, and the translation that lets the CUDA
 * driver run NVIDIA OpenCL's PTX. No CI runner has a GPU, so this is the only
 * place CI sees them.
 *
 * The translation reads a file named on the command line, so it is checked
 * with hostile input as well as real input. Its output buffer once had a fixed
 * margin that enough "%envreg6" tokens overran; under the sanitizer job that
 * is a failure rather than a quiet corruption.
 */

#include "device.h"
#include "cuda_ptx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;

static void expect(int ok, const char *what)
{
    checks++;
    if (!ok) {
        failures++;
        printf("  FAIL  %s\n", what);
    }
}

/* ---- slicing ------------------------------------------------------------- */

/* The pieces must tile the whole, in order, differ by at most one group, and
   put the largest first -- set_overlap sizes both chunk buffers from piece 0. */
static void check_slices(uint64_t total, int n)
{
    char what[96];
    uint64_t next = 0, lo = UINT64_MAX, hi = 0, first0 = 0, count0 = 0;
    int ok = 1;

    for (int i = 0; i < n; i++) {
        uint64_t first, count;
        vb_dev_slice(total, n, i, &first, &count);
        if (i == 0) {
            first0 = first;
            count0 = count;
        }
        if (first != next)
            ok = 0;
        next = first + count;
        if (count < lo) lo = count;
        if (count > hi) hi = count;
    }
    snprintf(what, sizeof what, "slice %llu groups into %d",
             (unsigned long long) total, n);
    expect(ok && next == total && hi - lo <= 1 && first0 == 0 &&
           count0 == hi, what);
}

/* ---- the chunk count overlap chooses ------------------------------------ */

static void check_chunks(const char *what, uint64_t n_groups,
                         uint64_t group_bytes, uint64_t global_mem,
                         unsigned want)
{
    unsigned got = vb_dev_auto_chunks(n_groups, group_bytes, global_mem);
    if (got != want) {
        char msg[160];
        snprintf(msg, sizeof msg, "%s: %u chunks, want %u", what, got, want);
        expect(0, msg);
        return;
    }
    expect(1, what);
}

/* ---- OpenCL's PTX for the CUDA driver ----------------------------------- */

static char *translate(const char *in, int *abi, char *err, size_t errn)
{
    err[0] = '\0';
    return vb_ptx_from_opencl(in, abi, err, errn);
}

static void check_translation(void)
{
    char err[256];
    int abi = -1;

    /* The shape NVIDIA's OpenCL compiler emits: qualified parameters, a
       __local argument, and the three launch-environment registers. */
    const char *ocl =
        ".version 8.7\n.target sm_120\n.address_size 64\n"
        ".entry k(\n"
        "\t.param .u64 .ptr .global .align 4 k_param_0,\n"
        "\t.param .u64 .ptr .shared .align 16 k_param_1\n"
        ")\n{\n"
        "\tmov.b32 %r1, %envreg6;\n"
        "\tmov.u32 %r2, %envreg3;\n"
        "\tadd.s32 %r3, %envreg0, %r2;\n"
        "}\n";
    char *out = translate(ocl, &abi, err, sizeof err);
    expect(out != NULL, "translate OpenCL PTX");
    if (out) {
        expect(strstr(out, ".ptr") == NULL, "parameter qualifiers dropped");
        expect(strstr(out, "\t.param .u64 k_param_0,\n") != NULL,
               "parameter kept without its qualifiers");
        expect(strstr(out, "mov.u32 %r1, %nctaid.x;") != NULL,
               "group count read as %nctaid.x, with the .u32 spelling");
        expect(strstr(out, "mov.u32 %r2, 0;") != NULL,
               "global offset read as zero");
        expect(strstr(out, "add.s32 %r3, 0, %r2;") != NULL,
               "group offset read as zero");
        expect(strstr(out, "%envreg") == NULL, "no launch register left");
        free(out);
    }
    expect(abi == 1, "a __local argument selects the OpenCL launch ABI");

    out = translate(".entry k(\n\t.param .u64 .ptr .global .align 4 p\n)\n",
                    &abi, err, sizeof err);
    expect(out != NULL && abi == 0, "no __local argument, no OpenCL ABI");
    free(out);

    /* A register with no CUDA meaning is refused, not guessed. */
    out = translate("\tmov.b32 %r1, %envreg5;\n", &abi, err, sizeof err);
    expect(out == NULL && strstr(err, "%envreg5") != NULL,
           "an unmapped launch register is refused, by name");
    free(out);

    /* A qualifier with no .align close behind is not the pattern; it is
       copied, not cut at some later .align. */
    const char *odd = " .ptr .global p\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n .align 8";
    out = translate(odd, &abi, err, sizeof err);
    expect(out != NULL && strcmp(out, odd) == 0,
           "a distant .align does not swallow what lies between");
    free(out);

    /* The .u32 fix-up looks only at whole, already-written candidates: a
       line ending in a prefix of "mov.b32" must come through as it was. */
    out = translate("mov.b3%envreg6", &abi, err, sizeof err);
    expect(out != NULL && strcmp(out, "mov.b3%nctaid.x") == 0,
           "a partial mov.b32 is not rewritten");
    free(out);
    out = translate("mov.b32 %r9, %envreg6", &abi, err, sizeof err);
    expect(out != NULL && strcmp(out, "mov.u32 %r9, %nctaid.x") == 0,
           "a mov.b32 at the very start is rewritten");
    free(out);

    /* The worst case for growth: nothing but %envreg6, each one byte longer
       translated. Far more of them than any fixed margin would allow. */
    const size_t reps = 200000;
    char *big = malloc(reps * 8 + 1);
    if (big) {
        for (size_t i = 0; i < reps; i++)
            memcpy(big + i * 8, "%envreg6", 8);
        big[reps * 8] = '\0';
        out = translate(big, &abi, err, sizeof err);
        int ok = out != NULL && strlen(out) == reps * 9;
        for (size_t i = 0; ok && i < reps; i++)
            ok = memcmp(out + i * 9, "%nctaid.x", 9) == 0;
        expect(ok, "200,000 %envreg6: every one translated, nothing overrun");
        free(out);
        free(big);
    }
}

int main(void)
{
    printf("Device layer, no device\n");

    static const uint64_t totals[] = { 1, 2, 7, 64, 1000, 32760, 32768 };
    for (size_t t = 0; t < sizeof totals / sizeof totals[0]; t++)
        for (int n = 1; n <= 17 && (uint64_t) n <= totals[t]; n++)
            check_slices(totals[t], n);

    const uint64_t GiB = 1ull << 30, MiB = 1ull << 20;
    check_chunks("256 MiB on 16 GiB: one chunk", 32768, 8192, 16 * GiB, 1);
    check_chunks("3 GiB on 16 GiB: split in two", 3 * 131072, 8192,
                 16 * GiB, 2);
    check_chunks("256 MiB on 1 GiB: two, exactly at the budget", 32768, 8192,
                 GiB, 2);
    check_chunks("memory unknown: one chunk", 32768, 8192, 0, 1);
    check_chunks("never more chunks than groups", 3, GiB, GiB, 3);
    check_chunks("never more than 256", 100000, MiB, GiB, 256);

    check_translation();

    printf("  %s    device-layer  (%d checks)\n", failures ? "FAIL" : "ok",
           checks);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
