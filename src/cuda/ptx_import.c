/*
 * ptx_import.c -- NVIDIA OpenCL's PTX, translated for the CUDA driver.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * NVIDIA's OpenCL compiler emits PTX, and the CUDA driver can run it once
 * OpenCL's calling convention is translated -- which makes a 2x2 of frontend
 * against runtime, and lets Nsight Compute, which sees only CUDA, profile the
 * code OpenCL's compiler produced. The translation touches no hash code:
 *
 *   - kernel parameters carry OpenCL's state-space qualifiers (".ptr .global
 *     .align 4"), which the CUDA loader rejects as an invalid image; they are
 *     annotations, and the pointers are used as they are either way;
 *   - OpenCL reads its launch environment from %envreg registers the CUDA
 *     runtime never sets: %envreg0 and %envreg3 are the group and global
 *     offsets, zero for any launch here, and %envreg6 is the group count,
 *     which CUDA calls %nctaid.x. Any other %envreg is refused, not guessed;
 *   - the __local scratch argument stays, and the launch passes it the
 *     address of the dynamic shared region.
 *
 * Checked on an RTX PRO 2000 against NVRTC's kernel on the same data before
 * it was trusted, and the checksum gate checks it on every run after.
 *
 * The input is a file named on the command line, so nothing about it is
 * trusted: the output is sized for the worst case the rewriting can reach,
 * not the typical one.
 */

#include "cuda_ptx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *vb_ptx_from_opencl(const char *in, int *ocl_abi, char *err, size_t errn)
{
    /* Only "%envreg6" grows, by one byte, and it is eight bytes long; every
       other rewrite shrinks. So the output is at most n + n/8, plus the
       terminator. A fixed margin, as this used to have, overflows on input
       with enough of them. */
    size_t n = strlen(in);
    char *out = malloc(n + n / 8 + 1);
    if (!out) {
        snprintf(err, errn, "out of memory");
        return NULL;
    }
    size_t o = 0;
    *ocl_abi = strstr(in, ".ptr .shared") != NULL;

    for (const char *p = in; *p; ) {
        if (!strncmp(p, " .ptr .", 7)) {
            /* " .ptr .<space> .align <n>" -- drop it. */
            const char *q = strstr(p, ".align ");
            if (q && q - p < 32) {
                q += 7;
                while (*q >= '0' && *q <= '9')
                    q++;
                p = q;
                continue;
            }
        }
        if (!strncmp(p, "%envreg", 7)) {
            int reg = atoi(p + 7);
            const char *rest = p + 7;
            while (*rest >= '0' && *rest <= '9')
                rest++;
            const char *with = NULL;
            if (reg == 0 || reg == 3)
                with = "0";
            else if (reg == 6)
                with = "%nctaid.x";
            if (!with) {
                snprintf(err, errn, "the PTX reads %%envreg%d, which has no "
                         "CUDA equivalent here", reg);
                free(out);
                return NULL;
            }
            /* mov.b32 from a special register wants the .u32 spelling. The
               mnemonic is in this instruction, so the search goes back no
               further than the last ';' or newline, and never more than a
               line's worth: unbounded, it made input with no line breaks
               quadratic. Only whole candidates already written count --
               out[o] onwards is not written yet. */
            if (reg == 6) {
                char *mv = NULL;
                for (size_t k = o; k > 0 && o - k < 256 && out[k - 1] != '\n' &&
                                   out[k - 1] != ';'; k--)
                    if (o - (k - 1) >= 7 && !strncmp(out + k - 1, "mov.b32", 7))
                        mv = out + k - 1;
                if (mv)
                    memcpy(mv, "mov.u32", 7);
            }
            size_t wl = strlen(with);
            memcpy(out + o, with, wl);
            o += wl;
            p = rest;
            continue;
        }
        out[o++] = *p++;
    }
    out[o] = '\0';
    return out;
}
