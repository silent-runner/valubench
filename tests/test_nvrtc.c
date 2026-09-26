/*
 * test_nvrtc.c -- the CUDA dialect must compile, with no GPU at all.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * CUDA kernels only execute on NVIDIA hardware, which no CI runner has. But
 * NVRTC compiles without a device, so the one thing that can silently rot --
 * the CUDA spelling of the shared kernel sources -- can be checked anywhere
 * libnvrtc can be loaded: from a CUDA toolkit, or from the nvidia-cuda-nvrtc
 * pip wheel with its directory on LD_LIBRARY_PATH.
 *
 * Every algorithm at every stream count, with the NVIDIA steers the backend
 * would compile in and with none, for the oldest and newest architectures
 * this NVRTC supports. The program text comes from the device layer itself,
 * so this compiles exactly what the backend would.
 *
 * Without libnvrtc it reports a skip and succeeds: an absent optional
 * toolchain is not a failure, as with the OpenCL checks.
 */

#define _GNU_SOURCE

#include "device.h"
#include "vb_cuda.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef vb_nvrtcResult (*nvrtc_fn_NumArchs)(int *);
typedef vb_nvrtcResult (*nvrtc_fn_Archs)(int *);

static struct {
    nvrtc_fn_Version         Version;
    nvrtc_fn_CreateProgram   CreateProgram;
    nvrtc_fn_DestroyProgram  DestroyProgram;
    nvrtc_fn_CompileProgram  CompileProgram;
    nvrtc_fn_GetSize         GetProgramLogSize;
    nvrtc_fn_GetData         GetProgramLog;
    nvrtc_fn_NumArchs        GetNumSupportedArchs;
    nvrtc_fn_Archs           GetSupportedArchs;
} nv;

static void *open_nvrtc(void)
{
    static const char *const names[] = {
        "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so",
    };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        void *h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (h)
            return h;
    }
    return NULL;
}

static int compile(const char *src, const char *name, const char **opts,
                   int n_opts)
{
    vb_nvrtcProgram p = NULL;
    if (nv.CreateProgram(&p, src, name, 0, NULL, NULL) != VB_NVRTC_SUCCESS)
        return -1;
    int rc = nv.CompileProgram(p, n_opts, opts) == VB_NVRTC_SUCCESS ? 0 : -1;
    if (rc != 0) {
        size_t n = 0;
        nv.GetProgramLogSize(p, &n);
        char *log = calloc(1, n + 1);
        if (log && n) {
            nv.GetProgramLog(p, log);
            fprintf(stderr, "%.600s\n", log);
        }
        free(log);
    }
    nv.DestroyProgram(&p);
    return rc;
}

int main(void)
{
    void *h = open_nvrtc();
    if (!h) {
        printf("  skip  nvrtc  (libnvrtc not found; install a CUDA toolkit or "
               "nvidia-cuda-nvrtc)\n");
        return 0;
    }
    *(void **) (&nv.Version)              = dlsym(h, "nvrtcVersion");
    *(void **) (&nv.CreateProgram)        = dlsym(h, "nvrtcCreateProgram");
    *(void **) (&nv.DestroyProgram)       = dlsym(h, "nvrtcDestroyProgram");
    *(void **) (&nv.CompileProgram)       = dlsym(h, "nvrtcCompileProgram");
    *(void **) (&nv.GetProgramLogSize)    = dlsym(h, "nvrtcGetProgramLogSize");
    *(void **) (&nv.GetProgramLog)        = dlsym(h, "nvrtcGetProgramLog");
    *(void **) (&nv.GetNumSupportedArchs) = dlsym(h, "nvrtcGetNumSupportedArchs");
    *(void **) (&nv.GetSupportedArchs)    = dlsym(h, "nvrtcGetSupportedArchs");
    if (!nv.Version || !nv.CreateProgram || !nv.CompileProgram ||
        !nv.DestroyProgram || !nv.GetProgramLogSize || !nv.GetProgramLog) {
        printf("  FAIL  nvrtc  libnvrtc is missing entry points\n");
        return 1;
    }

    int major = 0, minor = 0;
    nv.Version(&major, &minor);

    /* The oldest and newest architectures this NVRTC compiles for; without
       the query, one that every NVRTC since 12.0 supports. */
    int archs[2] = { 75, 75 }, n_arch = 1;
    int n = 0;
    if (nv.GetNumSupportedArchs && nv.GetSupportedArchs &&
        nv.GetNumSupportedArchs(&n) == VB_NVRTC_SUCCESS && n > 0) {
        int *all = calloc((size_t) n, sizeof *all);
        if (all && nv.GetSupportedArchs(all) == VB_NVRTC_SUCCESS) {
            archs[0] = all[0];
            archs[1] = all[n - 1];
            n_arch = archs[1] != archs[0] ? 2 : 1;
        }
        free(all);
    }

    static const vb_alg_id ALGS[] = { VB_ALG_MD5, VB_ALG_SHA1, VB_ALG_SHA512 };
    int checks = 0, failures = 0;

    for (size_t a = 0; a < sizeof ALGS / sizeof ALGS[0]; a++) {
        const char *entry = NULL;
        char *src = vb_dev_program_source(ALGS[a], VB_DIALECT_CUDA, &entry);
        if (!src) {
            printf("  FAIL  nvrtc  no CUDA program for algorithm %d\n",
                   (int) ALGS[a]);
            failures++;
            continue;
        }
        for (int neutral = 0; neutral <= 1; neutral++) {
            char defs[256], record[128];
            if (vb_device_steers(VB_VENDOR_NVIDIA, VB_DIALECT_CUDA, neutral,
                                 defs, sizeof defs, record,
                                 sizeof record) != 0) {
                printf("  FAIL  nvrtc  NVIDIA's steer set: %s\n", record);
                failures++;
                continue;
            }
            for (int ai = 0; ai < n_arch; ai++)
                for (unsigned s = 1; s <= 4; s++) {
                    char arch[48], streams[32], defs_copy[256];
                    snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d",
                             archs[ai]);
                    snprintf(streams, sizeof streams, "-DSTREAMS=%u", s);
                    snprintf(defs_copy, sizeof defs_copy, "%s", defs);
                    const char *opts[16] = { arch, "-DLANES=64", streams };
                    int k = 3;
                    for (char *t = strtok(defs_copy, " "); t && k < 16;
                         t = strtok(NULL, " "))
                        opts[k++] = t;
                    checks++;
                    if (compile(src, entry, opts, k) != 0) {
                        printf("  FAIL  nvrtc  %s s%u compute_%d %s\n", entry,
                               s, archs[ai], neutral ? "neutral" : "steered");
                        failures++;
                    }
                }
        }
        free(src);
    }

    if (failures == 0) {
        if (n_arch > 1)
            printf("  ok    nvrtc  (NVRTC %d.%d: %d compiles, compute_%d and "
                   "compute_%d)\n", major, minor, checks, archs[0], archs[1]);
        else
            printf("  ok    nvrtc  (NVRTC %d.%d: %d compiles, compute_%d)\n",
                   major, minor, checks, archs[0]);
    }
    return failures ? 1 : 0;
}
