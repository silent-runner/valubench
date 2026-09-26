/*
 * registry.c -- the table of available kernels.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Every (algorithm, ISA, stream count) triple is an entry, and all of them are
 * expanded from the table in src/kernels/cpu/matrix.h rather than written out
 * here.
 * There is nothing to keep in sync: a kernel that exists in the matrix is
 * declared and registered automatically, and one that does not cannot be
 * half-added.
 *
 * The harness measures the available entries for the selected algorithm and
 * picks a winner by measurement rather than assumption -- the optimum moves
 * with microarchitecture, and guessing wrong costs more than the margin
 * the project calls significant.
 */

#include "valubench.h"
#include "cpu_features.h"
#include "device.h"
#include "opencl.h"
#include "kernels/cpu/matrix.h"

#include <pthread.h>
#include <stddef.h>

static int always(void) { return 1; }

/*
 * A device kernel is available if OpenCL loads and reports at least one device.
 * Both are normal to fail -- a machine with no GPU simply has no such kernels.
 */
static int have_opencl(void)
{
    static int cached = -1;
    if (cached < 0) {
        vb_ocl_device d[VB_OCL_MAX_DEVICES];
        cached = vb_ocl_devices(d, VB_OCL_MAX_DEVICES) > 0;
    }
    return cached;
}

/* ---- forward declarations, one per matrix cell -------------------------- */

#define VB_DECL_KERNEL(alg, isa, st, isaname, algid, avail, lanes, lfn)     \
    void VB_KSYM(alg, isa, st)(const void *corpus, uint64_t n_groups,       \
                               uint32_t blocks, uint32_t iterations,        \
                               uint64_t checksum[VB_MAX_DIGEST_WORDS]);

VB_FOR_EACH_KERNEL(VB_DECL_KERNEL)

/* ---- the table ---------------------------------------------------------- */

#define VB_ROW_KERNEL(alg, isa, st, isaname, algid, avail, lanes, lfn)      \
    { VB_KNAME(alg, isa, st), isaname, algid, lanes, st,                    \
      VB_KSYM(alg, isa, st), avail, 0, (lanes) == 0, -1 },

/*
 * Not const, because of the SVE rows below. Everything else about this table
 * is fixed at compile time.
 */
static vb_kernel kernels[] = {
    VB_FOR_EACH_KERNEL(VB_ROW_KERNEL)

    /*
     * Device rows, expanded from the same list src/opencl/backend.c builds its
     * program table from, so the two cannot disagree.
     */
#define VB_DEV_ROW(alg, ALG, algid, entry, st)                              \
    { VB_KNAME(alg, ocl, st), "OpenCL", algid, VB_OCL_LANES, st,            \
      NULL, have_opencl, 1, 0, VB_BACKEND_OPENCL },
#define VB_DEV_ALG(alg, ALG, algid, entry)      \
    VB_DEV_ROW(alg, ALG, algid, entry, 1)       \
    VB_DEV_ROW(alg, ALG, algid, entry, 2)       \
    VB_DEV_ROW(alg, ALG, algid, entry, 3)       \
    VB_DEV_ROW(alg, ALG, algid, entry, 4)

    VB_FOR_EACH_DEVICE_ALG(VB_DEV_ALG)

#undef VB_DEV_ALG
#undef VB_DEV_ROW
};

/*
 * Lane counts that are only known at run time.
 *
 * A vector-length-agnostic ISA registers `lanes = 0` and a function here; every
 * fixed-width row has NULL and is untouched. The table is filled in once, on
 * the first call to vb_kernels(), so that by the time anything reads
 * `k->lanes` it is correct -- which means no caller changes, and there are
 * about a dozen of them.
 *
 * This assumes the vector length is fixed for the life of the process. It is,
 * unless something calls prctl(PR_SVE_SET_VL), which nothing here does; a
 * per-thread vector length would break this and much else besides.
 */
#define VB_LANES_FN(alg, isa, st, isaname, algid, avail, lanes, lfn) lfn,

static unsigned (*const lanes_fn[])(void) = {
    VB_FOR_EACH_KERNEL(VB_LANES_FN)
};

#undef VB_LANES_FN

static void resolve_lanes(void)
{
    size_t n = sizeof lanes_fn / sizeof lanes_fn[0];
    for (size_t i = 0; i < n; i++)
        /* available() first, and not as a courtesy: lanes_fn is compiled with
           the ISA's -march flag, so calling it on a CPU without that ISA is an
           illegal instruction, not a wrong answer. A kernel that cannot run
           here keeps lanes = 0, which vb_batch_divides() reads as "no group
           size" rather than dividing by it. */
        if (lanes_fn[i] && kernels[i].available())
            kernels[i].lanes = lanes_fn[i]();
}

const vb_kernel *vb_kernels(size_t *count)
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, resolve_lanes);

    *count = sizeof kernels / sizeof kernels[0];
    return kernels;
}
