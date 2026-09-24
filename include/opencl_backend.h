/*
 * opencl_backend.h -- a built, uploaded, ready-to-launch OpenCL device context.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 */

#ifndef VALUBENCH_OPENCL_BACKEND_H
#define VALUBENCH_OPENCL_BACKEND_H

#include "valubench.h"
#include "opencl.h"

#include <stdarg.h>
#include <stdint.h>

typedef struct {
    vb_ocl_device dev;

    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    cl_mem           d_corpus;
    cl_mem           d_partial;
    cl_event         event;      /* in-flight launch, for profiling */
    cl_event         xfer_event;  /* in-flight upload, streaming mode only */

    unsigned  lanes;        /* corpus interleave width */
    /* Digest shape of the work-group partial: 4x4 md5, 5x4 sha1, 8x8 sha512.
       Sizes the readback, the work-group scratch and the final fold. */
    unsigned  partial_words;
    unsigned  partial_word_bytes;
    unsigned  streams;      /* messages each work-item hashes per group */
    uint32_t  blocks;       /* per message */
    uint64_t  first_group;  /* this device's slice within the corpus */
    uint64_t  n_groups;     /* groups in this device's slice */
    uint64_t  n_messages;   /* messages in this device's slice */

    /*
     * Launch geometry, chosen by measurement in vb_ocl_ctx_init and independent
     * of the corpus: each work-item strides over as many groups as needed. This
     * is what lets the device be saturated regardless of --working-set-kb.
     */
    size_t    global_size;
    size_t    local_size;

    /*
     * Sweeps of the corpus per launch, always odd. Amplifies work so a small
     * (cache-resident) working set can still saturate the device, and amortises
     * launch overhead. Multiply by n_messages for hashes done per launch.
     */
    uint32_t  repeats;

    uint64_t  n_partials;   /* one digest per work-group in the launch */
    uint64_t  partial_cap;  /* how many the buffers hold; n_partials <= this */
    void     *partials;     /* host staging for the readback; elements are
                               partial_word_bytes wide */

    /*
     * Streaming mode: re-upload the corpus slice before every launch, so the
     * host-to-device link is inside the timed region. Off by default -- see
     * vb_ocl_ctx_set_stream.
     */
    int         stream;
    const void *host_slice;   /* not owned; the corpus outlives the context */
    size_t      corpus_bytes;

    /* Device-side times of the last run, from queue profiling. Compare against
       wall time to see launch and synchronisation overhead, and against each
       other to see which side of the PCIe crossover this point sits on. */
    uint64_t  last_kernel_ns;
    uint64_t  last_transfer_ns;   /* 0 unless streaming */

    char error[512];
} vb_ocl_ctx;

/*
 * Build the program for this (lanes, streams) pair and upload the slice of the
 * corpus spanning groups [first_group, first_group + n_groups). Passing the
 * whole corpus is the single-device case.
 *
 * Slicing is what makes multi-device work: each device holds only its own
 * groups, verifies its own partial checksum, and the XOR of the partials
 * reproduces the single-device value -- exactly as the CPU thread pool does.
 *
 * Returns 0 on success; on failure ctx->error explains why and nothing leaks.
 */
int  vb_ocl_ctx_init(vb_ocl_ctx *c, const vb_ocl_device *dev,
                     const vb_corpus *corpus, unsigned streams,
                     uint64_t first_group, uint64_t n_groups);

/*
 * Turn streaming on or off after init.
 *
 * Deliberately not an argument to vb_ocl_ctx_init: geometry tuning and the
 * repeat calibration must run against the kernel alone, or they would be
 * measuring the upload. Init therefore always tunes in resident mode and this
 * flips the mode afterwards.
 *
 * Enabling it forces repeats to 1. That is not a tuning choice but a
 * correctness one: `repeats` amplifies compute without amplifying transfer, so
 * any value above 1 would inflate the compute side of the very ratio this mode
 * exists to measure.
 */
void vb_ocl_ctx_set_stream(vb_ocl_ctx *c, int on);

/* Launch once over this context's slice and fold the partials. 0 on success. */
int  vb_ocl_ctx_run(vb_ocl_ctx *c, uint32_t iterations,
                    uint64_t checksum[VB_MAX_DIGEST_WORDS]);

/*
 * Split of the above, so several devices run concurrently: enqueue on every
 * device first, then collect from each. Calling run() in a loop would serialise
 * them, because its readback blocks.
 */
int  vb_ocl_ctx_enqueue(vb_ocl_ctx *c, uint32_t iterations);
int  vb_ocl_ctx_collect(vb_ocl_ctx *c,
                        uint64_t checksum[VB_MAX_DIGEST_WORDS]);

void vb_ocl_ctx_free(vb_ocl_ctx *c);

#endif /* VALUBENCH_OPENCL_BACKEND_H */
