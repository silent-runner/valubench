/*
 * backend.c -- OpenCL device context: build, upload, launch, verify.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * This file knows nothing about hash functions. Kernel source arrives complete
 * -- each kernel in src/kernels/gpu carries its own constants and schedules, and
 * is embedded as a byte array -- and everything here, from upload through geometry tuning to
 * the partial fold, is driven by the algorithm descriptor rather than by which
 * algorithm it is. LANES and STREAMS reach the device compiler as -D defines,
 * so one source specialises into every variant exactly as the CPU template
 * does.
 *
 * It used to assemble that source itself, at every context setup: ~290 lines of
 * MD5, SHA-1 and SHA-512 schedule generation sat in the middle of the OpenCL
 * driver, recomputing text that never varied. What remains here is the driver.
 *
 * One program per algorithm, in the table at the bottom of this section. What
 * varies between them is the entry point, the embedded source, and how many
 * words a digest is.
 */

#define _GNU_SOURCE

#include "opencl_backend.h"
#include "kernels/cpu/matrix.h"
#include "md5_kernel.h"
#include "sha1_kernel.h"
#include "sha512_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the program table -------------------------------------------------- */

/*
 * Adding a device algorithm is a .cl file and one line in VB_FOR_EACH_DEVICE_ALG
 * over in the kernel matrix, which this table and the registry both expand.
 *
 * The digest shape is deliberately not a column: it is already in the algorithm
 * descriptor, and a second copy here would be a second thing to get right. It
 * sizes the readback buffer, the work-group scratch and the final fold, none of
 * which are otherwise algorithm-specific -- SHA-512 needs 64 bytes per
 * work-group digest where MD5 needs 16.
 */
typedef struct {
    vb_alg_id   alg;
    const char *entry;
    const char *source;             /* complete, generated at build time */
} vb_ocl_program;

#define VB_DEV_PROGRAM(alg, ALG, algid, entry) \
    { algid, entry, VB_OCL_##ALG##_SOURCE },

static const vb_ocl_program PROGRAMS[] = {
    VB_FOR_EACH_DEVICE_ALG(VB_DEV_PROGRAM)
};

#undef VB_DEV_PROGRAM

static const vb_ocl_program *program_for(vb_alg_id alg)
{
    for (size_t i = 0; i < sizeof PROGRAMS / sizeof PROGRAMS[0]; i++)
        if (PROGRAMS[i].alg == alg)
            return &PROGRAMS[i];
    return NULL;
}

int vb_ocl_ctx_run(vb_ocl_ctx *c, uint32_t iterations,
                   uint64_t checksum[VB_MAX_DIGEST_WORDS]);
int vb_ocl_ctx_enqueue(vb_ocl_ctx *c, uint32_t iterations);
int vb_ocl_ctx_collect(vb_ocl_ctx *c, uint64_t checksum[VB_MAX_DIGEST_WORDS]);

static void set_err(vb_ocl_ctx *c, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->error, sizeof c->error, fmt, ap);
    va_end(ap);
}

void vb_ocl_ctx_free(vb_ocl_ctx *c)
{
    const vb_ocl *cl = vb_ocl_api();
    if (!cl || !c)
        return;

    if (c->event)      cl->ReleaseEvent(c->event);
    if (c->xfer_event) cl->ReleaseEvent(c->xfer_event);
    if (c->kernel)    cl->ReleaseKernel(c->kernel);
    if (c->program)   cl->ReleaseProgram(c->program);
    if (c->d_corpus)  cl->ReleaseMemObject(c->d_corpus);
    if (c->d_partial) cl->ReleaseMemObject(c->d_partial);
    if (c->queue)     cl->ReleaseCommandQueue(c->queue);
    if (c->context)   cl->ReleaseContext(c->context);

    free(c->partials);
    memset(c, 0, sizeof *c);
}


/*
 * Launch geometry bounds. The reduction halves its span each step, so the
 * work-group size must be a power of two; 64 is small enough for any device we
 * expect and 512 large enough that going further costs occupancy without
 * helping the reduction.
 */
#define VB_OCL_MIN_LOCAL 64u
#define VB_OCL_MAX_LOCAL 512u

/* The smallest size tune_geometry() will fall back to when a kernel permits
   less than VB_OCL_MIN_LOCAL. */
#define VB_OCL_FLOOR_LOCAL 8u

/* Bytes one work-group's reduced digest occupies. */
static size_t partial_bytes(const vb_ocl_ctx *c)
{
    return (size_t) c->partial_words * c->partial_word_bytes;
}

/* Target device time for one launch. Long enough that the ~40us of launch and
   synchronisation overhead is noise, short enough to stay responsive. */
#define VB_OCL_TARGET_LAUNCH_NS 20000000ull      /* 20 ms */

/*
 * Hard ceiling on work-items. Past one per corpus group there is nothing left
 * to stride over, but a small corpus on a large device still wants every
 * compute unit busy, so allow a generous multiple of nominal parallelism.
 * Rounded up to VB_OCL_MAX_LOCAL so every candidate geometry divides it and the
 * partial buffer sized from it is always big enough.
 */
static size_t vb_ocl_max_global(const vb_ocl_ctx *c)
{
    size_t by_corpus = (size_t) c->n_groups * c->lanes;
    size_t by_device = (size_t) c->dev.compute_units * 8192u;
    size_t cap = by_corpus > by_device ? by_corpus : by_device;

    size_t unit = VB_OCL_MAX_LOCAL > c->lanes ? VB_OCL_MAX_LOCAL : c->lanes;
    cap = ((cap + unit - 1) / unit) * unit;
    return cap;
}

/*
 * The range of work-group sizes tune_geometry() will consider for this kernel.
 * A function of its own because the partial buffers are sized from the floor
 * before tuning runs, and the two must agree.
 */
static void local_bounds(const vb_ocl_ctx *c, size_t *floor_out,
                         size_t *ceiling_out)
{
    const vb_ocl *cl = vb_ocl_api();

    /*
     * The ceiling is the smaller of what the device allows in general and what
     * it allows for *this* kernel. clGetKernelWorkGroupInfo was loaded and
     * never called, so a register-poor device -- some Mali parts cap SHA-512
     * s4 below 64 -- was offered nothing it could accept and reported "no
     * workable launch geometry" instead of running at 32.
     */
    size_t ceiling = c->dev.max_work_group ? c->dev.max_work_group
                                           : VB_OCL_MAX_LOCAL;
    {
        size_t kmax = 0;
        if (cl->GetKernelWorkGroupInfo &&
            cl->GetKernelWorkGroupInfo(c->kernel, c->dev.device,
                                       CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof kmax, &kmax, NULL) == CL_SUCCESS
            && kmax > 0 && kmax < ceiling)
            ceiling = kmax;
    }

    /* Prefer 64, but drop to what the kernel actually permits rather than
       giving up. The reduction halves its span each step, so the size must
       stay a power of two; 8 is the smallest worth attempting. */
    size_t floor_local = VB_OCL_MIN_LOCAL;
    while (floor_local > VB_OCL_FLOOR_LOCAL && floor_local > ceiling)
        floor_local >>= 1;

    *floor_out = floor_local;
    *ceiling_out = ceiling;
}

/*
 * Pick the launch geometry by measurement rather than by formula.
 *
 * The right number of work-items is a device property, not a workload one: it
 * depends on compute units, threads resident per unit, and how much the
 * scheduler needs in flight to hide latency. Guessing wrong is expensive -- an
 * earlier version launched exactly one work-item per message, tying occupancy
 * to --working-set-kb and costing 1.8x on the development iGPU.
 *
 * Probes run once at init, outside any timed region.
 */
static int tune_geometry(vb_ocl_ctx *c)
{
    const size_t cap = vb_ocl_max_global(c);
    double best = -1.0;
    size_t best_global = 0, best_local = 0;

    c->repeats = 1;

    size_t floor_local, ceiling;
    local_bounds(c, &floor_local, &ceiling);

    for (size_t local = floor_local; local <= VB_OCL_MAX_LOCAL;
         local <<= 1) {
        if (local > ceiling)
            break;

        /*
         * The reduction needs one digest of scratch per work-item, so a wide
         * digest bounds the work-group size: SHA-512 at 512 work-items wants
         * 32 KiB of local memory. Both this and max_work_group only rule out
         * larger groups, so stopping is right rather than skipping.
         */
        if (c->dev.local_mem &&
            (cl_ulong) local * partial_bytes(c) > c->dev.local_mem)
            break;

        size_t unit = local > c->lanes ? local : c->lanes;
        if (unit % local || unit % c->lanes)
            continue;                   /* geometry the stride math cannot use */

        for (size_t mult = 1; ; mult <<= 2) {
            size_t global = local * mult * c->dev.compute_units;

            global = ((global + unit - 1) / unit) * unit;
            int last = 0;
            if (global >= cap) {
                global = cap;
                last = 1;
            }

            c->global_size = global;
            c->local_size = local;
            c->n_partials = global / local;

            /*
             * A probe that fails disqualifies this geometry rather than the
             * whole context: a device can accept the local-memory request and
             * still refuse the launch for resource reasons the query does not
             * expose. Only having no workable geometry at all is fatal.
             */
            uint64_t cs[VB_MAX_DIGEST_WORDS];
            if (vb_ocl_ctx_run(c, 1, cs) != 0)      /* warm */
                break;
            uint64_t t0 = c->last_kernel_ns;
            if (vb_ocl_ctx_run(c, 1, cs) != 0)      /* measured */
                break;
            uint64_t t1 = c->last_kernel_ns;

            uint64_t ns = (t0 && t1) ? (t0 < t1 ? t0 : t1) : 0;
            double rate = ns ? (double) c->n_groups / (double) ns : 0.0;

            if (rate > best) {
                best = rate;
                best_global = global;
                best_local = local;
            }

            if (last)
                break;
        }
    }

    if (best_global == 0) {
        /* Keep whatever the last launch said if it said anything -- it is more
           informative than the generic message. */
        if (c->error[0] == 0)
            set_err(c, "could not find a workable launch geometry");
        return -1;
    }
    c->error[0] = '\0';

    c->global_size = best_global;
    c->local_size = best_local;
    c->n_partials = best_global / best_local;

    /*
     * Now choose how many corpus sweeps make a launch long enough that per-
     * launch overhead is negligible. Odd, so the XOR checksum still equals the
     * single-sweep value and verification stays strong.
     */
    uint64_t cs[VB_MAX_DIGEST_WORDS];
    if (vb_ocl_ctx_run(c, 1, cs) != 0)
        return -1;
    uint64_t one = c->last_kernel_ns;

    if (one > 0 && one < VB_OCL_TARGET_LAUNCH_NS) {
        uint64_t want = VB_OCL_TARGET_LAUNCH_NS / one;
        if (want < 1)
            want = 1;
        if (want > 0xffffu)
            want = 0xffffu;
        c->repeats = (uint32_t) (want | 1u);        /* force odd */
    }

    return 0;
}

int vb_ocl_ctx_init(vb_ocl_ctx *c, const vb_ocl_device *dev,
                    const vb_corpus *corpus, unsigned streams,
                    uint64_t first_group, uint64_t n_groups)
{
    const vb_ocl *cl = vb_ocl_api();
    cl_int err;

    memset(c, 0, sizeof *c);

    if (!cl) {
        set_err(c, "OpenCL not loaded");
        return -1;
    }

    const vb_ocl_program *prog = program_for(corpus->alg->id);
    if (!prog) {
        set_err(c, "no OpenCL kernel for %s", corpus->alg->name);
        return -1;
    }

    c->dev = *dev;
    c->lanes = corpus->lanes;
    c->streams = streams;
    c->blocks = corpus->blocks;
    /* From the descriptor the corpus already carries, not from a second copy. */
    c->partial_words = corpus->alg->digest_words;
    c->partial_word_bytes = corpus->alg->word_bytes;

    size_t group = (size_t) c->lanes * streams;
    if (corpus->n_messages % group != 0) {
        set_err(c, "corpus of %llu messages is not a multiple of the %zu-message group",
                (unsigned long long) corpus->n_messages, group);
        return -1;
    }
    if (n_groups == 0) {
        set_err(c, "empty corpus slice");
        return -1;
    }

    c->first_group = first_group;
    c->n_groups = n_groups;
    c->n_messages = n_groups * group;

    c->context = cl->CreateContext(NULL, 1, &c->dev.device, NULL, NULL, &err);
    if (!c->context) {
        set_err(c, "clCreateContext: %s", vb_ocl_strerror(err));
        return -1;
    }

    /* Profiling lets us report device kernel time next to wall time, so launch
       and synchronisation overhead is visible rather than silently folded into
       the throughput figure. */
    c->queue = cl->CreateCommandQueue(c->context, c->dev.device,
                                      CL_QUEUE_PROFILING_ENABLE, &err);
    if (!c->queue) {
        set_err(c, "clCreateCommandQueue: %s", vb_ocl_strerror(err));
        goto fail;
    }

    /* ---- program ---- */

    const char *srcs[1] = { prog->source };
    c->program = cl->CreateProgramWithSource(c->context, 1, srcs, NULL, &err);
    if (!c->program) {
        set_err(c, "clCreateProgramWithSource: %s", vb_ocl_strerror(err));
        goto fail;
    }

    char opts[128];
    snprintf(opts, sizeof opts, "-DLANES=%u -DSTREAMS=%u -cl-std=CL1.2",
             c->lanes, streams);

    err = cl->BuildProgram(c->program, 1, &c->dev.device, opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        cl->GetProgramBuildInfo(c->program, c->dev.device, CL_PROGRAM_BUILD_LOG,
                                0, NULL, &log_len);
        char *log = malloc(log_len + 1);
        if (log) {
            cl->GetProgramBuildInfo(c->program, c->dev.device,
                                    CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
            log[log_len] = '\0';
            set_err(c, "kernel build failed (%s): %.400s",
                    vb_ocl_strerror(err), log);
            free(log);
        } else {
            set_err(c, "kernel build failed: %s", vb_ocl_strerror(err));
        }
        goto fail;
    }

    c->kernel = cl->CreateKernel(c->program, prog->entry, &err);
    if (!c->kernel) {
        set_err(c, "clCreateKernel: %s", vb_ocl_strerror(err));
        goto fail;
    }

    /* ---- buffers ---- */

    /* Upload only this device's slice. */
    size_t slot_words = vb_corpus_slot_words(corpus);
    size_t slice_words = (size_t) n_groups * streams * slot_words;
    size_t word_bytes = corpus->alg->word_bytes;
    const void *slice = (const unsigned char *) corpus->words
                      + (size_t) first_group * streams * slot_words * word_bytes;

    size_t corpus_bytes = slice_words * word_bytes;
    if (corpus_bytes > c->dev.max_alloc) {
        set_err(c, "corpus is %.1f MiB but the device caps a single allocation "
                   "at %.1f MiB -- reduce --working-set-kb",
                (double) corpus_bytes / 1048576.0,
                (double) c->dev.max_alloc / 1048576.0);
        goto fail;
    }

    /* Kept so streaming mode can re-upload the same bytes each launch. The
       corpus outlives every context built from it, so this does not own it. */
    c->host_slice = slice;
    c->corpus_bytes = corpus_bytes;

    c->d_corpus = cl->CreateBuffer(c->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   corpus_bytes, (void *) (uintptr_t) slice,
                                   &err);
    if (!c->d_corpus) {
        set_err(c, "corpus upload (%.1f MiB): %s",
                (double) corpus_bytes / 1048576.0, vb_ocl_strerror(err));
        goto fail;
    }

    /*
     * Partial buffer is sized for the largest launch we might choose, since
     * there is one digest per work-group and the geometry is picked below. The
     * most work-groups come from the smallest group the tuner can pick for
     * this kernel. That used to be assumed to be VB_OCL_MIN_LOCAL, but a
     * kernel capped below 64 work-items is tuned down to 32 or less, and its
     * launch wrote past the end of the buffer.
     */
    size_t floor_local, ceiling;
    local_bounds(c, &floor_local, &ceiling);
    size_t max_global = vb_ocl_max_global(c);
    c->partial_cap = max_global / floor_local;
    c->n_partials = c->partial_cap;

    c->partials = malloc(c->partial_cap * partial_bytes(c));
    if (!c->partials) {
        set_err(c, "out of memory for %llu partials",
                (unsigned long long) c->partial_cap);
        goto fail;
    }

    c->d_partial = cl->CreateBuffer(c->context, CL_MEM_WRITE_ONLY,
                                    c->partial_cap * partial_bytes(c),
                                    NULL, &err);
    if (!c->d_partial) {
        set_err(c, "partial buffer: %s", vb_ocl_strerror(err));
        goto fail;
    }

    if (tune_geometry(c) != 0)
        goto fail;

    return 0;

fail:
    {
        char saved[sizeof c->error];
        memcpy(saved, c->error, sizeof saved);
        vb_ocl_ctx_free(c);
        memcpy(c->error, saved, sizeof saved);
    }
    return -1;
}

void vb_ocl_ctx_set_stream(vb_ocl_ctx *c, int on)
{
    c->stream = on ? 1 : 0;
    if (on)
        c->repeats = 1;
}

int vb_ocl_ctx_enqueue(vb_ocl_ctx *c, uint32_t iterations)
{
    const vb_ocl *cl = vb_ocl_api();
    cl_int err;

    size_t global = c->global_size;
    size_t local  = c->local_size;

    /* Each work-group writes one partial and readback fetches all of them, so
       a launch with more groups than the buffers hold writes out of bounds on
       the device and reads out of bounds on the host. Refuse it. */
    if (local == 0 || global / local > c->partial_cap) {
        set_err(c, "launch of %llu work-groups exceeds the %llu partials "
                   "allocated", (unsigned long long) (local ? global / local : 0),
                (unsigned long long) c->partial_cap);
        return -1;
    }

    cl_uint a = 0;
    err  = cl->SetKernelArg(c->kernel, a++, sizeof c->d_corpus, &c->d_corpus);
    err |= cl->SetKernelArg(c->kernel, a++, sizeof(cl_uint), &c->blocks);
    err |= cl->SetKernelArg(c->kernel, a++, sizeof(cl_uint), &iterations);
    cl_ulong groups = c->n_groups;
    err |= cl->SetKernelArg(c->kernel, a++, sizeof groups, &groups);
    err |= cl->SetKernelArg(c->kernel, a++, sizeof(cl_uint), &c->repeats);
    err |= cl->SetKernelArg(c->kernel, a++, sizeof c->d_partial, &c->d_partial);
    /* Local scratch for the work-group reduction: one digest per work-item. */
    err |= cl->SetKernelArg(c->kernel, a++, local * partial_bytes(c), NULL);

    if (err != CL_SUCCESS) {
        set_err(c, "clSetKernelArg: %s", vb_ocl_strerror(err));
        return -1;
    }

    if (c->event) {
        cl->ReleaseEvent(c->event);
        c->event = NULL;
    }
    if (c->xfer_event) {
        cl->ReleaseEvent(c->xfer_event);
        c->xfer_event = NULL;
    }

    /*
     * Streaming: put the corpus back across the link before every launch, so
     * the host-to-device copy is inside the timed region and the measurement
     * can find where compute overtakes transfer.
     *
     * The write is non-blocking and the queue is in-order, so the kernel waits
     * on it without the host blocking -- the copy and the launch are timed
     * separately by their own events rather than by bracketing wall clock.
     */
    if (c->stream) {
        err = cl->EnqueueWriteBuffer(c->queue, c->d_corpus, CL_FALSE, 0,
                                     c->corpus_bytes,
                                     (void *) (uintptr_t) c->host_slice,
                                     0, NULL, &c->xfer_event);
        if (err != CL_SUCCESS) {
            set_err(c, "corpus upload (%.1f MiB): %s",
                    (double) c->corpus_bytes / 1048576.0,
                    vb_ocl_strerror(err));
            return -1;
        }
    }

    err = cl->EnqueueNDRangeKernel(c->queue, c->kernel, 1, NULL, &global,
                                   &local, 0, NULL, &c->event);
    if (err != CL_SUCCESS) {
        set_err(c, "clEnqueueNDRangeKernel (global=%zu local=%zu): %s",
                global, local, vb_ocl_strerror(err));
        return -1;
    }

    /* Push it to the device now rather than at the blocking read, so several
       devices actually overlap instead of starting one at a time. */
    cl->Flush(c->queue);
    return 0;
}

/* Elapsed device time for a completed event, then release it. 0 if unavailable
   -- profiling is best-effort and a driver that declines is not an error. */
static uint64_t event_ns(const vb_ocl *cl, cl_event *ev)
{
    uint64_t ns = 0;

    if (*ev) {
        cl_ulong t0 = 0, t1 = 0;
        if (cl->GetEventProfilingInfo(*ev, CL_PROFILING_COMMAND_START,
                                      sizeof t0, &t0, NULL) == CL_SUCCESS &&
            cl->GetEventProfilingInfo(*ev, CL_PROFILING_COMMAND_END,
                                      sizeof t1, &t1, NULL) == CL_SUCCESS &&
            t1 > t0)
            ns = (uint64_t) (t1 - t0);
        cl->ReleaseEvent(*ev);
        *ev = NULL;
    }
    return ns;
}

int vb_ocl_ctx_collect(vb_ocl_ctx *c, uint64_t checksum[VB_MAX_DIGEST_WORDS])
{
    const vb_ocl *cl = vb_ocl_api();
    cl_int err;

    err = cl->EnqueueReadBuffer(c->queue, c->d_partial, CL_TRUE, 0,
                                c->n_partials * partial_bytes(c),
                                c->partials, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_err(c, "reading partials: %s", vb_ocl_strerror(err));
        return -1;
    }

    c->last_kernel_ns = event_ns(cl, &c->event);
    c->last_transfer_ns = event_ns(cl, &c->xfer_event);

    /* Fold this device's work-group partials. XOR is associative, so folding
       here and again across devices gives the same value as one device would.
       Slots past the digest width stay zero, which is what the reference
       checksum carries there too. */
    for (unsigned w = 0; w < VB_MAX_DIGEST_WORDS; w++)
        checksum[w] = 0;

    if (c->partial_word_bytes == 8) {
        const uint64_t *p = (const uint64_t *) c->partials;
        for (uint64_t i = 0; i < c->n_partials; i++)
            for (unsigned w = 0; w < c->partial_words; w++)
                checksum[w] ^= p[i * c->partial_words + w];
    } else {
        const uint32_t *p = (const uint32_t *) c->partials;
        for (uint64_t i = 0; i < c->n_partials; i++)
            for (unsigned w = 0; w < c->partial_words; w++)
                checksum[w] ^= p[i * c->partial_words + w];
    }

    return 0;
}

int vb_ocl_ctx_run(vb_ocl_ctx *c, uint32_t iterations,
                   uint64_t checksum[VB_MAX_DIGEST_WORDS])
{
    if (vb_ocl_ctx_enqueue(c, iterations) != 0)
        return -1;
    return vb_ocl_ctx_collect(c, checksum);
}
