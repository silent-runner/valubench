/*
 * device.c -- the backend-neutral half of running a device kernel.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Program composition, launch geometry, repeat calibration, the partial fold,
 * the multi-device split and the device list. It knows nothing about hash
 * functions -- the digest shape comes from the algorithm descriptor -- and
 * nothing about any API: every API call goes through a vb_dev_backend.
 */

#define _GNU_SOURCE

#include "device.h"
#include "kernels/cpu/matrix.h"
#include "device_sources.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the program table -------------------------------------------------- */

/*
 * One core per algorithm, from the list the registry also expands, so the two
 * cannot disagree. The digest shape is deliberately not a column: it is in the
 * algorithm descriptor already.
 */
typedef struct {
    vb_alg_id   alg;
    const char *name;
    const char *entry;
    const char *core;               /* src/kernels/gpu/<alg>_device_impl.h */
} vb_dev_program;

#define VB_DEV_PROGRAM(alg, ALG, algid, entry) \
    { algid, #alg, entry, VB_DEV_##ALG##_DEVICE_IMPL },

static const vb_dev_program PROGRAMS[] = {
    VB_FOR_EACH_DEVICE_ALG(VB_DEV_PROGRAM)
};

#undef VB_DEV_PROGRAM

static const vb_dev_program *program_for(vb_alg_id alg)
{
    for (size_t i = 0; i < sizeof PROGRAMS / sizeof PROGRAMS[0]; i++)
        if (PROGRAMS[i].alg == alg)
            return &PROGRAMS[i];
    return NULL;
}

/* Dialect, primitives, core: the same composition for every backend, so the
   core is the same text whichever API compiles it. Caller frees. */
static char *compose(vb_dialect d, const char *core)
{
    const char *dialect = d == VB_DIALECT_CUDA ? VB_DEV_DIALECT_CUDA
                                               : VB_DEV_DIALECT_OPENCL;
    size_t a = strlen(dialect), b = strlen(VB_DEV_DEVICE_PRIMITIVES),
           c = strlen(core);
    char *s = malloc(a + b + c + 1);
    if (!s)
        return NULL;
    memcpy(s, dialect, a);
    memcpy(s + a, VB_DEV_DEVICE_PRIMITIVES, b);
    memcpy(s + a + b, core, c + 1);
    return s;
}

char *vb_dev_program_source(vb_alg_id alg, vb_dialect d, const char **entry)
{
    const vb_dev_program *p = program_for(alg);
    if (!p)
        return NULL;
    if (entry)
        *entry = p->entry;
    return compose(d, p->core);
}

/* ---- options and small helpers ------------------------------------------ */

void vb_dev_options_default(vb_dev_options *o)
{
    memset(o, 0, sizeof *o);
    o->iterations = 1;
    o->compile_mode = VB_COMPILE_PTX_JIT;
}

static void set_err(vb_dev_ctx *c, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->error, sizeof c->error, fmt, ap);
    va_end(ap);
}

static size_t partial_bytes(const vb_dev_ctx *c)
{
    return (size_t) c->partial_words * c->partial_word_bytes;
}

void vb_dev_slice(uint64_t total_groups, int n, int i,
                  uint64_t *first, uint64_t *count)
{
    uint64_t base = total_groups / (uint64_t) n;
    uint64_t extra = total_groups % (uint64_t) n;
    uint64_t ui = (uint64_t) i;
    *count = base + (ui < extra ? 1 : 0);
    *first = ui * base + (ui < extra ? ui : extra);
}

/* ---- launch geometry ---------------------------------------------------- */

/*
 * Bounds. The reduction halves its span each step, so the work-group size must
 * be a power of two; 64 is small enough for any device we expect and 512 large
 * enough that going further costs occupancy without helping the reduction.
 */
#define VB_DEV_MIN_LOCAL 64u
#define VB_DEV_MAX_LOCAL 512u

/* The smallest size the tuner falls back to when a kernel permits less than
   VB_DEV_MIN_LOCAL. */
#define VB_DEV_FLOOR_LOCAL 8u

/* Target device time for one launch. Long enough that the ~40us of launch and
   synchronisation overhead is noise, short enough to stay responsive. */
#define VB_DEV_TARGET_LAUNCH_NS 20000000ull      /* 20 ms */

/*
 * Hard ceiling on work-items. Past one per corpus group there is nothing left
 * to stride over, but a small corpus on a large device still wants every
 * compute unit busy, so allow a generous multiple of nominal parallelism.
 * Rounded up to VB_DEV_MAX_LOCAL so every candidate geometry divides it.
 */
static size_t max_global(const vb_dev_ctx *c)
{
    size_t by_corpus = (size_t) c->n_groups * c->lanes;
    size_t by_device = (size_t) c->dev.compute_units * 8192u;
    size_t cap = by_corpus > by_device ? by_corpus : by_device;

    size_t unit = VB_DEV_MAX_LOCAL > c->lanes ? VB_DEV_MAX_LOCAL : c->lanes;
    return ((cap + unit - 1) / unit) * unit;
}

/*
 * The range of work-group sizes to consider for this kernel. The ceiling is
 * the smaller of what the device allows in general and what it allows for
 * *this* kernel: a register-poor device can cap SHA-512 s4 below 64, and must
 * then be offered something it can accept rather than nothing.
 */
static void local_bounds(const vb_dev_ctx *c, size_t *floor_out,
                         size_t *ceiling_out)
{
    size_t ceiling = c->dev.max_work_group ? c->dev.max_work_group
                                           : VB_DEV_MAX_LOCAL;
    size_t kmax = c->be->max_local(c);
    if (kmax > 0 && kmax < ceiling)
        ceiling = kmax;

    size_t floor_local = VB_DEV_MIN_LOCAL;
    while (floor_local > VB_DEV_FLOOR_LOCAL && floor_local > ceiling)
        floor_local >>= 1;

    *floor_out = floor_local;
    *ceiling_out = ceiling;
}

/* A launch the scratch and the stride arithmetic can both serve. */
static int geometry_ok(const vb_dev_ctx *c, size_t global, size_t local)
{
    if (local == 0 || (local & (local - 1)) || global == 0 || global % local)
        return 0;
    size_t unit = local > c->lanes ? local : c->lanes;
    if (unit % local || unit % c->lanes || global % c->lanes)
        return 0;
    if (c->dev.local_mem && (uint64_t) local * partial_bytes(c) > c->dev.local_mem)
        return 0;
    return 1;
}

/*
 * Pick the launch geometry by measurement rather than by formula. The right
 * number of work-items is a device property, not a workload one; guessing
 * wrong is expensive -- an early version launched one work-item per message,
 * tying occupancy to --working-set-kb, and cost 1.8x on an iGPU.
 */
static int tune_geometry(vb_dev_ctx *c)
{
    const size_t cap = max_global(c);
    double best = -1.0;
    size_t best_global = 0, best_local = 0;
    size_t floor_local, ceiling;

    c->repeats = 1;
    local_bounds(c, &floor_local, &ceiling);

    for (size_t local = floor_local; local <= VB_DEV_MAX_LOCAL; local <<= 1) {
        if (local > ceiling)
            break;
        /* Scratch is one digest per work-item, so a wide digest bounds the
           group size; only larger groups are ruled out, so stop here. */
        if (c->dev.local_mem &&
            (uint64_t) local * partial_bytes(c) > c->dev.local_mem)
            break;

        size_t unit = local > c->lanes ? local : c->lanes;
        if (unit % local || unit % c->lanes)
            continue;

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

            /* A probe that fails disqualifies this geometry rather than the
               context: a device can accept the scratch request and still
               refuse the launch for reasons no query exposes. */
            uint64_t cs[VB_MAX_DIGEST_WORDS];
            if (vb_dev_ctx_run(c, 1, cs) != 0)      /* warm */
                break;
            uint64_t t0 = c->last_kernel_ns;
            if (vb_dev_ctx_run(c, 1, cs) != 0)      /* measured */
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
        if (c->error[0] == 0)
            set_err(c, "could not find a workable launch geometry");
        return -1;
    }
    c->error[0] = '\0';
    c->global_size = best_global;
    c->local_size = best_local;
    c->n_partials = best_global / best_local;
    return 0;
}

/*
 * How many corpus sweeps make one launch long enough that per-launch overhead
 * is negligible. Odd, so the XOR checksum still equals the single-sweep value
 * and verification stays strong.
 *
 * Calibrated at the iteration count being measured. It was calibrated at one
 * iteration and kept, so a 1,024-iteration point launched for about twenty
 * seconds -- harmless on a datacenter card, and long enough for a display
 * GPU's watchdog to kill the launch.
 */
static int calibrate_repeats(vb_dev_ctx *c, uint32_t iterations)
{
    uint64_t cs[VB_MAX_DIGEST_WORDS];

    c->repeats = 1;
    if (vb_dev_ctx_run(c, iterations ? iterations : 1, cs) != 0)
        return -1;
    uint64_t one = c->last_kernel_ns;
    if (one > 0 && one < VB_DEV_TARGET_LAUNCH_NS) {
        uint64_t want = VB_DEV_TARGET_LAUNCH_NS / one;
        if (want < 1)
            want = 1;
        if (want > 0xffffu)
            want = 0xffffu;
        c->repeats = (uint32_t) (want | 1u);
    }
    return 0;
}

/* ---- the context -------------------------------------------------------- */

void vb_dev_ctx_free(vb_dev_ctx *c)
{
    if (!c)
        return;
    if (c->be && c->impl)
        c->be->destroy(c);
    free(c->partials);
    memset(c, 0, sizeof *c);
}

static void fail_keep_error(vb_dev_ctx *c)
{
    char saved[sizeof c->error];
    memcpy(saved, c->error, sizeof saved);
    vb_dev_ctx_free(c);
    memcpy(c->error, saved, sizeof saved);
}

int vb_dev_ctx_init(vb_dev_ctx *c, const vb_dev_backend *be,
                    const vb_dev_info *dev, const vb_corpus *corpus,
                    unsigned streams, uint64_t first_group, uint64_t n_groups,
                    const vb_dev_options *o)
{
    memset(c, 0, sizeof *c);
    c->be = be;
    c->dev = *dev;

    const vb_dev_program *prog = program_for(corpus->alg->id);
    if (!prog) {
        set_err(c, "no device kernel for %s", corpus->alg->name);
        return -1;
    }

    c->lanes = corpus->lanes;
    c->streams = streams;
    c->blocks = corpus->blocks;
    c->partial_words = corpus->alg->digest_words;
    c->partial_word_bytes = corpus->alg->word_bytes;

    size_t group = (size_t) c->lanes * streams;
    if (corpus->n_messages % group != 0) {
        set_err(c, "corpus of %llu messages is not a multiple of the "
                   "%zu-message group",
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

    /* ---- program ---- */

    /* Steered by the device's vendor, never by the API: see
       src/kernels/gpu/device_primitives.h. */
    char steer_defs[256];
    vb_vendor vendor = vb_vendor_classify(dev->vendor, dev->pci_vendor_id);
    if (vb_device_steers(vendor, be->dialect, o->neutral, steer_defs,
                         sizeof steer_defs, c->steers,
                         sizeof c->steers) != 0) {
        set_err(c, "%s", c->steers);
        return -1;
    }

    char defines[384];
    snprintf(defines, sizeof defines, "-DLANES=%u -DSTREAMS=%u%s%s",
             c->lanes, streams, steer_defs[0] ? " " : "", steer_defs);

    char *source = compose(be->dialect, prog->core);
    if (!source) {
        set_err(c, "out of memory composing the %s program", prog->name);
        return -1;
    }

    snprintf(c->label, sizeof c->label, "%s-s%u", prog->name, streams);
    int rc = be->build(c, source, prog->entry, defines, o);
    free(source);
    if (rc != 0) {
        fail_keep_error(c);
        return -1;
    }

    /* ---- buffers ---- */

    size_t slot_words = vb_corpus_slot_words(corpus);
    size_t slice_words = (size_t) n_groups * streams * slot_words;
    size_t word_bytes = corpus->alg->word_bytes;
    const void *slice = (const unsigned char *) corpus->words
                      + (size_t) first_group * streams * slot_words * word_bytes;
    size_t corpus_bytes = slice_words * word_bytes;

    if (c->dev.max_alloc && corpus_bytes > c->dev.max_alloc) {
        set_err(c, "corpus is %.1f MiB but the device caps a single "
                   "allocation at %.1f MiB -- reduce --working-set-kb",
                (double) corpus_bytes / 1048576.0,
                (double) c->dev.max_alloc / 1048576.0);
        fail_keep_error(c);
        return -1;
    }
    c->host_slice = slice;
    c->corpus_bytes = corpus_bytes;

    /*
     * The partial buffer is sized for the largest launch that might be chosen:
     * one digest per work-group, so the most come from the smallest group the
     * tuner can pick -- or from a pinned geometry, which may exceed anything
     * the tuner would try.
     */
    size_t floor_local, ceiling;
    local_bounds(c, &floor_local, &ceiling);
    c->partial_cap = max_global(c) / floor_local;
    if (o->pin_global && o->pin_local &&
        o->pin_global / o->pin_local > c->partial_cap)
        c->partial_cap = o->pin_global / o->pin_local;
    c->n_partials = c->partial_cap;

    c->partials = malloc(c->partial_cap * partial_bytes(c));
    if (!c->partials) {
        set_err(c, "out of memory for %llu partials",
                (unsigned long long) c->partial_cap);
        fail_keep_error(c);
        return -1;
    }

    if (be->alloc(c, slice, corpus_bytes,
                  c->partial_cap * partial_bytes(c)) != 0) {
        fail_keep_error(c);
        return -1;
    }

    /* ---- geometry and repeats ---- */

    if (o->pin_global || o->pin_local) {
        /*
         * A pinned geometry is an experimental control -- the same launch on
         * every backend, so a difference at it can only come from the
         * compiled kernel -- and is used exactly as given or refused.
         */
        size_t g = o->pin_global, l = o->pin_local;
        if (!geometry_ok(c, g, l) || l > ceiling || l < floor_local) {
            set_err(c, "--device-geometry %zu,%zu is not a launch this kernel "
                       "can use here: the group must be a power of two in "
                       "%zu..%zu that fits the work-group scratch, and the "
                       "total a multiple of it and of %u", g, l, floor_local,
                    ceiling, c->lanes);
            fail_keep_error(c);
            return -1;
        }
        c->global_size = g;
        c->local_size = l;
        c->n_partials = g / l;
        c->geometry_pinned = 1;
    } else if (tune_geometry(c) != 0) {
        fail_keep_error(c);
        return -1;
    }

    if (calibrate_repeats(c, o->iterations) != 0) {
        fail_keep_error(c);
        return -1;
    }
    return 0;
}

void vb_dev_ctx_set_stream(vb_dev_ctx *c, int on, int pinned)
{
    c->stream = on ? 1 : 0;
    if (on) {
        c->repeats = 1;
        if (pinned && !c->host_pinned)
            c->be->pin_staging(c);
    }
}

int vb_dev_ctx_enqueue(vb_dev_ctx *c, uint32_t iterations)
{
    /* Each work-group writes one partial and the readback fetches all of them,
       so a launch with more groups than the buffers hold would write out of
       bounds on the device and read out of bounds on the host. Refuse it. */
    if (c->local_size == 0 || c->global_size / c->local_size > c->partial_cap) {
        set_err(c, "launch of %llu work-groups exceeds the %llu partials "
                   "allocated",
                (unsigned long long) (c->local_size
                                      ? c->global_size / c->local_size : 0),
                (unsigned long long) c->partial_cap);
        return -1;
    }
    return c->be->launch(c, iterations, c->local_size * partial_bytes(c));
}

int vb_dev_ctx_collect(vb_dev_ctx *c, uint64_t checksum[VB_MAX_DIGEST_WORDS])
{
    if (c->be->read(c, c->n_partials * partial_bytes(c)) != 0)
        return -1;

    /* Fold this device's work-group partials. XOR is associative, so folding
       here and again across devices gives the value one device would. Slots
       past the digest width stay zero, as the reference carries them. */
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

int vb_dev_ctx_run(vb_dev_ctx *c, uint32_t iterations,
                   uint64_t checksum[VB_MAX_DIGEST_WORDS])
{
    if (vb_dev_ctx_enqueue(c, iterations) != 0)
        return -1;
    return vb_dev_ctx_collect(c, checksum);
}

/* ---- the device list ---------------------------------------------------- */

extern const vb_dev_backend vb_opencl_backend;
extern const vb_dev_backend vb_cuda_backend;

const vb_dev_backend *vb_backend(vb_backend_id b)
{
    switch (b) {
    case VB_BACKEND_OPENCL: return &vb_opencl_backend;
    case VB_BACKEND_CUDA:   return &vb_cuda_backend;
    default:                return NULL;
    }
}

const char *vb_backend_name(vb_backend_id b)
{
    return b == VB_BACKEND_CUDA ? "cuda" : "opencl";
}

const char *vb_backend_unavailable(vb_backend_id b)
{
    const vb_dev_backend *be = vb_backend(b);
    return be ? be->unavailable() : "not built";
}

const vb_dev_info *vb_device_info(const vb_device *d)
{
    for (int b = 0; b < VB_BACKEND_COUNT; b++)
        if (d->present[b])
            return &d->via[b];
    return NULL;
}

/*
 * Every backend's devices, merged by PCI address, so one card seen through
 * OpenCL and CUDA is one entry. A device whose API gives no address cannot be
 * matched and stands alone; that is the honest outcome rather than a guess by
 * name, which two identical cards would defeat.
 */
int vb_devices(vb_device *out, int max)
{
    int n = 0;
    if (max > VB_DEV_MAX)
        max = VB_DEV_MAX;

    for (int b = 0; b < VB_BACKEND_COUNT; b++) {
        const vb_dev_backend *be = vb_backend((vb_backend_id) b);
        if (!be)
            continue;
        vb_dev_info info[VB_DEV_MAX];
        int k = be->enumerate(info, VB_DEV_MAX);
        for (int i = 0; i < k; i++) {
            int match = -1;
            if (info[i].pci[0])
                for (int j = 0; j < n && match < 0; j++)
                    for (int o = 0; o < b; o++)
                        if (out[j].present[o] && !out[j].present[b] &&
                            !strcmp(out[j].via[o].pci, info[i].pci))
                            match = j;
            if (match < 0) {
                if (n >= max)
                    continue;
                memset(&out[n], 0, sizeof out[n]);
                match = n++;
            }
            out[match].present[b] = 1;
            out[match].via[b] = info[i];
        }
    }
    return n;
}
