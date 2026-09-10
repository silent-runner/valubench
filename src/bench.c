/*
 * bench.c -- measurement harness: validation, autotune, threading, timing.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 *
 * THE VERIFICATION PROTOCOL
 * =========================
 *
 * Two layers, because they catch different failures (docs/research.md 2.6).
 *
 * 1. Absolute correctness, before timing. Every candidate kernel hashes the
 *    verified batch and its checksum is compared against the scalar reference,
 *    which is itself checked against the RFC 1321 vectors and coreutils md5sum.
 *    A kernel that fails here is excluded; if the selected kernel fails, the run
 *    aborts. This catches miscompilation, bad intrinsics, and broken hardware.
 *
 * 2. Continuous correctness, during timing. The timed loop re-hashes that same
 *    verified batch and compares on EVERY iteration. A startup-only self-test
 *    cannot catch a CPU or GPU that is correct when cold and wrong under
 *    sustained load, which is the failure mode that actually matters when
 *    benchmarking near thermal or power limits.
 *
 * Because the timed range is the range the reference verified, layer 2 checks
 * against an absolutely-known value, not merely against itself.
 *
 *
 * SIZING THE BATCH
 * ================
 *
 * The batch is chosen to hit a target working set in bytes, not a fixed message
 * count. That is what makes the memory axis controllable: with message length
 * fixed, sweeping the working set from tens of kilobytes to hundreds of
 * megabytes walks the result from L1-resident to DRAM-bound, which is the
 * roofline measurement the second goal asks for.
 *
 * The count is rounded to a multiple of VB_BATCH_LCM so every kernel's group
 * size divides it and the checksum covers identical messages regardless of
 * which kernel ran.
 *
 *
 * THREADING
 * =========
 *
 * The verified batch is partitioned across threads, so one "rep" is all threads
 * together hashing the batch exactly once. Each thread holds the reference
 * checksum for its own slice, computed once at setup, and verifies its own
 * partial result every rep -- no synchronisation in the hot path.
 *
 * XOR is associative and commutative, so XORing the partials reproduces the
 * single-threaded value. The reported checksum is therefore identical at any
 * thread count, which keeps it usable as a cross-machine fingerprint rather
 * than merely a self-check.
 *
 * Workers are persistent and synchronised with two barriers per rep-batch, so
 * thread creation never lands inside a timed region, and the barrier cost is
 * amortised over every rep in that batch rather than paid per rep.
 *
 * The driving thread is itself worker 0 and runs a slice inline. An earlier
 * version had it merely wait, which left N workers plus the driver competing
 * for N cores: one core ran two runnable threads, whichever core that was
 * varied per sample, and the coefficient of variation exceeded 25%. Having the
 * driver do real work makes the runnable-thread count exactly match the core
 * count.
 */

#define _GNU_SOURCE

#include "bench.h"
#include "valubench.h"
#include "opencl_backend.h"
#include "power.h"
#include "vb_threads.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void vb_config_defaults(vb_config *cfg)
{
    /*
     * Zero first, then set the fields that have a non-zero default.
     *
     * Assigning field by field leaves anything not named holding whatever the
     * caller's stack held -- and main() puts this struct on the stack.
     * `have_expected` was never named, so an unlucky frame makes the run skip
     * the reference computation and compare against `expected`, which is
     * garbage from the same stack: a correct kernel reported VERIFICATION
     * FAILED, non-deterministically, varying with compiler and optimisation
     * level. Adding a field to vb_config must not be able to do that again.
     */
    memset(cfg, 0, sizeof *cfg);

    cfg->target_ms      = 100;
    cfg->n_samples      = 10;
    cfg->warmup_ms      = 300;
    cfg->threads        = 0;    /* one per online CPU */
    cfg->iterations     = 1;
    cfg->message_bytes  = VB_DEFAULT_MSG_BYTES;
    cfg->alg            = vb_algorithm_by_id(VB_ALG_MD5);
    cfg->working_set_kb = 1024; /* 1 MiB: L2-resident on most machines */
    cfg->device_count   = 0;   /* every device */
    cfg->transfer       = VB_TRANSFER_RESIDENT;
    cfg->where          = VB_WHERE_ANY;
    cfg->cov_threshold  = 3.5;  /* same spirit as the PTS default, RESEARCH 1.2 */
    cfg->pin_cpu        = 1;
    cfg->force_kernel   = NULL;
}

uint64_t vb_now_ns(void)
{
    struct timespec ts;
    /* CLOCK_MONOTONIC_RAW is not slewed by NTP, unlike CLOCK_MONOTONIC. */
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

unsigned vb_online_cpus(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1)
        return 1;
    if (n > VB_MAX_THREADS)
        n = VB_MAX_THREADS;
    return (unsigned) n;
}

/*
 * The CPUs this process is actually allowed on, in ascending order.
 *
 * Workers used to be pinned to 0, 1, 2 ... on the assumption that the online
 * CPUs and the permitted ones are the same set. Inside a cpuset, a container,
 * a batch scheduler or a systemd slice they are not: a process allowed only on
 * {4, 6, 8, 10} would ask for CPU 0 and be refused. The refusal was ignored,
 * so the run looked pinned and was not -- and pinning is a claim the results
 * carry.
 *
 * Falls back to the online count if the mask cannot be read, which is the old
 * behaviour and no worse than it.
 *
 * Queried and cached on the first call rather than every call. Worker 0 in
 * pool_create() runs on the thread that calls this function and pins that
 * very thread to allowed[0] -- so a second call after any pool has run no
 * longer sees the process's real mask, it sees the narrowed one the first
 * pin left behind. Autotune builds and tears down a pool per candidate
 * kernel on the same calling thread, so by the second probe every later
 * pool, including the one behind the reported result, was pinning every
 * worker to that single CPU: eight "threads" contending for one core, no
 * pin failure reported because CPU 0 always accepted the request. The cache
 * captures the mask before the first pin can touch it.
 */
unsigned vb_allowed_cpus(int *out, unsigned max)
{
    static int cached[VB_MAX_THREADS];
    static unsigned cached_n = 0;
    static int have_cache = 0;

    if (max == 0)
        return 0;

    if (!have_cache) {
        unsigned n = 0;
        int n_read = vb_self_cpu_mask(cached, VB_MAX_THREADS);

        if (n_read < 0) {
            /* No affinity notion here, or the mask could not be read: fall
               back to the online set, which is the old behaviour and no
               worse than it. */
            unsigned online = vb_online_cpus();
            for (unsigned i = 0; i < online && n < VB_MAX_THREADS; i++)
                cached[n++] = (int) i;
        } else {
            n = (unsigned) n_read;
            if (n == 0)               /* an empty mask should not happen */
                cached[n++] = 0;
        }
        cached_n = n;
        have_cache = 1;
    }

    unsigned n = cached_n < max ? cached_n : max;
    for (unsigned i = 0; i < n; i++)
        out[i] = cached[i];
    return n;
}

int vb_batch_divides(const vb_kernel *k)
{
    unsigned group = k->lanes * k->streams;

    /* lanes is 0 for a run-time-width kernel this CPU cannot run, and AArch64
       UDIV by zero yields 0 rather than trapping -- so without this the modulo
       quietly returns VB_BATCH_LCM and the kernel looks merely misconfigured.
       On x86 the same expression is SIGFPE. */
    if (group == 0)
        return 0;

    return (VB_BATCH_LCM % group) == 0;
}

uint64_t vb_batch_messages(const vb_config *cfg)
{
    uint64_t padded = (uint64_t) vb_alg_blocks_for(cfg->alg, cfg->message_bytes)
                    * cfg->alg->block_bytes;
    uint64_t target = (uint64_t) cfg->working_set_kb * 1024u;
    uint64_t n = (padded ? target / padded : 0);

    n = (n / VB_BATCH_LCM) * VB_BATCH_LCM;
    if (n < VB_BATCH_LCM)
        n = VB_BATCH_LCM;       /* smallest count every kernel can divide */
    return n;
}

/* ---- worker pool -------------------------------------------------------- */

typedef struct vb_pool vb_pool;

typedef struct {
    pthread_t  tid;
    vb_pool   *pool;
    int        cpu;             /* -1 = do not pin */

    const void *corpus;         /* start of this slice within the corpus */
    uint64_t   groups;          /* groups in this slice */
    uint64_t   expected[VB_MAX_DIGEST_WORDS];  /* reference for this slice */
    uint64_t   checksum[VB_MAX_DIGEST_WORDS];  /* what this slice produced */

    /* What this worker needs to compute `expected` itself. The reference pass
       costs iterations x messages of scalar hashing and used to run serially on
       the calling thread, which put a 30-core machine at load 1.00 for minutes
       before it measured anything. Each slice is independent, so each worker
       does its own. */
    const vb_algorithm *ref_alg;
    uint32_t   ref_start;
    uint64_t   ref_count;
    uint32_t   ref_message_bytes;
    uint32_t   ref_iterations;

    int        ok;              /* 0 if verification failed */
} vb_worker;

struct vb_pool {
    unsigned          n;
    vb_worker        *w;
    const vb_kernel  *k;
    uint32_t          blocks;
    uint32_t          iterations;
    uint64_t          reps;     /* set by the driver before each release */
    int               stop;
    /* A whole-corpus reference supplied by the caller. See vb_config. */
    int               have_expected;
    uint64_t          expected[VB_MAX_DIGEST_WORDS];

    /*
     * Workers wait here before touching a barrier: 0 pending, 1 go, -1 abort.
     *
     * Barriers used to be sized for the requested thread count, then destroyed
     * and reinitialised smaller if a pthread_create failed -- while workers
     * already released may have been blocked on them, which POSIX does not
     * allow. The gate means a barrier is initialised once, for a count that is
     * already final, and an aborted worker never reaches one.
     */
    pthread_mutex_t   gate_m;
    pthread_cond_t    gate_cv;
    int               gate;

    /* Set by any worker whose requested CPU was refused. Reported, because a
       run that could not pin is not the run that was asked for. */
    int               pin_failed;
    unsigned          pinned_cpus;   /* distinct CPUs the workers were pinned to */
    vb_barrier        start_bar;
    vb_barrier        done_bar;
};

/* Returns 0 on success, -1 if the CPU was refused. A refusal must be visible:
   a run that says it pinned and did not is a different measurement. Where the
   platform has no affinity API vb_thread_pin() is a no-op that returns 0 and
   vb_thread_pin_supported() reports the absence; see pinned_cpus below. */
static int pin_self(int cpu)
{
    if (cpu < 0)
        return 0;

    return vb_thread_pin(cpu);
}

/* Hash this worker's slice `reps` times, verifying its own partial checksum. */
static void run_slice(vb_pool *p, vb_worker *w)
{
    uint64_t cs[VB_MAX_DIGEST_WORDS];
    int ok = 1;

    for (uint64_t r = 0; r < p->reps && ok; r++) {
        p->k->fn(w->corpus, w->groups, p->blocks, p->iterations, cs);

        if (p->have_expected) {
            /* A supplied reference is for the whole corpus, so there is no
               per-slice value to compare against and the gate moves to the
               aggregate in pool_run. Rep-to-rep agreement still belongs here:
               it costs nothing, needs no reference, and catches a kernel that
               is not a function of its input -- which is most of what the
               per-rep comparison was doing. */
            if (r == 0)
                memcpy(w->checksum, cs, sizeof cs);
            else if (memcmp(cs, w->checksum, sizeof cs) != 0)
                ok = 0;
        } else if (memcmp(cs, w->expected, sizeof cs) != 0) {
            /* Each thread checks its own slice against its own reference value,
               so verification needs no cross-thread synchronisation. */
            ok = 0;
        }
    }

    w->ok = ok;
}

/* The reference pass for one slice. Pure per message, so slices share nothing
   and the result is identical to computing the whole range serially. */
static void worker_reference(vb_worker *w)
{
    if (w->pool->have_expected)
        return;                 /* the caller already knows the answer */
    vb_reference_checksum(w->ref_alg, w->ref_start, w->ref_count,
                          w->ref_message_bytes, w->ref_iterations, w->expected);
}

static void *worker_main(void *arg)
{
    vb_worker *w = (vb_worker *) arg;
    vb_pool *p = w->pool;

    /* Nothing before this touches a barrier, so a pool that gives up can let
       these threads go without any of them being mid-wait. */
    pthread_mutex_lock(&p->gate_m);
    while (p->gate == 0)
        pthread_cond_wait(&p->gate_cv, &p->gate_m);
    int go = p->gate;
    pthread_mutex_unlock(&p->gate_m);

    if (go < 0)
        return NULL;

    if (pin_self(w->cpu) != 0)
        p->pin_failed = 1;
    worker_reference(w);

    for (;;) {
        vb_barrier_wait(&p->start_bar);

        if (p->stop)
            break;

        run_slice(p, w);
        vb_barrier_wait(&p->done_bar);
    }

    return NULL;
}

static void pool_destroy(vb_pool *p)
{
    if (!p->w)
        return;

    p->stop = 1;
    vb_barrier_wait(&p->start_bar);

    for (unsigned i = 1; i < p->n; i++)
        pthread_join(p->w[i].tid, NULL);

    vb_barrier_destroy(&p->start_bar);
    vb_barrier_destroy(&p->done_bar);
    pthread_cond_destroy(&p->gate_cv);
    pthread_mutex_destroy(&p->gate_m);
    free(p->w);
    p->w = NULL;
}

/*
 * Partition the batch across threads and compute each slice's reference
 * checksum. Returns 0 on success.
 */
static int pool_create(vb_pool *p, const vb_kernel *k, const vb_config *cfg,
                       const vb_corpus *corpus, unsigned threads)
{
    unsigned group = k->lanes * k->streams;
    uint64_t total_groups = corpus->n_messages / group;
    size_t slot_words = vb_corpus_slot_words(corpus);

    if (threads > total_groups)
        threads = (unsigned) total_groups;   /* never leave a thread idle */
    if (threads < 1)
        threads = 1;

    memset(p, 0, sizeof *p);
    p->n = threads;
    p->k = k;
    p->blocks = corpus->blocks;
    p->iterations = cfg->iterations;
    p->have_expected = cfg->have_expected;
    memcpy(p->expected, cfg->expected, sizeof p->expected);

    p->w = calloc(threads, sizeof *p->w);
    if (!p->w)
        return -1;

    /* Pin from the set this process is allowed on, not from 0..n. */
    int allowed[VB_MAX_THREADS];
    unsigned n_allowed = vb_allowed_cpus(allowed, VB_MAX_THREADS);
    if (n_allowed == 0) {
        allowed[0] = 0;
        n_allowed = 1;
    }

    uint64_t base = total_groups / threads;
    uint64_t extra = total_groups % threads;
    uint64_t off = 0;

    for (unsigned i = 0; i < threads; i++) {
        vb_worker *w = &p->w[i];
        uint64_t first_msg = off * group;

        w->pool = p;
        w->cpu = cfg->pin_cpu ? allowed[i % n_allowed] : -1;
        w->groups = base + (i < extra ? 1 : 0);
        /* A corpus slot holds `lanes` messages, so scale the message offset.
           Elements are alg->word_bytes wide, hence the byte arithmetic. */
        w->corpus = (const unsigned char *) corpus->words
                  + (first_msg / corpus->lanes) * slot_words
                    * corpus->alg->word_bytes;

        w->ref_alg           = cfg->alg;
        w->ref_start         = corpus->start_index + (uint32_t) first_msg;
        w->ref_count         = w->groups * group;
        w->ref_message_bytes = cfg->message_bytes;
        w->ref_iterations    = cfg->iterations;
        off += w->groups;
    }

    /*
     * How many distinct CPUs this pool actually spread over.
     *
     * Recorded because the failure it catches is invisible otherwise. When
     * vb_allowed_cpus() was re-read per pool, autotune's first probe narrowed
     * the calling thread's mask to one CPU and every pool after it put all of
     * its workers there -- while still reporting the thread count it was asked
     * for. The result was 2.9x slow on four cores, verified, and carried no
     * warning, because pinning to a CPU you already occupy always succeeds.
     * threads_used=8 with pinned_cpus=1 now says so on the face of the result.
     *
     * Only counted where the platform can actually pin. Without an affinity
     * API vb_thread_pin() is a no-op, so every worker "succeeds" onto its
     * nominal CPU and this loop would report a full spread that never
     * happened; pinned_cpus stays 0 -- its documented "unpinned" value -- and
     * the environment capture's can_pin warning explains why.
     */
    if (cfg->pin_cpu && vb_thread_pin_supported()) {
        for (unsigned i = 0; i < threads; i++) {
            unsigned j = 0;
            while (j < i && p->w[j].cpu != p->w[i].cpu)
                j++;
            if (j == i)
                p->pinned_cpus++;
        }
    }

    /* Worker 0 is the driving thread, so only threads-1 are spawned and the
       barriers count `threads` participants in total. */
    pthread_mutex_init(&p->gate_m, NULL);
    pthread_cond_init(&p->gate_cv, NULL);
    p->gate = 0;

    vb_barrier_init(&p->start_bar, threads);
    vb_barrier_init(&p->done_bar, threads);

    if (pin_self(p->w[0].cpu) != 0)
        p->pin_failed = 1;

    /*
     * All of the workers or none of them.
     *
     * Continuing with fewer used to look like graceful degradation and was
     * not: the slices were already sized for the requested count, so the
     * workers that never started left their share of the corpus unhashed --
     * while hashes_per_iter still counted the whole of it. A pool that fell
     * back from four threads to one reported 176 MH/s against a true 43, and
     * called it verified. Four times too fast and wrong, which is the one
     * result this benchmark must never produce.
     *
     * Re-slicing for the smaller pool would be the alternative, but a machine
     * that cannot start a thread is not one to trust a measurement from.
     */
    unsigned live = 1;                     /* worker 0 is this thread */
    int short_by = 0;

    for (unsigned i = 1; i < threads; i++) {
        if (pthread_create(&p->w[i].tid, NULL, worker_main, &p->w[i]) != 0) {
            short_by = 1;
            break;
        }
        live++;
    }

    pthread_mutex_lock(&p->gate_m);
    p->gate = short_by ? -1 : 1;
    pthread_cond_broadcast(&p->gate_cv);
    pthread_mutex_unlock(&p->gate_m);

    if (short_by) {
        for (unsigned i = 1; i < live; i++)
            pthread_join(p->w[i].tid, NULL);
        vb_barrier_destroy(&p->start_bar);
        vb_barrier_destroy(&p->done_bar);
        pthread_cond_destroy(&p->gate_cv);
        pthread_mutex_destroy(&p->gate_m);
        free(p->w);
        p->w = NULL;
        fprintf(stderr, "valubench: could not start %u threads; refusing to "
                        "measure a partial pool. Try a smaller --threads.\n",
                threads);
        return -1;
    }

    /* Worker 0 runs on the calling thread. Doing its slice here, after the
       others are spawned, means every slice is computed concurrently. */
    worker_reference(&p->w[0]);

    return 0;
}

/* Run `reps` batches across the pool. Returns elapsed ns, 0 on verify failure. */
static uint64_t pool_run(vb_pool *p, uint64_t reps, int *ok)
{
    p->reps = reps;

    uint64_t t0 = vb_now_ns();
    vb_barrier_wait(&p->start_bar);
    run_slice(p, &p->w[0]);          /* the driver is worker 0 */
    vb_barrier_wait(&p->done_bar);
    uint64_t elapsed = vb_now_ns() - t0;

    *ok = 1;
    for (unsigned i = 0; i < p->n; i++)
        if (!p->w[i].ok)
            *ok = 0;

    /* With a supplied reference the slices have nothing of their own to check
       against, so the whole-corpus XOR is where the comparison happens. It is
       weaker than per-slice by exactly one case -- two slices erring so as to
       cancel -- and the single-threaded gate in vb_validate_kernel has already
       run this kernel over the whole corpus against the same value. */
    if (*ok && p->have_expected && p->reps > 0) {
        uint64_t agg[VB_MAX_DIGEST_WORDS] = { 0 };
        for (unsigned i = 0; i < p->n; i++)
            for (unsigned j = 0; j < VB_MAX_DIGEST_WORDS; j++)
                agg[j] ^= p->w[i].checksum[j];
        if (memcmp(agg, p->expected, sizeof agg) != 0)
            *ok = 0;
    }

    return *ok ? elapsed : 0;
}

/* ---- validation --------------------------------------------------------- */

int vb_validate_kernel(const vb_kernel *k, const vb_config *cfg,
                       const vb_corpus *corpus,
                       uint64_t checksum_out[VB_MAX_DIGEST_WORDS],
                       uint64_t expected[VB_MAX_DIGEST_WORDS])
{
    unsigned group = k->lanes * k->streams;

    if (!vb_batch_divides(k) || (corpus->n_messages % group) != 0)
        return 0;

    /* No worker pool on this path -- it is what a device kernel and `make check`
       use -- so parallelise the reference itself. This was the single largest
       cost of a GPU crossover sweep: one core, minutes per point.

       Unless the caller already knows the answer: a sweep that precomputed the
       whole ladder in one pass passes it in, and this pass disappears. */
    if (cfg->have_expected)
        memcpy(expected, cfg->expected, VB_MAX_DIGEST_WORDS * sizeof *expected);
    else
        vb_reference_checksum_mt(cfg->alg, corpus->start_index,
                                 corpus->n_messages, cfg->message_bytes,
                                 cfg->iterations, vb_online_cpus(), expected);
    k->fn(corpus->words, corpus->n_messages / group, corpus->blocks,
          cfg->iterations, checksum_out);

    return memcmp(checksum_out, expected,
                  VB_MAX_DIGEST_WORDS * sizeof(uint64_t)) == 0;
}

/* ---- statistics --------------------------------------------------------- */

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

/* Newton's method, so the benchmark does not link libm for one square root. */
static double vb_sqrt(double x)
{
    if (x <= 0.0)
        return 0.0;

    double g = x, prev = 0.0;
    for (int i = 0; i < 64 && g != prev; i++) {
        prev = g;
        g = 0.5 * (g + x / g);
    }
    return g;
}

static void compute_stats(vb_result *r)
{
    double sorted[VB_MAX_SAMPLES];
    memcpy(sorted, r->sample_hps, r->n_samples * sizeof(double));
    qsort(sorted, r->n_samples, sizeof(double), cmp_double);

    r->min = sorted[0];
    r->max = sorted[r->n_samples - 1];
    r->median = (r->n_samples % 2)
        ? sorted[r->n_samples / 2]
        : 0.5 * (sorted[r->n_samples / 2 - 1] + sorted[r->n_samples / 2]);

    double sum = 0.0;
    for (unsigned i = 0; i < r->n_samples; i++)
        sum += r->sample_hps[i];
    r->mean = sum / r->n_samples;

    double var = 0.0;
    for (unsigned i = 0; i < r->n_samples; i++) {
        double d = r->sample_hps[i] - r->mean;
        var += d * d;
    }
    /* Sample standard deviation; n-1 because these are samples, not the
       population of all possible runs. */
    r->stddev = (r->n_samples > 1) ? vb_sqrt(var / (r->n_samples - 1)) : 0.0;
    r->cov = (r->mean > 0.0) ? 100.0 * r->stddev / r->mean : 0.0;
}

/*
 * Choose a repetition count that makes one timed iteration last ~target_ms.
 *
 * Two things this has to get right, both learned the hard way.
 *
 * The first probe must be discarded. Releasing the barrier wakes sleeping
 * worker threads, and that wake-up cost lands entirely in the first
 * measurement. Extrapolating from it calibrated against overhead rather than
 * work: a four-thread run asked for 100 ms samples and produced 5 ms ones,
 * a 20x undershoot that made every CPU result far shorter than requested.
 *
 * And the probe must get close to the target before extrapolating. Scaling up
 * from a 2 ms measurement multiplies whatever fixed overhead it contained by
 * fifty. Growing until the probe is within 4x of the target bounds that error.
 */
static uint64_t calibrate_reps(vb_pool *p, unsigned target_ms)
{
    const double target_ns = (double) target_ms * 1e6;
    const double floor_ns = target_ns / 4.0;
    uint64_t reps = 1;
    int ok;

    pool_run(p, 1, &ok);                /* warm: pay the thread wake-up once */
    if (!ok)
        return 0;

    for (;;) {
        uint64_t ns = pool_run(p, reps, &ok);
        if (!ok)
            return 0;

        if ((double) ns >= floor_ns) {
            double want = (double) reps * (target_ns / (double) ns);
            return (uint64_t) (want < 1.0 ? 1.0 : want);
        }

        /* Grow toward the floor directly rather than in blind 4x steps, but
           never by less than 2x, so this always terminates. */
        double grow = ns ? floor_ns / (double) ns : 4.0;
        if (grow < 2.0)  grow = 2.0;
        if (grow > 64.0) grow = 64.0;

        uint64_t next = (uint64_t) ((double) reps * grow);
        reps = (next > reps) ? next : reps * 2;

        if (reps > (1ull << 40))
            return reps;                /* pathologically fast; stop scaling */
    }
}

/* ---- measurement -------------------------------------------------------- */

/* ---- device measurement ------------------------------------------------- */

/*
 * Device kernels bypass the thread pool entirely: the corpus is uploaded once
 * and one host thread drives the queue. Threads-in-flight is a device-side
 * property here (work-items), not a host one, so `threads` is reported as 1.
 */
static int measure_device(const vb_kernel *k, const vb_config *cfg,
                          const vb_corpus *corpus, vb_result *out)
{
    vb_ocl_device devs[VB_OCL_MAX_DEVICES];
    int n_avail = vb_ocl_devices(devs, VB_OCL_MAX_DEVICES);
    int idx[VB_OCL_MAX_DEVICES];
    int n_use = 0;

    if (n_avail <= 0) {
        out->verified = 0;
        return 1;
    }

    /* Which devices: all of them, or the ones named by --device. */
    if (cfg->device_count <= 0) {
        for (int i = 0; i < n_avail; i++)
            idx[n_use++] = i;
    } else {
        for (int i = 0; i < cfg->device_count; i++) {
            if (cfg->device_index[i] < 0 || cfg->device_index[i] >= n_avail) {
                snprintf(out->device_error, sizeof out->device_error,
                         "no OpenCL device %d (there are %d)",
                         cfg->device_index[i], n_avail);
                out->verified = 0;
                return 1;
            }
            idx[n_use++] = cfg->device_index[i];
        }
    }

    unsigned group = k->lanes * k->streams;
    uint64_t total_groups = corpus->n_messages / group;

    if ((uint64_t) n_use > total_groups)
        n_use = (int) total_groups;     /* never leave a device with no work */

    vb_ocl_ctx ctx[VB_OCL_MAX_DEVICES];
    uint64_t expect[VB_OCL_MAX_DEVICES][VB_MAX_DIGEST_WORDS];
    int        n_init = 0;

    /*
     * Split the corpus into contiguous group ranges, one per device, mirroring
     * the CPU thread pool. Each device verifies its own slice against its own
     * reference value, and the XOR of the slices reproduces the single-device
     * checksum.
     *
     * The split is equal, not proportional to device speed. On a heterogeneous
     * set the aggregate is therefore paced by the slowest device -- a known
     * limitation, stated here rather than silently averaged away.
     */
    uint64_t base = total_groups / (uint64_t) n_use;
    uint64_t extra = total_groups % (uint64_t) n_use;
    uint64_t off = 0;

    for (int i = 0; i < n_use; i++) {
        uint64_t mine = base + ((uint64_t) i < extra ? 1 : 0);

        if (vb_ocl_ctx_init(&ctx[i], &devs[idx[i]], corpus, k->streams,
                            off, mine) != 0) {
            snprintf(out->device_error, sizeof out->device_error, "%s",
                     ctx[i].error);
            goto fail_init;
        }
        n_init++;
        vb_ocl_ctx_set_stream(&ctx[i], cfg->transfer == VB_TRANSFER_STREAM);

        /* Parallel, like the other two paths. This is the one that hurt: a
           device crossover sweep verifies iterations x messages of scalar
           hashing per point, and on an A10 that was 733 seconds of a single
           core for one point, with nothing else running. Each device's slice
           is independent, so the split within a slice is free. */
        if (!cfg->have_expected)
            vb_reference_checksum_mt(cfg->alg,
                                     corpus->start_index
                                         + (uint32_t) (off * group),
                                     mine * group, cfg->message_bytes,
                                     cfg->iterations, vb_online_cpus(),
                                     expect[i]);
        off += mine;
    }

    /* Provenance: name the first device, and say how many are in play. */
    snprintf(out->device_name, sizeof out->device_name, "%s", ctx[0].dev.name);
    snprintf(out->device_vendor, sizeof out->device_vendor, "%s",
             ctx[0].dev.vendor);
    snprintf(out->device_driver, sizeof out->device_driver, "%s",
             ctx[0].dev.driver_version);
    out->device_count = n_use;
    out->threads = 1;

    /* ---- one pass over every device, concurrently ---- */
    uint64_t got[VB_MAX_DIGEST_WORDS];
    #define VB_DEV_PASS(ok_label)                                          \
        do {                                                               \
            for (int i = 0; i < n_use; i++)                                \
                if (vb_ocl_ctx_enqueue(&ctx[i], cfg->iterations) != 0)     \
                    goto ok_label;                                         \
            for (int i = 0; i < n_use; i++) {                              \
                uint64_t part[VB_MAX_DIGEST_WORDS];                                          \
                if (vb_ocl_ctx_collect(&ctx[i], part) != 0)                \
                    goto ok_label;                                         \
                if (!cfg->have_expected                                    \
                    && memcmp(part, expect[i], sizeof part) != 0)           \
                    mismatch = 1;                                          \
                for (int w = 0; w < VB_MAX_DIGEST_WORDS; w++) got[w] ^= part[w];             \
            }                                                              \
        } while (0)

    int mismatch = 0;
    memset(got, 0, sizeof got);
    VB_DEV_PASS(fail_run);
    if (mismatch) {
        memcpy(out->checksum, got, sizeof got);
        goto fail_run;
    }

    uint64_t expected_all[VB_MAX_DIGEST_WORDS] = { 0 };
    if (cfg->have_expected) {
        /* With a supplied value there is nothing to check per device, so the
           whole-corpus XOR is the gate. That is weaker by exactly one case --
           two devices erring so as to cancel -- and stronger in the way that
           matters here, because the value came from a pass that no device
           influenced. */
        memcpy(expected_all, cfg->expected, sizeof expected_all);
        if (memcmp(got, expected_all, sizeof got) != 0) {
            memcpy(out->checksum, got, sizeof got);
            goto fail_run;
        }
    } else {
        for (int i = 0; i < n_use; i++)
            for (int w = 0; w < VB_MAX_DIGEST_WORDS; w++)
                expected_all[w] ^= expect[i][w];
    }
    memcpy(out->checksum, expected_all, sizeof expected_all);

    /* Calibrate reps so a timed sample lasts roughly target_ms, growing until
       the probe is within 4x of the target so fixed costs are not multiplied
       up by the extrapolation. Launches are already ~20 ms by construction, so
       this usually settles immediately. */
    const double target_ns = (double) cfg->target_ms * 1e6;
    const double floor_ns = target_ns / 4.0;
    uint64_t reps = 1;

    for (;;) {
        uint64_t t0 = vb_now_ns();
        for (uint64_t r = 0; r < reps; r++) {
            memset(got, 0, sizeof got);
            VB_DEV_PASS(fail_run);
        }
        uint64_t ns = vb_now_ns() - t0;

        if ((double) ns >= floor_ns) {
            double want = (double) reps * (target_ns / (double) ns);
            reps = (uint64_t) (want < 1.0 ? 1.0 : want);
            break;
        }

        double grow = ns ? floor_ns / (double) ns : 4.0;
        if (grow < 2.0)  grow = 2.0;
        if (grow > 64.0) grow = 64.0;
        uint64_t next = (uint64_t) ((double) reps * grow);
        reps = (next > reps) ? next : reps * 2;

        if (reps > (1ull << 32))
            break;
    }

    uint64_t warm_end = vb_now_ns() + (uint64_t) cfg->warmup_ms * 1000000ull;
    while (vb_now_ns() < warm_end) {
        memset(got, 0, sizeof got);
        VB_DEV_PASS(fail_run);
    }

    /* Re-calibrate post warm-up, for the same reason as the CPU path. */
    {
        uint64_t t0 = vb_now_ns();
        for (uint64_t r = 0; r < reps; r++) {
            memset(got, 0, sizeof got);
            VB_DEV_PASS(fail_run);
        }
        uint64_t ns = vb_now_ns() - t0;
        if (ns > 0) {
            double want = (double) reps * (target_ns / (double) ns);
            reps = (uint64_t) (want < 1.0 ? 1.0 : want);
        }
    }

    unsigned n_samples = cfg->n_samples;
    if (n_samples > VB_MAX_SAMPLES)
        n_samples = VB_MAX_SAMPLES;

    /* Hashes per pass: every device sweeps its own slice repeats times. */
    uint64_t per_pass = 0;
    for (int i = 0; i < n_use; i++)
        per_pass += ctx[i].n_messages * ctx[i].repeats;
    out->hashes_per_iter = reps * per_pass;

    out->device_global = ctx[0].global_size;
    out->device_local = ctx[0].local_size;
    out->device_repeats = ctx[0].repeats;
    out->transfer = cfg->transfer;

    /* Bytes crossing the link per pass: every device uploads its own slice. */
    if (cfg->transfer == VB_TRANSFER_STREAM)
        for (int i = 0; i < n_use; i++)
            out->device_transfer_bytes += ctx[i].corpus_bytes;

    uint64_t kernel_ns = 0, transfer_ns = 0;
    vb_power_begin(&out->power);

    /* Sampled per timed iteration, which is the granularity that matters: a
       clock that falls between the first sample and the last is the difference
       between a boost figure and a sustained one. Warm-up is deliberately
       excluded -- the question is what the *measured* region ran at. */
    vb_gpu_clocks_reset(&out->gpu_clocks);

    for (unsigned si = 0; si < n_samples; si++) {
        uint64_t t0 = vb_now_ns();
        vb_gpu_clocks_sample(&out->gpu_clocks);
        for (uint64_t r = 0; r < reps; r++) {
            memset(got, 0, sizeof got);
            VB_DEV_PASS(fail_run);
            if (memcmp(got, expected_all, sizeof got) != 0) {
                out->verified = 0;
                out->n_samples = si;
                goto fail_run;
            }
            /* Device time of the slowest device: they run concurrently, so the
               pass is only as fast as its laggard. Transfer is accounted the
               same way and separately, so the two can be compared. */
            uint64_t slowest = 0, slowest_xfer = 0;
            for (int i = 0; i < n_use; i++) {
                if (ctx[i].last_kernel_ns > slowest)
                    slowest = ctx[i].last_kernel_ns;
                if (ctx[i].last_transfer_ns > slowest_xfer)
                    slowest_xfer = ctx[i].last_transfer_ns;
            }
            kernel_ns += slowest;
            transfer_ns += slowest_xfer;
        }
        double sec = (double) (vb_now_ns() - t0) / 1e9;
        vb_gpu_clocks_sample(&out->gpu_clocks);
        out->sample_hps[si] = (double) out->hashes_per_iter / sec;
        out->total_seconds += sec;
        out->total_hashes += out->hashes_per_iter;
    }

    vb_power_end(&out->power, out->total_seconds);

    if (out->total_seconds > 0.0) {
        out->device_busy = (double) kernel_ns / 1e9 / out->total_seconds;
        out->device_transfer_busy =
            (double) transfer_ns / 1e9 / out->total_seconds;
    }

    /*
     * The goal 2 figure. Above 1.0 the kernel outlasts the upload, so the
     * device is compute-limited and the accelerator is earning its place;
     * below 1.0 the link binds and more device throughput buys nothing.
     */
    if (transfer_ns > 0) {
        out->compute_transfer_ratio = (double) kernel_ns / (double) transfer_ns;

        uint64_t passes = reps * n_samples;
        double moved = (double) out->device_transfer_bytes * (double) passes;
        out->device_transfer_gbps = moved / ((double) transfer_ns / 1e9) / 1e9;

        /* Raw per-pass times, so a sweep can solve for the balance point
           rather than bracket it. See the note in bench.h. */
        if (passes > 0) {
            out->device_kernel_ns_per_pass = kernel_ns / passes;
            out->device_transfer_ns_per_pass = transfer_ns / passes;
        }
    }

    for (int i = 0; i < n_init; i++)
        vb_ocl_ctx_free(&ctx[i]);

    out->n_samples = n_samples;
    out->verified = 1;
    compute_stats(out);
    return 0;

fail_run:
    if (out->device_error[0] == 0 && n_init > 0)
        snprintf(out->device_error, sizeof out->device_error, "%s",
                 ctx[0].error);
fail_init:
    for (int i = 0; i < n_init; i++)
        vb_ocl_ctx_free(&ctx[i]);
    out->verified = 0;
    return 1;
    #undef VB_DEV_PASS
}

static int measure_with_corpus(const vb_kernel *k, const vb_config *cfg,
                               const vb_corpus *corpus, vb_result *out)
{
    uint64_t expected[VB_MAX_DIGEST_WORDS], got[VB_MAX_DIGEST_WORDS];
    unsigned threads = cfg->threads ? cfg->threads : vb_online_cpus();

    memset(out, 0, sizeof *out);
    out->kernel = k;
    out->iterations = cfg->iterations;
    out->message_bytes = cfg->message_bytes;
    out->alg = cfg->alg;
    out->blocks = corpus->blocks;
    out->batch_messages = corpus->n_messages;
    /* Not 64: SHA-512 has 128-byte blocks, and hardcoding the MD5/SHA-1 size
       here reported half the corpus for it. The corpus knows its own algorithm;
       ask it rather than assuming. A second implementation of this arithmetic
       used to exist in vb_working_set_bytes(), correct and never called, which
       is how the two came to disagree unnoticed; it is gone. */
    out->working_set_bytes = corpus->n_messages * (uint64_t) corpus->blocks
                           * corpus->alg->block_bytes;

    /*
     * One exit for the CPU path. Six error returns used to leave vb_power
     * owning open sysfs descriptors and, where NVML loaded, a live session --
     * and the timed-sample failure returned after vb_power_begin() without
     * ever ending it. Scattered teardown is also the thing that makes the next
     * change to measurement state easy to get wrong.
     */
    int rc = 1;
    int pool_ready = 0;
    int power_running = 0;

    vb_power_open(&out->power);

    if (k->device) {
        rc = measure_device(k, cfg, corpus, out);
        vb_power_close(&out->power);
        return rc;
    }

    /* Single-threaded absolute gate first: cheapest way to reject a broken
       kernel, and it establishes the canonical checksum. */
    if (!vb_validate_kernel(k, cfg, corpus, got, expected)) {
        out->verified = 0;
        memcpy(out->checksum, got, sizeof got);
        goto done;
    }
    memcpy(out->checksum, expected, sizeof expected);

    vb_pool pool;
    if (pool_create(&pool, k, cfg, corpus, threads) != 0) {
        out->verified = 0;
        goto done;
    }
    pool_ready = 1;
    out->threads = pool.n;
    out->pinned_cpus = pool.pinned_cpus;

    uint64_t reps = calibrate_reps(&pool, cfg->target_ms);

    /*
     * Read after the first pass, not before it.
     *
     * Workers set pin_failed from worker_main, after they clear the start
     * gate; pool_create returns once they are created, not once they have
     * pinned. Reading it there caught only the driving thread's own failure,
     * so a worker refused a CPU went unreported -- the precise mislabelling
     * this field exists to prevent. calibrate_reps drives a full pass through
     * both barriers, which is the happens-before that makes every worker's
     * write visible here.
     */
    out->pin_failed = pool.pin_failed;
    if (reps == 0) {
        out->verified = 0;
        goto done;
    }

    /* Warm up: reach steady clocks and let caches and predictors settle.
       Warm-up results are discarded, but still verified. */
    uint64_t warm_end = vb_now_ns() + (uint64_t) cfg->warmup_ms * 1000000ull;
    while (vb_now_ns() < warm_end) {
        int ok;
        pool_run(&pool, reps, &ok);
        if (!ok) {
            out->verified = 0;
            goto done;
        }
    }

    /*
     * Re-calibrate now that warm-up has ramped clocks and filled caches. The
     * first calibration necessarily runs on a cold machine, which on a
     * frequency-scaling part under-counts what a rep will cost by enough to
     * leave samples well short of --time-ms.
     */
    {
        int ok;
        uint64_t ns = pool_run(&pool, reps, &ok);
        if (!ok) {
            out->verified = 0;
            goto done;
        }
        if (ns > 0) {
            double want = (double) reps *
                          ((double) cfg->target_ms * 1e6 / (double) ns);
            reps = (uint64_t) (want < 1.0 ? 1.0 : want);
        }
    }

    unsigned n = cfg->n_samples;
    if (n > VB_MAX_SAMPLES)
        n = VB_MAX_SAMPLES;

    /* One rep is all threads together covering the batch exactly once. */
    out->hashes_per_iter = reps * corpus->n_messages;

    vb_power_begin(&out->power);
    power_running = 1;

    for (unsigned i = 0; i < n; i++) {
        int ok;
        uint64_t ns = pool_run(&pool, reps, &ok);
        if (!ok) {
            out->verified = 0;
            out->n_samples = i;
            goto done;
        }
        double sec = (double) ns / 1e9;
        out->sample_hps[i] = (double) out->hashes_per_iter / sec;
        out->total_seconds += sec;
        out->total_hashes += out->hashes_per_iter;
    }

    vb_power_end(&out->power, out->total_seconds);
    power_running = 0;

    out->n_samples = n;
    out->verified = 1;
    rc = 0;
    compute_stats(out);

    /* Success falls through: the teardown is the same either way. */
done:
    if (power_running)
        vb_power_end(&out->power, out->total_seconds);
    if (pool_ready)
        pool_destroy(&pool);
    vb_power_close(&out->power);
    return rc;
}

int vb_measure(const vb_kernel *k, const vb_config *cfg, vb_result *out)
{
    vb_corpus corpus;
    int rc;

    if (vb_corpus_build(&corpus, cfg->alg, k->lanes, 0,
                        vb_batch_messages(cfg), cfg->message_bytes) != 0) {
        memset(out, 0, sizeof *out);
        out->kernel = k;
        return 1;
    }

    rc = measure_with_corpus(k, cfg, &corpus, out);
    vb_corpus_free(&corpus);
    return rc;
}

const vb_kernel *vb_autotune(const vb_config *cfg, int verbose)
{
    size_t count;
    const vb_kernel *ks = vb_kernels(&count);
    const vb_kernel *best = NULL;
    double best_hps = 0.0;

    /* Short probes -- enough to rank, not to publish. */
    vb_config probe = *cfg;
    probe.target_ms = 20;
    probe.n_samples = 3;
    probe.warmup_ms = 20;

    /*
     * The corpus layout depends only on lane count, and the registry groups
     * kernels by ISA, so a single-entry cache means one build per lane width
     * rather than one per kernel. That matters at large message sizes, where
     * the corpus can be hundreds of megabytes and rebuilding it sixteen times
     * would dominate the run.
     */
    vb_corpus corpus;
    unsigned corpus_lanes = 0;
    memset(&corpus, 0, sizeof corpus);

    if (verbose)
        printf("Autotune (%u threads):\n",
               cfg->threads ? cfg->threads : vb_online_cpus());

    for (size_t i = 0; i < count; i++) {
        const vb_kernel *k = &ks[i];
        vb_result r;

        if (k->alg != cfg->alg->id)
            continue;                   /* different algorithm entirely */

        /* Restricting where the winner may run is how a CPU baseline gets
           measured on a machine whose device kernel would otherwise win every
           probe. Silent here rather than verbose: an excluded half is a
           deliberate request, not a fact about the machine. */
        if ((cfg->where == VB_WHERE_CPU && k->device) ||
            (cfg->where == VB_WHERE_DEVICE && !k->device))
            continue;

        if (!k->available()) {
            if (verbose)
                printf("  %-14s unavailable here\n", k->name);
            continue;
        }

        if (k->lanes != corpus_lanes) {
            vb_corpus_free(&corpus);
            if (vb_corpus_build(&corpus, cfg->alg, k->lanes, 0,
                                vb_batch_messages(cfg), cfg->message_bytes) != 0) {
                if (verbose)
                    printf("  %-14s corpus allocation failed -- skipped\n",
                           k->name);
                corpus_lanes = 0;
                continue;
            }
            corpus_lanes = k->lanes;
        }

        if (measure_with_corpus(k, &probe, &corpus, &r) != 0) {
            if (verbose)
                printf("  %-14s FAILED VERIFICATION -- excluded\n", k->name);
            continue;
        }

        if (verbose)
            printf("  %-14s %10.2f MH/s\n", k->name, r.median / 1e6);

        if (r.median > best_hps) {
            best_hps = r.median;
            best = k;
        }
    }

    vb_corpus_free(&corpus);
    return best;
}
