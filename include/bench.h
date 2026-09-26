/*
 * bench.h -- measurement harness.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 */

#ifndef VALUBENCH_BENCH_H
#define VALUBENCH_BENCH_H

#include "valubench.h"
#include "power.h"
#include "opencl.h"

#include <stdint.h>

#define VB_MAX_SAMPLES 256
#define VB_MAX_THREADS 1024

/*
 * How the corpus reaches a device.
 *
 * RESIDENT uploads it once at setup and launches against it repeatedly. That is
 * the right model for work that lives on the accelerator, and it is what every
 * device figure in this benchmark meant before streaming existed.
 *
 * STREAM re-uploads it before every launch, putting the host-to-device link
 * inside the timed region. This is the mode goal 2 needs: without it there is
 * no PCIe traffic to be limited by, so "how many iterations keep the
 * computation GPU-limited rather than PCIe-limited" has nothing to measure.
 */
typedef enum {
    VB_TRANSFER_RESIDENT = 0,
    VB_TRANSFER_STREAM
} vb_transfer_mode;

typedef struct {
    const vb_kernel *kernel;
    unsigned threads;
    unsigned iterations;
    unsigned message_bytes;
    unsigned blocks;              /* compression blocks per message; the
                                     block size is the algorithm's, 64 bytes
                                     for MD5 and SHA-1, 128 for SHA-512 */
    uint64_t batch_messages;
    uint64_t working_set_bytes;   /* size of the message corpus */

    unsigned n_samples;
    double   sample_hps[VB_MAX_SAMPLES];   /* hashes/sec per iteration */

    double   min, max, median, mean, stddev, cov;

    uint64_t hashes_per_iter;
    uint64_t total_hashes;
    double   total_seconds;

    int      verified;          /* checksum matched on every timed iteration */
    uint64_t checksum[VB_MAX_DIGEST_WORDS];
    const vb_algorithm *alg;

    /*
     * Set when the kernel never got as far as producing an answer: the corpus
     * could not be allocated, the worker threads could not all start, or a
     * device could not be set up. Empty when the kernel ran.
     *
     * This is kept apart from `verified` because the two call for opposite
     * reactions. A verification failure means the hardware computed a wrong
     * digest, which should stop a sweep and send someone to check the
     * machine. A run that never started says nothing about the hardware, and
     * reporting it as a wrong answer sent people looking at clocks and cooling
     * for what was an out-of-memory corpus.
     */
    char run_error[512];   /* as large as device_error, which it may carry */

    /* Device kernels only; empty otherwise. */
    char device_name[128];
    char device_vendor[128];
    char device_driver[64];
    char device_error[512];
    size_t   device_global;     /* work-items launched, first device */
    size_t   device_local;      /* work-group size, first device */
    uint32_t device_repeats;    /* corpus sweeps per launch, first device */
    double   device_busy;       /* fraction of wall time spent in the kernel */
    int      device_count;      /* devices used concurrently */

    /*
     * Transfer accounting. Meaningful only in streaming mode; zero otherwise.
     *
     * compute_transfer_ratio is the figure goal 2 asks for: kernel time over
     * transfer time for the same work. Above 1 the device is compute-limited
     * and the accelerator is earning its place; below 1 the link is the
     * constraint and more device throughput buys nothing.
     *
     * Note the ratio answers the question for a *pipelined* implementation too,
     * even though this one uploads and computes in sequence. Overlapping the
     * two can hide the smaller of them but never the larger, so whichever side
     * exceeds 1.0 is the binding constraint either way.
     */
    vb_transfer_mode transfer;
    double   device_transfer_busy;   /* fraction of wall spent uploading */
    double   device_transfer_gbps;   /* effective host-to-device rate */
    uint64_t device_transfer_bytes;  /* uploaded per pass, all devices */
    double   compute_transfer_ratio; /* kernel ns / transfer ns */

    /*
     * Per-pass device times, raw. These are what let a sweep *solve* for the
     * balance point instead of bracketing it.
     *
     * Transfer is constant in the iteration count -- the same bytes cross the
     * link at every point -- while kernel time is linear in it, because the
     * workload is defined so that every iteration is identical work. So
     *
     *     kernel_ns(N) = a + b*N          transfer_ns = T
     *
     * and the balance point is N* = (T - a) / b, obtainable from two points and
     * refined by a fit over more. `a` is the fixed per-launch cost; ignoring it
     * is what makes the cruder estimate N/ratio drift.
     */
    uint64_t device_kernel_ns_per_pass;
    uint64_t device_transfer_ns_per_pass;

    /* Pinning was requested and at least one worker was refused its CPU. */
    int      pin_failed;
    unsigned pinned_cpus;       /* distinct CPUs the workers ran on; 0 if unpinned */

    vb_power power;             /* energy over the timed region */
    vb_gpu_clocks gpu_clocks;   /* SM clock across the timed region, NVML only */
} vb_result;

/*
 * Which kernels autotune is allowed to consider.
 *
 * Device and CPU kernels compete in the same ranking by default, which is what
 * "give me the fastest thing this machine has" means. But two questions need
 * the losing side on purpose: "what is the whole CPU worth here", which is the
 * baseline any accelerator is judged against, and "what does the device do",
 * without a fast CPU hiding it. Without this, a GPU box cannot autotune a CPU
 * number at all -- the device kernel wins every probe and the result is
 * labelled a CPU baseline while being nothing of the sort.
 */
typedef enum {
    VB_WHERE_ANY = 0,
    VB_WHERE_CPU,
    VB_WHERE_DEVICE
} vb_where;

typedef struct {
    unsigned target_ms;         /* wall time per timed iteration */
    unsigned n_samples;         /* timed iterations */
    unsigned warmup_ms;
    unsigned threads;           /* 0 = one per online CPU */
    unsigned iterations;        /* chained MD5s per hash; >= 1 */
    unsigned message_bytes;     /* message length */
    const vb_algorithm *alg;    /* which hash to benchmark */
    unsigned working_set_kb;    /* target corpus size; sets the batch count */
    /* OpenCL devices to use. Empty means every device found. */
    /*
     * Sized by what the measurement path can hold, not by the thread limit.
     * These were VB_MAX_THREADS (1024) while measure_device() reserves
     * VB_OCL_MAX_DEVICES (32), and the copy between them was unchecked: 33
     * valid indices segfaulted. Out-of-range indices were already rejected, so
     * reaching it needed duplicates, which the parser accepted.
     */
    int      device_index[VB_OCL_MAX_DEVICES];
    int      device_count;
    vb_transfer_mode transfer;  /* how the corpus reaches a device */
    vb_where where;             /* which kernels autotune may pick from */
    double   cov_threshold;     /* result flagged unstable above this */
    int      pin_cpu;           /* pin worker threads to distinct cores */
    const char *force_kernel;   /* NULL = autotune */

    /*
     * A reference checksum supplied by the caller instead of recomputed here.
     *
     * The expected value is a pure function of (algorithm, message_bytes,
     * iterations, start_index, count) and of the reference implementation --
     * no machine dependence at all, which is the same invariance the
     * cross-machine fingerprint rests on. So a sweep can compute a whole
     * iteration ladder in one checkpointed pass (--reference-ladder) and hand
     * each point its answer, rather than every point walking the same chain
     * prefix again.
     *
     * The gate is unchanged in kind: the kernel's output must still equal a
     * value produced by the scalar reference. What changes is when that value
     * was computed, so the result records `expected_source` and a reader can
     * tell the difference.
     */
    uint64_t expected[VB_MAX_DIGEST_WORDS];
    int      have_expected;
} vb_config;

void vb_config_defaults(vb_config *cfg);

/* Monotonic nanoseconds, immune to NTP adjustment. */
uint64_t vb_now_ns(void);

/* Online CPUs, at least 1. */
unsigned vb_online_cpus(void);

/* Threads to use when --threads is not given: the CPUs this process is allowed
   on, which under taskset, a cpuset or a container is fewer than are online. */
unsigned vb_default_threads(void);

/*
 * Every kernel's group size must divide VB_BATCH_LCM, so the batch is always a
 * whole number of groups and the checksum covers the same messages regardless
 * of kernel, thread count or machine. Checked for every registered kernel by
 * tests/test_kernels.c.
 */
int vb_batch_divides(const vb_kernel *k);

/* Messages in the verified batch. The corpus size that implies is computed
   where the corpus is measured, from the corpus itself, so that the reported
   number describes what was actually built. */
uint64_t vb_batch_messages(const vb_config *cfg);

/*
 * Validate a kernel against the scalar reference over the whole corpus.
 * Returns 1 on match. This is the absolute correctness gate; it runs before
 * anything is timed, so a broken build or a faulty machine fails loudly rather
 * than reporting a fast wrong number.
 */
int vb_validate_kernel(const vb_kernel *k, const vb_config *cfg,
                       const vb_corpus *corpus,
                       uint64_t checksum_out[VB_MAX_DIGEST_WORDS],
                       uint64_t expected[VB_MAX_DIGEST_WORDS]);

/*
 * Measure one kernel. Returns 0 on success, non-zero otherwise. On failure,
 * out->run_error is non-empty if the kernel could not be run at all, and empty
 * if it ran and failed verification.
 */
int vb_measure(const vb_kernel *k, const vb_config *cfg, vb_result *out);

/*
 * Why autotune chose nothing. NULL from vb_autotune has three different
 * causes that deserve three different messages and exit codes: nothing
 * eligible was available, something ran and computed a wrong answer, or
 * everything eligible failed to start. Collapsing them is how a verification
 * failure under --where cpu came to be reported as "no kernel available".
 */
typedef struct {
    unsigned eligible;       /* available after the --where filter */
    unsigned verify_failed;  /* ran, and failed verification */
    unsigned could_not_run;  /* never produced an answer */
    char     run_error[512]; /* the first could-not-run reason */
} vb_autotune_outcome;

/*
 * Pick the fastest available kernel by short measurement. Every candidate is
 * validated first; candidates that fail validation are excluded and reported.
 * `why` may be NULL; when it is not, it says why NULL was returned.
 */
const vb_kernel *vb_autotune(const vb_config *cfg, int verbose,
                             vb_autotune_outcome *why);

#endif /* VALUBENCH_BENCH_H */
