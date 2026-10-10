/*
 * test_affinity.c -- a pool must leave its caller's CPU mask as it found it.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Worker 0 of a pool is the calling thread, so pinning it narrows the caller
 * to one CPU. pool_destroy() never widened it again, and Linux threads inherit
 * their creator's mask: every thread started after the first pool -- the
 * parallel reference behind a device kernel that autotune reached after its
 * CPU probes, and a device driver's own threads -- ran on that one CPU. Slow
 * rather than wrong, and invisible in the result, so this asserts the mask
 * directly.
 *
 * Twice, because the first fix for the related caching bug held for one pool
 * and not for the second.
 */

#define _GNU_SOURCE
#include "bench.h"
#include "valubench.h"
#include "algorithm.h"
#include "vb_threads.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
#if defined(__linux__)
    cpu_set_t before;
    if (sched_getaffinity(0, sizeof before, &before) != 0) {
        printf("  FAIL  affinity      could not read this thread's mask\n");
        return 1;
    }
    if (CPU_COUNT(&before) < 2) {
        printf("  skip  affinity      (needs >1 allowed cpu; this process "
               "has %d)\n", CPU_COUNT(&before));
        return 0;
    }

    size_t count;
    const vb_kernel *ks = vb_kernels(&count), *k = NULL;
    for (size_t i = 0; i < count; i++)
        if (!strcmp(ks[i].name, "md5/scalar-s1"))
            k = &ks[i];
    if (!k) {
        printf("  FAIL  affinity      md5/scalar-s1 is not registered\n");
        return 1;
    }

    vb_config cfg;
    vb_config_defaults(&cfg);
    cfg.alg = vb_algorithm_by_id(VB_ALG_MD5);
    cfg.threads = 2;
    cfg.pin_cpu = 1;
    cfg.working_set_kb = 256;
    cfg.n_samples = 2;
    cfg.target_ms = 5;
    cfg.warmup_ms = 5;

    int failures = 0;
    for (int run = 1; run <= 2; run++) {
        vb_result r;
        memset(&r, 0, sizeof r);
        if (vb_measure(k, &cfg, &r) != 0) {
            printf("  FAIL  affinity      measurement %d did not complete\n",
                   run);
            return 1;
        }
        /* The mask comes back whole trivially if it was never narrowed, so a
           sandbox that refuses the pin would pass this without testing it.
           Require the pin to have happened. */
        if (r.pin_failed || r.pinned_cpus != 2) {
            printf("  FAIL  affinity      pool %d did not pin (pinned_cpus "
                   "%u%s); nothing was tested\n", run, r.pinned_cpus,
                   r.pin_failed ? ", refused" : "");
            return 1;
        }

        cpu_set_t after;
        if (sched_getaffinity(0, sizeof after, &after) != 0) {
            printf("  FAIL  affinity      could not re-read the mask\n");
            return 1;
        }
        if (!CPU_EQUAL(&before, &after)) {
            printf("  FAIL  affinity      after pool %d the caller is on %d "
                   "of its %d cpus\n", run, CPU_COUNT(&after),
                   CPU_COUNT(&before));
            failures++;
        }
    }
    if (!failures)
        printf("  ok    affinity      caller's mask restored after each "
               "pool\n");
    return failures ? 1 : 0;
#else
    printf("  skip  affinity      (no thread affinity API on this platform)\n");
    return 0;
#endif
}
