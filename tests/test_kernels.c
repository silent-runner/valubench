/*
 * test_kernels.c -- every kernel must agree with the scalar reference.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * The XOR checksum is invariant to lane, stream and iteration assignment, so a
 * single expected value from the reference validates every (ISA, streams)
 * combination. Kernels are checked over several index ranges, including ranges
 * that are not a whole multiple of any kernel's group size, to catch off-by-one
 * errors in work distribution.
 */

#include "valubench.h"
#include "bench.h"
#include "cpu_features.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks   = 0;

/*
 * Run one kernel over one corpus, whichever kind it is. Device kernels upload
 * the corpus and drive a context; CPU kernels are a direct call. Both must
 * produce the same checksum -- that invariance is the whole point of the XOR
 * reduction (valubench.h).
 */
static int run_kernel(const vb_kernel *k, const vb_corpus *c, uint64_t groups,
                      uint32_t iters, uint64_t out[VB_MAX_DIGEST_WORDS],
                      char *err, size_t errn)
{
    if (!k->device) {
        k->fn(c->words, groups, c->blocks, iters, out);
        return 0;
    }

    /* The first device this kernel's backend reaches. */
    vb_device devs[VB_DEV_MAX];
    int n = vb_devices(devs, VB_DEV_MAX), d = 0;
    while (d < n && !devs[d].present[k->backend])
        d++;
    const vb_dev_backend *be = vb_backend((vb_backend_id) k->backend);
    if (d == n || !be) {
        snprintf(err, errn, "no %s device", vb_backend_name((vb_backend_id)
                                                             k->backend));
        return -1;
    }

    vb_dev_options opt;
    vb_dev_options_default(&opt);
    opt.iterations = iters;

    vb_dev_ctx ctx;
    if (vb_dev_ctx_init(&ctx, be, &devs[d].via[k->backend], c, k->streams, 0,
                        c->n_messages / (k->lanes * k->streams), &opt) != 0) {
        snprintf(err, errn, "%s", ctx.error);
        return -1;
    }
    int rc = vb_dev_ctx_run(&ctx, iters, out);
    if (rc != 0)
        snprintf(err, errn, "%s", ctx.error);
    vb_dev_ctx_free(&ctx);
    return rc;
}

static void check_range(const vb_kernel *k, uint32_t start, uint64_t groups,
                        uint32_t iters, uint32_t msg_bytes)
{
    uint64_t got[VB_MAX_DIGEST_WORDS], want[VB_MAX_DIGEST_WORDS];
    uint64_t count = groups * (k->lanes * k->streams);
    const vb_algorithm *alg = vb_algorithm_by_id(k->alg);
    vb_corpus c;
    char err[512] = "";

    checks++;

    if (vb_corpus_build(&c, alg, k->lanes, start, count, msg_bytes) != 0) {
        failures++;
        printf("  FAIL  %s: corpus build failed (msg=%u)\n", k->name, msg_bytes);
        return;
    }

    vb_reference_checksum(alg, start, count, msg_bytes, iters, want);
    if (run_kernel(k, &c, groups, iters, got, err, sizeof err) != 0) {
        failures++;
        printf("  FAIL  %s: %s\n", k->name, err);
        vb_corpus_free(&c);
        return;
    }
    vb_corpus_free(&c);

    if (memcmp(got, want, sizeof got) != 0) {
        failures++;
        printf("  FAIL  %s start=%u groups=%llu iters=%u msg=%u\n",
               k->name, start, (unsigned long long) groups, iters, msg_bytes);
    }
}

/* Block-count boundaries: 1, 2, 3, 4 and more blocks, plus each edge. */
static const uint32_t msg_sizes[] = {
    1, 16, 55, 56, 63, 64, 119, 120, 128, 183, 184, 200, 512, 1000, 4096,
};

int main(void)
{
    size_t count;
    const vb_kernel *ks = vb_kernels(&count);

    printf("Kernel agreement with scalar reference\n");
    printf("  cpu isa: sse2=%d avx2=%d avx512f=%d\n\n",
           vb_cpu_has_sse2(), vb_cpu_has_avx2(), vb_cpu_has_avx512f());

    /*
     * Structural check, independent of what this CPU can run: every kernel's
     * group size must divide VB_BATCH_LCM, or the batch would not be a whole
     * number of groups and its checksum would cover different messages from
     * every other kernel's.
     */
    for (size_t i = 0; i < count; i++) {
        /* Structural, but not knowable for every kernel: an SVE row takes its
           lane count from the hardware, and reports 0 when this CPU has no
           SVE. There is no group size to check, so there is nothing to fail. */
        if (ks[i].lanes == 0) {
            printf("  skip  %-16s (lane count needs the ISA)\n", ks[i].name);
            continue;
        }
        if (!vb_batch_divides(&ks[i])) {
            /* A run-time lane count that does not tile the batch is this
               machine's vector length, not a defect: main.c declines the
               kernel with a diagnostic and exit 2, and the digests it does
               produce are correct. Only a fixed-width kernel can be wrong
               here, and then it is wrong on every machine. */
            if (ks[i].lanes_runtime) {
                printf("  skip  %-16s (%u lanes x %u streams does not tile "
                       "%u at this vector length)\n", ks[i].name,
                       ks[i].lanes, ks[i].streams, VB_BATCH_LCM);
                continue;
            }
            checks++;
            failures++;
            printf("  FAIL  %s: group size %u does not divide VB_BATCH_LCM "
                   "(%u)\n", ks[i].name, ks[i].lanes * ks[i].streams,
                   VB_BATCH_LCM);
            continue;
        }
        checks++;
    }

    for (size_t i = 0; i < count; i++) {
        const vb_kernel *k = &ks[i];
        const vb_algorithm *alg = vb_algorithm_by_id(k->alg);
        uint32_t min_iter = vb_alg_min_iter_bytes(alg);

        if (!k->available()) {
            printf("  skip  %-16s (not available here)\n", k->name);
            continue;
        }

        /* A single group, a few groups, and a size that is prime relative to
           any plausible internal blocking. */
        check_range(k, 0, 1, 1, 55);
        check_range(k, 0, 7, 1, 55);
        check_range(k, 0, 64, 1, 55);
        check_range(k, 1, 13, 1, 55);

        /* Non-zero, non-aligned start indices. */
        check_range(k, 12345, 11, 1, 55);
        check_range(k, 0xfffff000u, 5, 1, 55);  /* near the 32-bit index wrap */

        /* Iterated hashing: the digest-feedback path is entirely separate
           code from the single-iteration path and needs its own coverage. */
        check_range(k, 0, 3, 2, min_iter);
        check_range(k, 0, 3, 5, min_iter + 39);
        check_range(k, 777, 5, 17, min_iter + 9);

        /*
         * Message lengths across every block-count boundary. 55/56 is where a
         * message stops fitting in one block, 119/120 the next, and so on --
         * exactly where multi-block chaining goes wrong if it is going to.
         */
        for (unsigned mb = 0; mb < sizeof msg_sizes / sizeof msg_sizes[0]; mb++)
            check_range(k, 0, 3, 1, msg_sizes[mb]);

        /* Multi-block and iterated together, the hardest combination: the
           digest feedback must land in block 0 while later blocks reload. */
        check_range(k, 0, 2, 3, min_iter + 200);
        check_range(k, 0, 2, 3, 200 > min_iter ? 200 : min_iter);
        check_range(k, 41, 2, 4, 1000);

        printf("  ok    %-16s (%s, %u lanes x %u streams)\n",
               k->name, alg->name, k->lanes, k->streams);
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
