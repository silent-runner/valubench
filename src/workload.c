/*
 * workload.c -- the workload definition, corpus, and reference oracle.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Algorithm-agnostic: everything specific comes from the vb_algorithm
 * descriptor. This is the single place that decides what bytes get hashed, so
 * the reference path and every kernel are guaranteed to agree on the input.
 */

#include "valubench.h"
#include "hashes.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

/*
 * Message content: the index in the first four bytes, then a fixed
 * deterministic pattern. Non-uniform so that a kernel which mixed up word
 * indices could not accidentally produce the right digest.
 */
static uint8_t pattern_byte(uint32_t offset)
{
    return (uint8_t) (offset * 0x9du + 0x3bu);
}

void vb_build_message(uint32_t index, uint32_t bytes, uint8_t *out)
{
    uint32_t n = bytes < 4u ? bytes : 4u;

    for (uint32_t i = 0; i < n; i++)
        out[i] = (uint8_t) (index >> (8 * i));

    for (uint32_t i = n; i < bytes; i++)
        out[i] = pattern_byte(i);
}

uint64_t vb_distinct_messages(uint32_t bytes)
{
    uint32_t n = bytes < 4u ? bytes : 4u;
    return (uint64_t) 1 << (8 * n);
}

const char *vb_workload_id(char *buf, size_t n, const vb_algorithm *alg,
                           uint32_t message_bytes, uint32_t iterations)
{
    snprintf(buf, n, "%s-full-%ux%u", alg->name, message_bytes, iterations);
    return buf;
}

/*
 * Pad a message into whole blocks exactly as the algorithm's own final() would:
 * a 0x80 terminator, zeros, then the bit length in the trailing length field,
 * in the algorithm's byte order. Doing this at corpus build time keeps padding
 * out of the kernels entirely -- they just hash `blocks` blocks.
 */
static void pad_message(const vb_algorithm *a, const uint8_t *msg,
                        uint32_t bytes, uint8_t *out, uint32_t blocks)
{
    size_t total = (size_t) blocks * a->block_bytes;
    uint64_t bit_len = (uint64_t) bytes * 8u;

    memcpy(out, msg, bytes);
    out[bytes] = 0x80;
    memset(out + bytes + 1, 0, total - bytes - 1);

    /* The length field is the last `length_bytes`; only its low 64 bits can
       ever be non-zero for messages this benchmark builds. */
    uint8_t *lf = out + total - 8;
    for (int i = 0; i < 8; i++) {
        if (a->big_endian)
            lf[i] = (uint8_t) (bit_len >> (56 - 8 * i));
        else
            lf[i] = (uint8_t) (bit_len >> (8 * i));
    }

    /* Little-endian algorithms put the length at the *start* of the field. */
    if (!a->big_endian && a->length_bytes > 8)
        memmove(out + total - a->length_bytes, lf, 8);
}

static uint64_t load_word(const vb_algorithm *a, const uint8_t *p)
{
    uint64_t v = 0;

    if (a->big_endian) {
        for (unsigned i = 0; i < a->word_bytes; i++)
            v = (v << 8) | p[i];
    } else {
        for (unsigned i = 0; i < a->word_bytes; i++)
            v |= (uint64_t) p[i] << (8 * i);
    }
    return v;
}

int vb_corpus_build(vb_corpus *c, const vb_algorithm *alg, uint32_t lanes,
                    uint32_t start_index, uint64_t n_messages,
                    uint32_t message_bytes)
{
    memset(c, 0, sizeof *c);

    if (!alg || lanes == 0 || n_messages == 0 || (n_messages % lanes) != 0)
        return -1;

    uint32_t blocks = vb_alg_blocks_for(alg, message_bytes);
    size_t n_words = (size_t) n_messages * blocks * VB_WORDS_PER_BLOCK;
    size_t bytes = n_words * alg->word_bytes;

    /* 64-byte aligned: one cache line, and a natural multiple of any block. */
    size_t alloc = (bytes + 63u) & ~(size_t) 63u;
    void *words = aligned_alloc(64, alloc);
    if (!words)
        return -1;

    uint8_t *msg = malloc(message_bytes);
    uint8_t *padded = malloc((size_t) blocks * alg->block_bytes);
    if (!msg || !padded) {
        free(msg);
        free(padded);
        free(words);
        return -1;
    }

    /*
     * Scatter each message into the lane-interleaved layout:
     *   word[((slot * blocks + b) * 16 + j) * lanes + lane]
     * so the `lanes` copies of word j sit contiguously and load as one vector.
     */
    for (uint64_t m = 0; m < n_messages; m++) {
        uint64_t slot = m / lanes;
        uint32_t lane = (uint32_t) (m % lanes);

        vb_build_message(start_index + (uint32_t) m, message_bytes, msg);
        pad_message(alg, msg, message_bytes, padded, blocks);

        for (uint32_t b = 0; b < blocks; b++) {
            for (uint32_t j = 0; j < VB_WORDS_PER_BLOCK; j++) {
                size_t at = (((slot * blocks) + b) * VB_WORDS_PER_BLOCK + j)
                          * lanes + lane;
                const uint8_t *src = padded + b * alg->block_bytes
                                   + (size_t) j * alg->word_bytes;
                uint64_t v = load_word(alg, src);

                if (alg->word_bytes == 8)
                    ((uint64_t *) words)[at] = v;
                else
                    ((uint32_t *) words)[at] = (uint32_t) v;
            }
        }
    }

    free(msg);
    free(padded);

    c->words = words;
    c->n_words = n_words;
    c->alg = alg;
    c->lanes = lanes;
    c->blocks = blocks;
    c->message_bytes = message_bytes;
    c->n_messages = n_messages;
    c->start_index = start_index;
    return 0;
}

void vb_corpus_free(vb_corpus *c)
{
    free(c->words);
    c->words = NULL;
    c->n_words = 0;
}

/* XOR is associative and commutative, so a range splits cleanly: the checksum
   of the whole equals the XOR of the checksums of its parts. That is what lets
   the reference pass -- iterations x messages of scalar hashing, and the
   dominant cost of a crossover sweep -- run on more than one core. */
struct ref_slice {
    const vb_algorithm *alg;
    uint32_t start;
    uint64_t count;
    uint32_t message_bytes;
    const uint32_t *iters;
    unsigned n_iters;
    uint64_t (*out)[VB_MAX_DIGEST_WORDS];   /* n_iters entries, owned by caller */
};

static void *ref_slice_main(void *arg)
{
    struct ref_slice *sl = (struct ref_slice *) arg;
    vb_reference_checksums(sl->alg, sl->start, sl->count, sl->message_bytes,
                           sl->iters, sl->n_iters, sl->out);
    return NULL;
}

void vb_reference_checksums_mt(const vb_algorithm *alg, uint32_t start,
                               uint64_t count, uint32_t message_bytes,
                               const uint32_t *iters, unsigned n_iters,
                               unsigned threads,
                               uint64_t out[][VB_MAX_DIGEST_WORDS])
{
    if (n_iters == 0)
        return;

    memset(out, 0, (size_t) n_iters * VB_MAX_DIGEST_WORDS * sizeof(uint64_t));

    if (threads < 2 || count < threads) {
        vb_reference_checksums(alg, start, count, message_bytes, iters,
                               n_iters, out);
        return;
    }

    struct ref_slice *sl = calloc(threads, sizeof *sl);
    pthread_t *tid = calloc(threads, sizeof *tid);
    /* One block for every slice's whole ladder, so the per-slice pointers are
       offsets into it and there is a single allocation to check and free. */
    uint64_t (*part)[VB_MAX_DIGEST_WORDS] =
        calloc((size_t) threads * n_iters, sizeof *part);

    if (!sl || !tid || !part) {           /* fall back rather than fail */
        free(sl); free(tid); free(part);
        vb_reference_checksums(alg, start, count, message_bytes, iters,
                               n_iters, out);
        return;
    }

    /*
     * Which threads actually started, per thread rather than as a high-water
     * mark. `spawned = i` assumed the successes formed a contiguous prefix:
     * if thread 1 failed and thread 2 started, the join loop treated 1 as
     * running, joined a pthread_t that was never created, and XORed slice 1's
     * untouched zeros into the reference. A wrong oracle is worse than a slow
     * one -- it fails correct kernels.
     */
    unsigned char *started = calloc(threads, 1);
    if (!started) {
        free(sl); free(tid); free(part);
        vb_reference_checksums(alg, start, count, message_bytes, iters,
                               n_iters, out);
        return;
    }

    uint64_t base = count / threads, extra = count % threads, off = 0;
    for (unsigned i = 0; i < threads; i++) {
        sl[i].alg = alg;
        sl[i].start = start + (uint32_t) off;
        sl[i].count = base + (i < extra ? 1 : 0);
        sl[i].message_bytes = message_bytes;
        sl[i].iters = iters;
        sl[i].n_iters = n_iters;
        sl[i].out = part + (size_t) i * n_iters;
        off += sl[i].count;
        if (i > 0 && pthread_create(&tid[i], NULL, ref_slice_main, &sl[i]) == 0)
            started[i] = 1;
    }
    ref_slice_main(&sl[0]);               /* this thread takes slice 0 */

    for (unsigned i = 0; i < threads; i++) {
        if (i > 0 && started[i]) {
            /* A join that fails leaves the slice in an unknown state, so
               compute it here rather than trust it. Doing the work twice is
               harmless: the slice is a pure function of its range. */
            if (pthread_join(tid[i], NULL) != 0)
                ref_slice_main(&sl[i]);
        } else if (i > 0) {
            ref_slice_main(&sl[i]);       /* a create failed; do it here */
        }
        for (unsigned k = 0; k < n_iters; k++)
            for (unsigned j = 0; j < VB_MAX_DIGEST_WORDS; j++)
                out[k][j] ^= sl[i].out[k][j];
    }
    free(started); free(sl); free(tid); free(part);
}

void vb_reference_checksum_mt(const vb_algorithm *alg, uint32_t start,
                              uint64_t count, uint32_t message_bytes,
                              uint32_t iterations, unsigned threads,
                              uint64_t checksum[VB_MAX_DIGEST_WORDS])
{
    vb_reference_checksums_mt(alg, start, count, message_bytes, &iterations, 1,
                              threads,
                              (uint64_t (*)[VB_MAX_DIGEST_WORDS]) checksum);
}

/*
 * Checksums for several iteration counts in one pass.
 *
 * The iteration scheme is a chain: hash, write the digest over the head of the
 * message, hash again. So the digest after k iterations is a *prefix* of the
 * chain for any larger k, and a sweep that asks for 1, 2, 4 ... 1024 iterations
 * is asking for the same chain over and over. Walking each message once to the
 * largest count and snapshotting at each requested one replaces the whole
 * ladder: 1024 passes instead of 2068 for a typical crossover sweep, and half
 * that again where the device and CPU sides ask for the same counts.
 *
 * `iters` must be ascending and non-empty. `out` receives one checksum per
 * entry.
 */
void vb_reference_checksums(const vb_algorithm *alg, uint32_t start,
                            uint64_t count, uint32_t message_bytes,
                            const uint32_t *iters, unsigned n_iters,
                            uint64_t out[][VB_MAX_DIGEST_WORDS])
{
    uint8_t *msg = malloc(message_bytes);

    memset(out, 0, (size_t) n_iters * VB_MAX_DIGEST_WORDS * sizeof(uint64_t));
    if (!msg || n_iters == 0)
        return;

    uint32_t maxit = iters[n_iters - 1];

    for (uint64_t n = 0; n < count; n++) {
        uint8_t digest[VB_MAX_DIGEST_BYTES];
        unsigned k = 0;

        vb_build_message(start + (uint32_t) n, message_bytes, msg);

        for (uint32_t it = 0; it < maxit; it++) {
            alg->hash(msg, message_bytes, digest);

            /* Snapshot every checkpoint that lands on this iteration. The
               guard is a while rather than an if so a duplicated count in
               `iters` is handled rather than silently skipped. */
            while (k < n_iters && iters[k] == it + 1) {
                for (unsigned j = 0; j < alg->digest_words; j++)
                    out[k][j] ^= load_word(alg,
                                           digest + (size_t) j * alg->word_bytes);
                k++;
            }

            /* Feed the digest back over the head of the message, leaving the
               rest intact, so every iteration is identical work. Callers
               guarantee message_bytes >= digest_bytes when iterations > 1. */
            if (it + 1 < maxit)
                memcpy(msg, digest, alg->digest_bytes);
        }
    }

    free(msg);
}

void vb_reference_checksum(const vb_algorithm *alg, uint32_t start,
                           uint64_t count, uint32_t message_bytes,
                           uint32_t iterations,
                           uint64_t checksum[VB_MAX_DIGEST_WORDS])
{
    /* One checkpoint is the general case with n_iters == 1. Keeping a single
       implementation means the ladder and the single value cannot drift. */
    vb_reference_checksums(alg, start, count, message_bytes,
                           &iterations, 1,
                           (uint64_t (*)[VB_MAX_DIGEST_WORDS]) checksum);
}
