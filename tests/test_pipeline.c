/*
 * test_pipeline.c -- the overlap pipeline's scheduling, against a mock device.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * vb_dev_pipe_begin/next/end queue chunks ahead of the one being collected,
 * across calls and on every device at once. On real hardware every way that
 * can go wrong -- a readback slot reused too soon, a chunk skipped or sent
 * twice, a device left behind -- surfaces only as a wrong checksum, if it
 * surfaces at all, and CI has no device to see it on. The mock backend here
 * turns each rule into an assertion instead, and needs no GPU:
 *
 *   - chunks are queued in order, and collected in order, each once;
 *   - chunk s + VB_PIPE_RING is never queued before chunk s is collected,
 *     since they share a readback slot;
 *   - each chunk is the slice of its pass the layer says it is;
 *   - every call returns with the chunks after it already queued, so the
 *     device never waits on the host between timed samples -- the property
 *     the per-sample drain lacked;
 *   - several devices, with different slices and chunk counts, fold into
 *     one checksum per pass;
 *   - a chunk that computes the wrong answer spoils exactly its own pass;
 *   - a failed upload or wait fails the call with the backend's reason, and
 *     the pipeline still ends and frees cleanly.
 *
 * The mock "hashes" a chunk by folding a function of each group index it
 * covers, so a group skipped or doubled changes the pass result.
 */

#include "device.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;

static void expect(int ok, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void expect(int ok, const char *fmt, ...)
{
    checks++;
    if (ok)
        return;
    failures++;
    va_list ap;
    va_start(ap, fmt);
    printf("  FAIL  ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

/* What one group contributes to a pass, per partial word. */
static uint64_t contribution(uint64_t group, unsigned word, uint32_t iterations)
{
    uint64_t z = group * 2 + word + ((uint64_t) iterations << 40) + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

#define NONE UINT64_MAX
#define MAXSEQ 4096

typedef struct {
    uint64_t first_group;        /* the device's slice of the corpus */
    uint32_t iterations;         /* what every chunk must be told */
    int64_t  fail_enqueue_at, fail_wait_at, corrupt_at;

    uint64_t queued, collected;  /* the mock's own count, to check the layer's */
    uint64_t slot_seq[VB_PIPE_RING];
    uint64_t slot[VB_PIPE_RING][2];
    char     violation[160];
    int      violations;
    int      opened;
} mock;

static mock *M(vb_dev_ctx *c) { return c->impl; }

static void violate(vb_dev_ctx *c, const char *what, uint64_t seq)
{
    mock *m = M(c);
    if (!m->violations++)
        snprintf(m->violation, sizeof m->violation, "%s (chunk %llu)", what,
                 (unsigned long long) seq);
}

static int mock_open(vb_dev_ctx *c, size_t chunk_bytes, size_t read_bytes)
{
    mock *m = M(c);
    uint64_t first, largest;
    vb_dev_slice(c->n_groups, (int) c->pipe_chunks, 0, &first, &largest);
    if (chunk_bytes != largest * c->group_bytes)
        violate(c, "chunk buffers not sized for the largest chunk", 0);
    if (read_bytes != c->n_partials * c->partial_words * c->partial_word_bytes)
        violate(c, "readback slots not sized for the partials", 0);
    for (int i = 0; i < VB_PIPE_RING; i++)
        m->slot_seq[i] = NONE;
    m->opened = 1;
    return 0;
}

static int mock_enqueue(vb_dev_ctx *c, uint64_t seq, size_t offset, size_t bytes,
                        uint64_t n_groups, uint32_t iterations, size_t shared,
                        size_t read_bytes)
{
    mock *m = M(c);
    (void) shared;
    (void) read_bytes;
    if (seq == (uint64_t) m->fail_enqueue_at) {
        snprintf(c->error, sizeof c->error, "mock: upload of chunk %llu failed",
                 (unsigned long long) seq);
        return -1;
    }
    if (seq != m->queued)
        violate(c, "chunk queued out of order", seq);
    if (iterations != m->iterations)
        violate(c, "chunk told the wrong iteration count", seq);

    uint64_t first, count;
    vb_dev_slice(c->n_groups, (int) c->pipe_chunks, (int) (seq % c->pipe_chunks),
                 &first, &count);
    if (offset != first * c->group_bytes || bytes != count * c->group_bytes ||
        n_groups != count)
        violate(c, "chunk is not the slice of its pass", seq);

    unsigned i = (unsigned) (seq % VB_PIPE_RING);
    if (m->slot_seq[i] != NONE)
        violate(c, "readback slot reused before its chunk was collected", seq);

    uint64_t g0 = offset / c->group_bytes;
    m->slot[i][0] = m->slot[i][1] = 0;
    for (uint64_t g = g0; g < g0 + n_groups; g++)
        for (unsigned w = 0; w < 2; w++)
            m->slot[i][w] ^= contribution(m->first_group + g, w, iterations);
    if (seq == (uint64_t) m->corrupt_at)
        m->slot[i][0] ^= 1;
    m->slot_seq[i] = seq;
    m->queued++;
    return 0;
}

static int mock_wait(vb_dev_ctx *c, uint64_t seq, const void **partials,
                     uint64_t *kernel_ns, uint64_t *transfer_ns)
{
    mock *m = M(c);
    if (seq == (uint64_t) m->fail_wait_at) {
        snprintf(c->error, sizeof c->error, "mock: waiting for chunk %llu failed",
                 (unsigned long long) seq);
        return -1;
    }
    if (seq != m->collected || seq >= m->queued)
        violate(c, "chunk collected out of order, or before it was queued", seq);
    unsigned i = (unsigned) (seq % VB_PIPE_RING);
    if (m->slot_seq[i] != seq)
        violate(c, "collected a slot holding another chunk", seq);
    m->slot_seq[i] = NONE;
    m->collected++;
    *partials = m->slot[i];
    *kernel_ns = 1000;
    *transfer_ns = 2000;
    return 0;
}

static void mock_destroy(vb_dev_ctx *c)
{
    mock *m = M(c);
    if (m && m->opened && m->fail_enqueue_at < 0 && m->fail_wait_at < 0 &&
        m->collected != m->queued)
        violate(c, "freed with chunks still in flight", m->collected);
    c->impl = NULL;
}

static const vb_dev_backend MOCK = {
    .name = "mock",
    .pipe_open = mock_open,
    .pipe_enqueue = mock_enqueue,
    .pipe_wait = mock_wait,
    .destroy = mock_destroy,
};

/* A context as vb_dev_ctx_init would leave it, as far as the pipeline cares. */
static void make(vb_dev_ctx *c, mock *m, uint64_t n_groups, uint64_t first_group,
                 uint64_t global_mem)
{
    memset(c, 0, sizeof *c);
    memset(m, 0, sizeof *m);
    m->first_group = first_group;
    m->fail_enqueue_at = m->fail_wait_at = m->corrupt_at = -1;
    c->be = &MOCK;
    c->impl = m;
    c->n_groups = n_groups;
    c->group_bytes = 1 << 20;
    c->local_size = 64;
    c->n_partials = 1;
    c->partial_words = 2;
    c->partial_word_bytes = 8;
    c->dev.global_mem = global_mem;
}

static void pass_value(const vb_dev_ctx *c, const mock *m, uint32_t iterations,
                       uint64_t out[2])
{
    for (unsigned w = 0; w < 2; w++)
        for (uint64_t g = 0; g < c->n_groups; g++)
            out[w] ^= contribution(m->first_group + g, w, iterations);
}

static int report_violations(const char *what, vb_dev_ctx *ctx, mock *ms, int n)
{
    int bad = 0;
    for (int d = 0; d < n; d++)
        if (ms[d].violations) {
            expect(0, "%s, device %d: %s", what, d, ms[d].violation);
            bad = 1;
        }
    (void) ctx;
    return bad;
}

/*
 * Run `calls` calls across `n` mock devices, checking every pass, the queue
 * depth between calls and the device time reported. Each call asks for
 * `passes` passes, or for seq[call] when a sequence is given: the harness
 * changes the count between calls -- calibration grows it, the timed samples
 * use another -- so a pass must be placed by where its call began, not by
 * any arithmetic that holds only while every call is the same size.
 */
static void run_seq(const char *what, int n, const uint64_t *groups,
                    const unsigned *chunks, int calls, uint64_t passes,
                    const uint64_t *seq)
{
    vb_dev_ctx ctx[VB_DEV_MAX];
    mock ms[VB_DEV_MAX];
    const uint32_t iters = 7;
    uint64_t want[VB_MAX_DIGEST_WORDS] = { 0 };
    uint64_t first = 0, most = 0;

    uint64_t widest = passes;
    for (int call = 0; seq && call < calls; call++)
        if (seq[call] > widest)
            widest = seq[call];

    for (int d = 0; d < n; d++) {
        make(&ctx[d], &ms[d], groups[d], first, 0);
        first += groups[d];
        ms[d].iterations = iters;
        expect(vb_dev_ctx_set_overlap(&ctx[d], chunks[d], 0) == 0,
               "%s: set_overlap on device %d", what, d);
        pass_value(&ctx[d], &ms[d], iters, want);
    }

    vb_dev_pipe_begin(ctx, n, iters);
    uint64_t (*out)[VB_MAX_DIGEST_WORDS] = calloc(widest, sizeof *out);
    for (int call = 0; call < calls; call++) {
        uint64_t kn = 0, tn = 0, now = seq ? seq[call] : passes;
        most = 0;
        for (int d = 0; d < n; d++)
            if ((uint64_t) ctx[d].pipe_chunks * now * 1000 > most)
                most = (uint64_t) ctx[d].pipe_chunks * now * 1000;
        memset(out, 0, widest * sizeof *out);
        int rc = vb_dev_pipe_next(ctx, n, now, out, &kn, &tn);
        expect(rc == 0, "%s: call %d failed: %s", what, call, ctx[0].error);
        for (uint64_t p = 0; p < now; p++)
            expect(out[p][0] == want[0] && out[p][1] == want[1],
                   "%s: call %d pass %llu has the wrong checksum", what, call,
                   (unsigned long long) p);
        expect(kn == most, "%s: call %d reported %llu ns hashing, want %llu", what,
               call, (unsigned long long) kn, (unsigned long long) most);
        /* The next chunks are queued before the call returns, so the device
           hashes while the caller reads the clock. */
        for (int d = 0; d < n; d++)
            expect(ctx[d].pipe_queued - ctx[d].pipe_collected == VB_PIPE_RING - 1,
                   "%s: call %d left device %d with %llu chunks queued ahead, "
                   "want %d", what, call, d,
                   (unsigned long long) (ctx[d].pipe_queued - ctx[d].pipe_collected),
                   VB_PIPE_RING - 1);
    }
    free(out);

    vb_dev_pipe_end(ctx, n);
    for (int d = 0; d < n; d++) {
        expect(!ctx[d].pipe_running && ctx[d].pipe_queued == ctx[d].pipe_collected,
               "%s: device %d not drained by end", what, d);
        expect(ms[d].queued == ctx[d].pipe_queued &&
               ms[d].collected == ctx[d].pipe_collected,
               "%s: device %d's counts disagree with the layer's", what, d);
    }
    report_violations(what, ctx, ms, n);
    for (int d = 0; d < n; d++)
        vb_dev_ctx_free(&ctx[d]);
    report_violations(what, ctx, ms, n);
}

static void run(const char *what, int n, const uint64_t *groups,
                const unsigned *chunks, int calls, uint64_t passes)
{
    run_seq(what, n, groups, chunks, calls, passes, NULL);
}

/* A chunk that hashes wrongly spoils its own pass and no other, and the
   caller finds it in the slot of that pass. Calls of 1 then 3 passes at 3
   chunks each: chunk 4 is the second pass overall, slot 0 of the second call. */
static void corrupt(void)
{
    vb_dev_ctx c;
    mock m;
    const uint32_t iters = 3;
    make(&c, &m, 13, 0, 0);
    m.iterations = iters;
    m.corrupt_at = 4;
    vb_dev_ctx_set_overlap(&c, 3, 0);
    uint64_t want[VB_MAX_DIGEST_WORDS] = { 0 };
    pass_value(&c, &m, iters, want);

    vb_dev_pipe_begin(&c, 1, iters);
    static const uint64_t sizes[] = { 1, 3, 2 };
    uint64_t out[3][VB_MAX_DIGEST_WORDS], kn, tn;
    for (int call = 0; call < 3; call++) {
        memset(out, 0, sizeof out);
        vb_dev_pipe_next(&c, 1, sizes[call], out, &kn, &tn);
        for (uint64_t p = 0; p < sizes[call]; p++) {
            int ok = out[p][0] == want[0] && out[p][1] == want[1];
            int bad_here = call == 1 && p == 0;
            expect(bad_here ? !ok : ok, "a wrong chunk in the second pass: call %d "
                   "slot %llu is %s", call, (unsigned long long) p, ok ? "right" : "wrong");
        }
    }
    vb_dev_pipe_end(&c, 1);
    report_violations("corrupt", &c, &m, 1);
    vb_dev_ctx_free(&c);
}

/* A failed upload or wait fails the call, with the backend's own reason,
   and the pipeline still ends and frees without hanging. */
static void failure(int on_wait, int64_t at)
{
    vb_dev_ctx c;
    mock m;
    make(&c, &m, 13, 0, 0);
    m.iterations = 1;
    if (on_wait)
        m.fail_wait_at = at;
    else
        m.fail_enqueue_at = at;
    vb_dev_ctx_set_overlap(&c, 3, 0);

    vb_dev_pipe_begin(&c, 1, 1);
    uint64_t out[4][VB_MAX_DIGEST_WORDS], kn, tn;
    int rc = 0;
    for (int call = 0; call < 3 && rc == 0; call++)
        rc = vb_dev_pipe_next(&c, 1, 4, out, &kn, &tn);
    expect(rc == -1, "a %s failing at chunk %lld fails the call", on_wait ? "wait" : "upload",
           (long long) at);
    expect(strstr(c.error, on_wait ? "waiting for chunk" : "upload of chunk") != NULL,
           "the %s failure is reported as the backend said it: \"%s\"",
           on_wait ? "wait" : "upload", c.error);
    vb_dev_pipe_end(&c, 1);
    expect(!c.pipe_running, "the pipeline ends after a failed %s", on_wait ? "wait" : "upload");
    report_violations("failure", &c, &m, 1);
    vb_dev_ctx_free(&c);
}

/* Left to choose, set_overlap takes the chunk count vb_dev_auto_chunks
   gives, and the pipeline runs with it. */
static void auto_chunks(void)
{
    vb_dev_ctx c;
    mock m;
    make(&c, &m, 13, 0, 16ull << 20);  /* 1 MiB groups, a quarter of 16 MiB */
    m.iterations = 1;
    expect(vb_dev_ctx_set_overlap(&c, 0, 0) == 0, "set_overlap choosing its own count");
    unsigned want = vb_dev_auto_chunks(13, 1 << 20, 16ull << 20);
    expect(c.pipe_chunks == want && want > 1,
           "the chosen count is vb_dev_auto_chunks's, and splits: %u, want %u",
           c.pipe_chunks, want);
    uint64_t sum[VB_MAX_DIGEST_WORDS] = { 0 }, out[3][VB_MAX_DIGEST_WORDS], kn, tn;
    pass_value(&c, &m, 1, sum);
    vb_dev_pipe_begin(&c, 1, 1);
    memset(out, 0, sizeof out);
    expect(vb_dev_pipe_next(&c, 1, 3, out, &kn, &tn) == 0 &&
           out[0][0] == sum[0] && out[2][1] == sum[1],
           "passes at the chosen count are right");
    vb_dev_pipe_end(&c, 1);
    report_violations("auto", &c, &m, 1);
    vb_dev_ctx_free(&c);
}

int main(void)
{
    printf("Overlap pipeline scheduling, against a mock device\n");

    /* One device: one chunk, a count that does not divide the groups, more
       chunks per pass than the readback ring, one group each, and more asked
       for than there are groups (clamped). Calls of one pass, two, and five. */
    static const unsigned counts[] = { 1, 3, 8, 13, 1000 };
    static const uint64_t per_call[] = { 1, 2, 5 };
    for (size_t i = 0; i < sizeof counts / sizeof counts[0]; i++)
        for (size_t j = 0; j < sizeof per_call / sizeof per_call[0]; j++) {
            char what[64];
            uint64_t g = 13;
            snprintf(what, sizeof what, "1 device, %u chunks, %llu passes a call",
                     counts[i], (unsigned long long) per_call[j]);
            run(what, 1, &g, &counts[i], 4, per_call[j]);
        }

    /* Several devices at once, each with its own slice and chunk count --
       one of them a single group, so its chunk request clamps to one. */
    {
        static const uint64_t g3[] = { 13, 7, 1 };
        static const unsigned c3[] = { 3, 1, 4 };
        run("3 devices, uneven", 3, g3, c3, 4, 3);
        static const uint64_t g4[] = { 64, 64, 64, 64 };
        static const unsigned c4[] = { 1, 1, 1, 1 };
        run("4 devices, one chunk each", 4, g4, c4, 5, 2);
        static const unsigned c4b[] = { 16, 16, 16, 16 };
        run("4 devices, 16 chunks each", 4, g4, c4b, 3, 2);
    }

    /* Calls of different sizes, as calibration, warm-up and the timed
       samples make them. */
    {
        static const uint64_t sizes[] = { 1, 3, 2, 5, 1, 4 };
        uint64_t g = 13;
        static const unsigned c1[] = { 3 };
        run_seq("1 device, calls of 1, 3, 2, 5, 1, 4 passes", 1, &g, c1, 6, 0, sizes);
        static const uint64_t g3[] = { 13, 7, 1 };
        static const unsigned c3[] = { 3, 1, 4 };
        run_seq("3 devices, calls of 1, 3, 2, 5, 1, 4 passes", 3, g3, c3, 6, 0, sizes);
    }

    corrupt();
    failure(1, 0);                      /* the very first wait */
    failure(1, 5);                      /* mid-pass, chunks still queued */
    failure(0, 0);                      /* the very first upload */
    failure(0, 19);                     /* the second call's first upload:
                                           the first call queues through 18 */
    auto_chunks();

    printf("  %s    pipeline  (%d checks)\n", failures ? "FAIL" : "ok", checks);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
