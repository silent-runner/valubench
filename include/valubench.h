/*
 * valubench.h -- workload definition and kernel interface.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 *
 * THE WORKLOAD
 * ============
 *
 * Workload family "<alg>-full", parameterised by algorithm, message length and
 * iteration count. One hash is `iterations` chained full hashes of a message of
 * `message_bytes`:
 *
 *     state = message(index)                  // message_bytes bytes
 *     repeat `iterations` times:
 *         digest = MD5(state)
 *         state[0..15] = digest               // rest of the message unchanged
 *     contribute digest to the checksum
 *
 * A message of L bytes occupies ceil((L + 9) / 64) blocks once padded, and each
 * block is one compression. So:
 *
 *     compressions/sec = hashes/sec * iterations * blocks_per_message
 *
 * TWO INDEPENDENT AXES. Iteration count raises compute per hash while touching
 * no additional memory. Message length raises both compute and bytes read. Swept
 * together they trace the surface that locates the compute-bound / memory-bound
 * crossover -- the project's second goal. See docs/research.md part 3.
 *
 * WHY THE DIGEST IS FED BACK INTO THE SAME MESSAGE, rather than simply hashing
 * the 16-byte digest again: every iteration must be the *same* work. Hashing a
 * 16-byte digest would collapse to a single block with 12 of 16 words constant,
 * so later iterations would be cheaper than the first. Overwriting the first 16
 * bytes of the full message keeps the instruction mix and the block count
 * identical on every iteration, which makes `iterations` a clean linear
 * multiplier. It also means iterations > 1 requires message_bytes >= 16.
 *
 *
 * WHERE MESSAGES COME FROM
 * ========================
 *
 * From a corpus in memory, built once outside the timed region, not generated
 * on the fly. That is deliberate: generated messages would make the benchmark
 * permanently compute-bound and unable to find the memory crossover at all. A
 * real corpus means the working set grows with message count and length, and
 * walks out of L1, L2, L3 and into DRAM as it does.
 *
 * The corpus is stored lane-interleaved -- word j of a block for all L lanes is
 * contiguous -- so a kernel loads message words with plain vector loads and no
 * transpose. This is the layout every fast multi-buffer hash implementation
 * uses; doing the transpose inside the timed region would measure shuffle
 * throughput instead of hash throughput.
 *
 * Messages are stored fully padded, so kernels never deal with padding.
 *
 *
 * VERIFICATION
 * ============
 *
 * A kernel returns the XOR of every final digest it computed. XOR is
 * commutative and associative, so this checksum is invariant to how work is
 * spread across lanes, streams, threads or devices -- every implementation must
 * produce the same value for the same parameters. That one property gives us a
 * correctness test for every kernel, continuous in-run verification, and a
 * fingerprint comparable across machines and thread counts.
 *
 * The checksum is comparable within a workload id, not across one: different
 * message lengths or iteration counts are different workloads (research.md 1.4).
 *
 * See docs/research.md section 2.6.
 */

#ifndef VALUBENCH_H
#define VALUBENCH_H

#include "algorithm.h"
#include "hashes.h"

#include <stddef.h>
#include <stdint.h>

#define VB_VERSION       "0.7.0"

/*
 * Exit status. Named because they are part of the interface: any tool driving
 * the binary has to distinguish "the hardware computed the wrong answer" from
 * "you asked for something impossible" from "the machine was too noisy to
 * trust the number". tools/sweep.py used to carry its own copy of this list.
 */
#define VB_EXIT_OK            0
#define VB_EXIT_VERIFY_FAILED 1
#define VB_EXIT_USAGE         2
#define VB_EXIT_NOISY         3

#define VB_DEFAULT_MSG_BYTES 55u
#define VB_MIN_MSG_BYTES     1u
#define VB_MAX_MSG_BYTES     (1u << 20)     /* 1 MiB */
#define VB_MAX_ITERS         1000000u

/*
 * Every kernel's group size (lanes * streams) must divide the batch, so the
 * checksum always covers the same messages regardless of which kernel ran.
 *
 * CPU lane counts are 1/4/8/16 and device kernels interleave 64, with stream
 * counts 1..4 throughout. The largest group is 64*4 = 256 and 3-stream variants
 * contribute a factor of 3, so the least common multiple is 768.
 *
 * This is hardcoded rather than computed from the registry because the registry
 * varies with build configuration (an AVX-512-capable toolchain registers more
 * kernels), and a batch that changed with build flags would change the checksum
 * with them. tests/test_kernels.c verifies every registered kernel divides it,
 * so adding a kernel with an awkward group size fails loudly at test time.
 *
 * Consequence worth knowing: the batch can never be smaller than 768 messages,
 * so the smallest reachable working set is 768 * padded_message_size. At the
 * default 55 bytes that is 48 KiB, which is fine; with megabyte messages the
 * floor is correspondingly large.
 */
/*
 * Upper bound on SIMD lanes in any kernel, for the fixed-size buffers the lane
 * fold needs. A vector-length-agnostic kernel does not know its lane count
 * until it runs, so the fold cannot size its scratch by MD5K_LANES any more.
 * 64 is SVE's architectural maximum (2048-bit vectors of 32-bit words) and
 * also the device interleave, so nothing can exceed it.
 */
#define VB_MAX_LANES 64u

#define VB_BATCH_LCM 768u

/* Build message `index` of `bytes` bytes. Buffer must hold `bytes`. */
void vb_build_message(uint32_t index, uint32_t bytes, uint8_t *out);

/*
 * Full workload identifier, e.g. "md5-full-55x1" or "md5-full-4096x64". Any
 * change to the work per hash changes this string, so results from different
 * workloads are never silently compared.
 */
const char *vb_workload_id(char *buf, size_t n, const vb_algorithm *alg,
                           uint32_t message_bytes, uint32_t iterations);

/*
 * A lane-interleaved, fully padded corpus of messages.
 *
 * Layout, for L lanes and B blocks per message, message m = slot*L + lane:
 *
 *     words[((slot * B + block) * 16 + word) * L + lane]
 *
 * so the L lanes of one word sit contiguously and load as one vector.
 */
typedef struct {
    void     *words;         /* owned; 64-byte aligned; elements are
                                alg->word_bytes wide */
    size_t    n_words;
    const vb_algorithm *alg;
    uint32_t  lanes;
    uint32_t  blocks;        /* per message */
    uint32_t  message_bytes;
    uint64_t  n_messages;
    uint32_t  start_index;
} vb_corpus;

int  vb_corpus_build(vb_corpus *c, const vb_algorithm *alg, uint32_t lanes,
                     uint32_t start_index, uint64_t n_messages,
                     uint32_t message_bytes);
void vb_corpus_free(vb_corpus *c);

/* Word stride from one message slot to the next. Multiply by
   c->alg->word_bytes for the byte stride. */
static inline size_t vb_corpus_slot_words(const vb_corpus *c)
{
    return (size_t) c->blocks * VB_WORDS_PER_BLOCK * c->lanes;
}

/*
 * A kernel hashes n_groups groups of (lanes * streams) messages starting at
 * `corpus`, which points at the first message slot of its slice, and XORs the
 * final digests into checksum[4]. `iterations` is >= 1, `blocks` >= 1.
 */
typedef void (*vb_kernel_fn)(const void *corpus, uint64_t n_groups,
                             uint32_t blocks, uint32_t iterations,
                             uint64_t checksum[VB_MAX_DIGEST_WORDS]);

typedef struct {
    const char  *name;        /* e.g. "md5/avx2-s3" */
    const char  *isa;         /* e.g. "AVX2" */
    vb_alg_id    alg;         /* which hash this kernel computes */
    unsigned     lanes;       /* SIMD lanes (independent messages per vector) */
    unsigned     streams;     /* independent vector chains interleaved */
    vb_kernel_fn fn;                 /* CPU path; NULL for device kernels */
    int        (*available)(void);   /* runtime ISA / device check */

    /*
     * Device kernels run on an accelerator rather than the CPU thread pool:
     * the corpus is uploaded once and the harness drives a single context.
     * `lanes` is then work-items per corpus slot and `streams` the messages
     * each work-item hashes, so the work decomposition -- and therefore the
     * checksum -- matches the CPU kernels exactly.
     */
    int          device;

    /*
     * Set when `lanes` came from the hardware rather than the build. A group
     * size that does not tile VB_BATCH_LCM is a build error for a fixed-width
     * kernel and merely this machine's vector length for a run-time one --
     * 12 lanes x 3 streams on 384-bit SVE tiles nothing, and the kernel is
     * blameless. Callers that refuse such a kernel should say which it is.
     */
    int          lanes_runtime;
} vb_kernel;

/* Registry. Kernels whose available() returns 0 are never selected. */
const vb_kernel *vb_kernels(size_t *count);

/* Rungs a single --reference-ladder may carry. A crossover sweep is a power-of
   -two ladder, so 64 covers any range the iteration count itself permits. */
#define VB_MAX_LADDER 64

/* Reference checksum over [start, start+count), via the algorithm's scalar
   reference implementation. The _mt form splits the range across `threads` and
   XORs the parts, which is exact -- XOR is associative and commutative -- and
   is what keeps a crossover sweep from spending minutes of one core proving an
   answer before it measures anything. */
void vb_reference_checksum(const vb_algorithm *alg, uint32_t start,
                           uint64_t count, uint32_t message_bytes,
                           uint32_t iterations,
                           uint64_t checksum[VB_MAX_DIGEST_WORDS]);

/* Checksums for several iteration counts in one pass over the corpus. The
   digest after k iterations is a prefix of the chain for any larger k, so a
   ladder costs one walk to its largest count rather than one walk per rung.
   `iters` must be ascending. */
void vb_reference_checksums(const vb_algorithm *alg, uint32_t start,
                            uint64_t count, uint32_t message_bytes,
                            const uint32_t *iters, unsigned n_iters,
                            uint64_t out[][VB_MAX_DIGEST_WORDS]);

void vb_reference_checksum_mt(const vb_algorithm *alg, uint32_t start,
                              uint64_t count, uint32_t message_bytes,
                              uint32_t iterations, unsigned threads,
                              uint64_t checksum[VB_MAX_DIGEST_WORDS]);

/* Both savings at once: one pass over the corpus for the whole ladder, split
   across `threads`. A crossover sweep asks for the same chain prefix at every
   rung, so this is the difference between a reference pass that dominates the
   session and one that disappears into it. */
void vb_reference_checksums_mt(const vb_algorithm *alg, uint32_t start,
                               uint64_t count, uint32_t message_bytes,
                               const uint32_t *iters, unsigned n_iters,
                               unsigned threads,
                               uint64_t out[][VB_MAX_DIGEST_WORDS]);

#endif /* VALUBENCH_H */
