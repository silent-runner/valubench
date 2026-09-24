/*
 * main.c -- command line entry point.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * One command, no configuration, a number. Everything is discovered or
 * autotuned; the flags exist so a result can be reproduced exactly, not so it
 * can be obtained at all (docs/research.md 1.5).
 */

#include "bench.h"
#include "valubench.h"
#include "report.h"
#include "sysinfo.h"
#include "opencl.h"

#include <errno.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *f, const char *argv0)
{
    fprintf(f,
"Usage: %s [options]\n"
"\n"
"  Integer SIMD microbenchmark. One hash is N chained full hashes of an\n"
"  L-byte message. Message length and iteration count are the two axes of\n"
"  the throughput surface; working set is the memory axis.\n"
"\n"
"Options:\n"
"  --algorithm NAME     md5 (default), sha1, or sha512\n"
"  --json               emit machine-readable JSON instead of a report.\n"
"                       With --list or --list-devices it describes the\n"
"                       binary rather than a result: algorithms, kernels,\n"
"                       limits, defaults, exit codes and devices.\n"
"  --list               list kernels and whether they run on this machine\n"
"  --list-devices       list OpenCL devices, or say why there are none\n"
"  --device LIST        OpenCL devices to use: an index, a comma-separated\n"
"                       list, or 'all' (the default). Several devices run\n"
"                       concurrently over slices of the same corpus.\n"
"  --kernel NAME        force a kernel (default: autotune)\n"
"  --where WHICH        restrict autotune to 'cpu' or 'device' kernels\n"
"                       ('any' is the default). A CPU baseline on a machine\n"
"                       with a GPU needs this: otherwise the device kernel\n"
"                       wins the probe and the result is not a CPU number.\n"
"  --threads N          worker threads (default: one per allowed CPU, %u here)\n"
"  --message-bytes L    message length (default %u, range %u..%u). Raises\n"
"                       both compute and bytes read per hash.\n"
"  --iterations N       chained hashes per hash (default 1, max %u). Raises\n"
"                       compute per hash without touching more memory.\n"
"                       Requires --message-bytes >= the digest size.\n"
"  --transfer MODE      how the corpus reaches an OpenCL device:\n"
"                       'resident' (default) uploads it once and launches\n"
"                       against it; 'stream' re-uploads before every launch,\n"
"                       putting the PCIe link inside the timed region. Use\n"
"                       stream with a large --working-set-kb to find where\n"
"                       compute overtakes transfer. No effect on CPU kernels.\n"
"  --working-set-kb K   target corpus size (default 1024). Sets how many\n"
"                       messages are hashed, so sweeping it walks the result\n"
"                       from L1-resident to DRAM-bound.\n"
"  --expect HEX         verify against this checksum instead of computing\n"
"                       one. The expected value depends only on algorithm,\n"
"                       message size, iterations and corpus range, never on\n"
"                       the machine, so a sweep can solve a whole iteration\n"
"                       ladder once and hand each point its answer.\n"
"  --reference-ladder L ascending iteration counts, comma-separated. Emits\n"
"                       the expected checksum for each as JSON, computed in\n"
"                       one pass: the digest after k iterations is a prefix\n"
"                       of the chain for any larger k, so a ladder costs one\n"
"                       walk rather than one walk per rung.\n"
"  --samples N          timed iterations (default 10, max %d)\n"
"  --time-ms N          wall time per iteration (default 100)\n"
"  --warmup-ms N        warm-up before timing (default 300)\n"
"  --no-pin             do not pin worker threads to cores\n"
"  --verbose            show autotune probes\n"
"  --version            print version\n"
"  -h, --help           this text\n"
"\n"
"Exit status: 0 success, 1 verification failure, 2 usage error,\n"
"             3 result too noisy to trust.\n",
            argv0, vb_default_threads(),
            VB_DEFAULT_MSG_BYTES, VB_MIN_MSG_BYTES, VB_MAX_MSG_BYTES,
            VB_MAX_ITERS, VB_MAX_SAMPLES);
}

static void list_kernels(void)
{
    size_t count;
    const vb_kernel *ks = vb_kernels(&count);

    printf("%-16s %-8s %6s %8s %7s  %s\n",
           "NAME", "ISA", "LANES", "STREAMS", "WHERE", "AVAILABLE");
    for (size_t i = 0; i < count; i++) {
        printf("%-16s %-8s %6u %8u %7s  %s\n",
               ks[i].name, ks[i].isa, ks[i].lanes, ks[i].streams,
               ks[i].device ? "device" : "cpu",
               ks[i].available() ? "yes" : "no");
    }
}

/* "all", or a comma-separated list of indices. */
static int parse_devices(const char *spec, vb_config *cfg)
{
    cfg->device_count = 0;

    if (!strcmp(spec, "all"))
        return 0;                       /* 0 means every device */

    for (const char *p = spec; *p; ) {
        char *end;

        errno = 0;
        long v = strtol(p, &end, 10);

        if (end == p) {
            fprintf(stderr, "valubench: --device wants indices or 'all', "
                            "got '%s'\n", spec);
            return -1;
        }
        if (errno == ERANGE || v < 0 || v > VB_OCL_MAX_DEVICES - 1) {
            fprintf(stderr, "valubench: --device index %ld is out of range "
                            "(0..%d)\n", v, VB_OCL_MAX_DEVICES - 1);
            return -1;
        }
        /* Refuse rather than truncate. Silently dropping the tail of a device
           list would measure something other than what was asked for. */
        if (cfg->device_count >= VB_OCL_MAX_DEVICES) {
            fprintf(stderr, "valubench: --device takes at most %d indices\n",
                    VB_OCL_MAX_DEVICES);
            return -1;
        }
        /* A repeated index would give one physical device two slices of the
           corpus and count it twice in the aggregate. */
        for (int i = 0; i < cfg->device_count; i++) {
            if (cfg->device_index[i] == (int) v) {
                fprintf(stderr, "valubench: --device lists device %ld twice\n",
                        v);
                return -1;
            }
        }
        cfg->device_index[cfg->device_count++] = (int) v;

        p = end;
        while (*p == ',' || *p == ' ')
            p++;
    }
    return 0;
}

static void list_devices(void)
{
    vb_ocl_device d[VB_OCL_MAX_DEVICES];
    int n = vb_ocl_devices(d, VB_OCL_MAX_DEVICES);

    if (n <= 0) {
        const char *why = vb_ocl_error();
        printf("No OpenCL devices.\n");
        if (why)
            printf("  %s\n", why);
        printf("  This is not an error: valubench runs CPU-only without them.\n");
        return;
    }

    for (int i = 0; i < n; i++) {
        printf("[%d] %s\n", i, d[i].name);
        printf("     vendor    %s\n", d[i].vendor);
        printf("     type      %s, %u compute units @ %u MHz\n",
               vb_ocl_type_name(d[i].type), d[i].compute_units, d[i].clock_mhz);
        printf("     memory    %llu MiB global, %llu MiB max allocation\n",
               (unsigned long long) (d[i].global_mem >> 20),
               (unsigned long long) (d[i].max_alloc >> 20));
        printf("     max wg    %zu\n", d[i].max_work_group);
        printf("     platform  %s (%s)\n", d[i].platform_name,
               d[i].platform_version);
        printf("     driver    %s, device %s\n", d[i].driver_version,
               d[i].device_version);
    }
}

static int need_arg(int i, int argc, const char *flag)
{
    if (i + 1 >= argc) {
        fprintf(stderr, "valubench: %s requires an argument\n", flag);
        return 0;
    }
    return 1;
}

/*
 * Parse an unsigned option argument, or fail.
 *
 * atoi() was used here and has no error return: a non-numeric argument becomes
 * 0 and a partly-numeric one is truncated, so `--threads abc` ran on one thread
 * and `--message-bytes 12x` measured twelve bytes. Neither was rejected, and the
 * JSON then recorded the substituted value as though it had been asked for --
 * which turns a typo in a sweep script into a result that looks deliberate.
 *
 * Rejects: empty strings, anything with trailing characters, negatives, and
 * values outside [lo, hi]. Leading whitespace is allowed because strtoul allows
 * it and a shell can introduce it.
 */
static int parse_uint(const char *flag, const char *arg,
                      unsigned long lo, unsigned long hi, unsigned *out)
{
    char *end = NULL;

    if (!arg || !*arg) {
        fprintf(stderr, "valubench: %s needs a number\n", flag);
        return 0;
    }
    /* strtoul happily wraps a negative into a huge unsigned; catch the sign
       before it can. */
    for (const char *p = arg; *p; p++) {
        if (*p == '-') {
            fprintf(stderr, "valubench: %s must not be negative (got '%s')\n",
                    flag, arg);
            return 0;
        }
        if (!isspace((unsigned char) *p))
            break;
    }

    errno = 0;
    unsigned long v = strtoul(arg, &end, 10);

    if (end == arg || (end && *end)) {
        fprintf(stderr, "valubench: %s wants a number, got '%s'\n", flag, arg);
        return 0;
    }
    if (errno == ERANGE || v < lo || v > hi) {
        fprintf(stderr, "valubench: %s must be between %lu and %lu (got '%s')\n",
                flag, lo, hi, arg);
        return 0;
    }
    *out = (unsigned) v;
    return 1;
}

/*
 * A checksum supplied on the command line, in the same hex the tool prints:
 * digest_words words, most significant first, each rendered at its natural
 * width. Parsed against the selected algorithm, so --expect and --algorithm
 * disagreeing is caught here rather than showing up as a verification failure.
 */
static int parse_expect(const char *hex, vb_config *cfg)
{
    unsigned nib = cfg->alg->word_bytes * 2;
    size_t want = (size_t) cfg->alg->digest_words * nib;

    if (strspn(hex, "0123456789abcdefABCDEF") != strlen(hex)
        || strlen(hex) != want) {
        fprintf(stderr, "valubench: --expect wants %zu hex digits for %s "
                        "(got %zu)\n", want, cfg->alg->name, strlen(hex));
        return 0;
    }

    memset(cfg->expected, 0, sizeof cfg->expected);
    for (unsigned i = 0; i < cfg->alg->digest_words; i++) {
        char word[17];
        memcpy(word, hex + (size_t) i * nib, nib);
        word[nib] = 0;
        cfg->expected[i] = strtoull(word, NULL, 16);
    }
    cfg->have_expected = 1;
    return 1;
}

/*
 * An ascending list of iteration counts, for --reference-ladder.
 *
 * Ascending is a requirement of the checkpointed walk rather than a
 * convenience: it snapshots as it passes each count, so an out-of-order entry
 * would be silently skipped. Rejecting it here is the difference between a
 * usage error and a wrong answer.
 */
static int parse_ladder(const char *arg, uint32_t *out, unsigned max,
                        unsigned *n_out)
{
    unsigned n = 0;

    for (const char *p = arg; *p; ) {
        char *end;
        errno = 0;
        unsigned long v = strtoul(p, &end, 10);

        /* The same ceiling --iterations enforces. A ladder exists to
           precompute checksums a later --expect run consumes, so a rung the
           benchmark will refuse to run is a rung nobody can use. */
        if (end == p || errno == ERANGE || v < 1 || v > VB_MAX_ITERS) {
            fprintf(stderr, "valubench: --reference-ladder wants iteration "
                            "counts between 1 and %u, got '%s'\n",
                    VB_MAX_ITERS, p);
            return 0;
        }
        if (n == max) {
            fprintf(stderr, "valubench: --reference-ladder takes at most %u "
                            "counts\n", max);
            return 0;
        }
        if (n && v <= out[n - 1]) {
            fprintf(stderr, "valubench: --reference-ladder must ascend "
                            "(%lu after %u)\n", v, out[n - 1]);
            return 0;
        }
        out[n++] = (uint32_t) v;

        p = end;
        if (*p == ',') p++;
        else if (*p)   { fprintf(stderr, "valubench: --reference-ladder wants "
                                         "a comma-separated list, got '%s'\n",
                                 p); return 0; }
    }

    if (n == 0) {
        fprintf(stderr, "valubench: --reference-ladder is empty\n");
        return 0;
    }
    *n_out = n;
    return 1;
}

/*
 * Emit the expected checksums for a whole iteration ladder.
 *
 * This is the caller the checkpointed reference was written for. A crossover
 * sweep asks for the same chain over and over -- the digest after k iterations
 * is a prefix of the chain for any larger k -- so one walk to the largest
 * count, snapshotting as it goes, replaces one walk per rung. Split across the
 * cores as well and a pass that used to dominate a session disappears into it.
 *
 * The corpus identity travels with the answers, because a checksum is only
 * meaningful for the exact range it was computed over: a consumer that reuses
 * these values under a different algorithm, message size or working set would
 * be verifying against the wrong truth. sweep.py keys its cache on them.
 */
static void emit_reference_ladder(const vb_config *cfg, const uint32_t *iters,
                                  unsigned n)
{
    uint64_t count = vb_batch_messages(cfg);
    uint64_t (*out)[VB_MAX_DIGEST_WORDS] = calloc(n, sizeof *out);

    if (!out) {
        fprintf(stderr, "valubench: out of memory\n");
        return;
    }

    vb_reference_checksums_mt(cfg->alg, 0, count, cfg->message_bytes,
                              iters, n, vb_default_threads(), out);

    printf("{\n");
    printf("  \"schema\": \"valubench/reference/1\",\n");
    printf("  \"algorithm\": \"%s\",\n", cfg->alg->name);
    printf("  \"message_bytes\": %u,\n", cfg->message_bytes);
    printf("  \"working_set_kb\": %u,\n", cfg->working_set_kb);
    printf("  \"start_index\": 0,\n");
    printf("  \"count\": %llu,\n", (unsigned long long) count);
    printf("  \"checksums\": [\n");
    for (unsigned k = 0; k < n; k++) {
        printf("    { \"iterations\": %u, \"checksum\": \"", iters[k]);
        for (unsigned i = 0; i < cfg->alg->digest_words; i++) {
            if (cfg->alg->word_bytes == 8)
                printf("%016llx", (unsigned long long) out[k][i]);
            else
                printf("%08x", (unsigned) out[k][i]);
        }
        printf("\" }%s\n", k + 1 < n ? "," : "");
    }
    printf("  ]\n}\n");
    free(out);
}

/*
 * Listing is an action rather than a run, but it cannot happen while the
 * command line is still being read: `--list --json` and `--json --list` have to
 * mean the same thing, and they did not when --list returned from inside the
 * parse loop. So the loop records what was asked for and the dispatch happens
 * once everything has been seen.
 */
typedef enum {
    ACT_RUN = 0,
    ACT_LIST_KERNELS,
    ACT_LIST_DEVICES,
    ACT_REFERENCE
} vb_action;

int main(int argc, char **argv)
{
    vb_config cfg;
    int as_json = 0, verbose = 0;
    vb_action action = ACT_RUN;
    /* Both are resolved after the loop, for the same reason --list is: they
       depend on --algorithm, which may appear either side of them. */
    const char *expect_arg = NULL, *ladder_arg = NULL;
    uint32_t ladder[VB_MAX_LADDER];
    unsigned n_ladder = 0;

    vb_config_defaults(&cfg);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];

        if (!strcmp(a, "--json")) {
            as_json = 1;
        } else if (!strcmp(a, "--verbose")) {
            verbose = 1;
        } else if (!strcmp(a, "--no-pin")) {
            cfg.pin_cpu = 0;
        } else if (!strcmp(a, "--list")) {
            action = ACT_LIST_KERNELS;
        } else if (!strcmp(a, "--list-devices")) {
            action = ACT_LIST_DEVICES;
        } else if (!strcmp(a, "--device")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (parse_devices(argv[++i], &cfg) != 0) return VB_EXIT_USAGE;
        } else if (!strcmp(a, "--version")) {
            char wid[64];
            printf("valubench %s (workload %s)\n", VB_VERSION,
                   vb_workload_id(wid, sizeof wid, cfg.alg,
                                  cfg.message_bytes, cfg.iterations));
            return VB_EXIT_OK;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout, argv[0]);
            return VB_EXIT_OK;
        } else if (!strcmp(a, "--algorithm") || !strcmp(a, "--alg")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            cfg.alg = vb_algorithm_by_name(argv[++i]);
            if (!cfg.alg) {
                fprintf(stderr, "valubench: unknown algorithm '%s' "
                                "(md5, sha1, sha512)\n", argv[i]);
                return VB_EXIT_USAGE;
            }
        } else if (!strcmp(a, "--transfer")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            const char *m = argv[++i];
            if (!strcmp(m, "resident")) {
                cfg.transfer = VB_TRANSFER_RESIDENT;
            } else if (!strcmp(m, "stream")) {
                cfg.transfer = VB_TRANSFER_STREAM;
            } else {
                fprintf(stderr, "valubench: unknown transfer mode '%s' "
                                "(resident, stream)\n", m);
                return VB_EXIT_USAGE;
            }
        } else if (!strcmp(a, "--where")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            const char *w = argv[++i];
            if (!strcmp(w, "any")) {
                cfg.where = VB_WHERE_ANY;
            } else if (!strcmp(w, "cpu")) {
                cfg.where = VB_WHERE_CPU;
            } else if (!strcmp(w, "device")) {
                cfg.where = VB_WHERE_DEVICE;
            } else {
                fprintf(stderr, "valubench: unknown --where '%s' "
                                "(any, cpu, device)\n", w);
                return VB_EXIT_USAGE;
            }
        } else if (!strcmp(a, "--kernel")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            cfg.force_kernel = argv[++i];
        } else if (!strcmp(a, "--threads")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (!parse_uint(a, argv[++i], 1, VB_MAX_THREADS, &cfg.threads))
                return VB_EXIT_USAGE;
        } else if (!strcmp(a, "--expect")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            expect_arg = argv[++i];
        } else if (!strcmp(a, "--reference-ladder")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            ladder_arg = argv[++i];
            action = ACT_REFERENCE;
        } else if (!strcmp(a, "--iterations")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (!parse_uint(a, argv[++i], 1, VB_MAX_ITERS, &cfg.iterations))
                return VB_EXIT_USAGE;
        } else if (!strcmp(a, "--message-bytes")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (!parse_uint(a, argv[++i], 1, 1u << 20, &cfg.message_bytes))
                return VB_EXIT_USAGE;
        } else if (!strcmp(a, "--working-set-kb")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (!parse_uint(a, argv[++i], 1, 1u << 24, &cfg.working_set_kb))
                return VB_EXIT_USAGE;
        } else if (!strcmp(a, "--samples")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (!parse_uint(a, argv[++i], 1, VB_MAX_SAMPLES, &cfg.n_samples))
                return VB_EXIT_USAGE;
        } else if (!strcmp(a, "--time-ms")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (!parse_uint(a, argv[++i], 1, 3600000, &cfg.target_ms))
                return VB_EXIT_USAGE;
        } else if (!strcmp(a, "--warmup-ms")) {
            if (!need_arg(i, argc, a)) return VB_EXIT_USAGE;
            if (!parse_uint(a, argv[++i], 0, 3600000, &cfg.warmup_ms))
                return VB_EXIT_USAGE;
        } else {
            fprintf(stderr, "valubench: unknown option '%s'\n", a);
            usage(stderr, argv[0]);
            return VB_EXIT_USAGE;
        }
    }

    /* Listing describes the binary, so it answers before the run parameters
       are validated -- asking what exists must work even with a nonsensical
       --iterations on the same line. */
    switch (action) {
    case ACT_LIST_KERNELS:
        if (as_json)
            vb_report_capabilities_json(stdout);
        else
            list_kernels();
        return VB_EXIT_OK;
    case ACT_LIST_DEVICES:
        if (as_json)
            vb_report_devices_json(stdout);
        else
            list_devices();
        return VB_EXIT_OK;
    case ACT_REFERENCE:
    case ACT_RUN:
        break;
    }

    /*
     * Forcing a kernel also fixes the algorithm: they are not separable. So the
     * kernel is found before anything that is validated against the algorithm.
     * It used to be found after, which checked --message-bytes and --expect
     * against the default MD5 and then ran SHA-512: `--kernel sha512/scalar-s1
     * --iterations 2` passed the digest-fits-message guard below and the oracle
     * wrote a 64-byte digest into a 55-byte message.
     */
    const vb_kernel *k = NULL;

    if (cfg.force_kernel) {
        size_t count;
        const vb_kernel *ks = vb_kernels(&count);
        for (size_t i = 0; i < count; i++) {
            if (!strcmp(ks[i].name, cfg.force_kernel)) {
                k = &ks[i];
                break;
            }
        }
        if (!k) {
            fprintf(stderr, "valubench: no kernel named '%s' (try --list)\n",
                    cfg.force_kernel);
            return VB_EXIT_USAGE;
        }
        cfg.alg = vb_algorithm_by_id(k->alg);
    }

    if (cfg.n_samples < 1 || cfg.n_samples > VB_MAX_SAMPLES) {
        fprintf(stderr, "valubench: --samples must be 1..%d\n", VB_MAX_SAMPLES);
        return VB_EXIT_USAGE;
    }

    if (cfg.iterations < 1 || cfg.iterations > VB_MAX_ITERS) {
        fprintf(stderr, "valubench: --iterations must be 1..%u\n",
                VB_MAX_ITERS);
        return VB_EXIT_USAGE;
    }

    if (cfg.message_bytes < VB_MIN_MSG_BYTES ||
        cfg.message_bytes > VB_MAX_MSG_BYTES) {
        fprintf(stderr, "valubench: --message-bytes must be %u..%u\n",
                VB_MIN_MSG_BYTES, VB_MAX_MSG_BYTES);
        return VB_EXIT_USAGE;
    }

    /*
     * Iterated hashing overwrites the leading 16 bytes with the digest, which
     * needs a message at least that long. Refuse rather than silently changing
     * what is being measured.
     */
    /* Resolved here rather than in the loop: interpreted against the algorithm,
       and --algorithm may come after it on the command line. Before the guard
       below rather than after, because that guard needs the top rung. */
    if (ladder_arg && !parse_ladder(ladder_arg, ladder, VB_MAX_LADDER,
                                    &n_ladder))
        return VB_EXIT_USAGE;

    /*
     * The largest iteration count this run will actually reach, which is not
     * always cfg.iterations: --reference-ladder walks to its top rung without
     * touching that field. Guarding on cfg.iterations alone let the ladder
     * reach vb_reference_checksums() with a message shorter than the digest,
     * where the feedback memcpy writes the digest over the head of the message
     * and past the end of its allocation -- 9 bytes for SHA-512 at the default
     * 55-byte message, confirmed by AddressSanitizer. A heap overflow inside
     * the correctness oracle.
     */
    uint32_t max_iterations = cfg.iterations;
    for (unsigned li = 0; li < (unsigned) n_ladder; li++)
        if (ladder[li] > max_iterations)
            max_iterations = ladder[li];

    if (max_iterations > 1 &&
        cfg.message_bytes < vb_alg_min_iter_bytes(cfg.alg)) {
        fprintf(stderr,
"valubench: %s with %s needs --message-bytes >= %u.\n"
"  Each iteration feeds the %u-byte digest back over the head of the\n"
"  message, so a shorter message has nowhere to put it.\n",
                n_ladder ? "a reference ladder above 1" : "--iterations > 1",
                cfg.alg->name, vb_alg_min_iter_bytes(cfg.alg),
                cfg.alg->digest_bytes);
        return VB_EXIT_USAGE;
    }

    if (cfg.working_set_kb < 1) {
        fprintf(stderr, "valubench: --working-set-kb must be >= 1\n");
        return VB_EXIT_USAGE;
    }

    /*
     * The corpus must not repeat a message. Repeated digests cancel under XOR,
     * so a repeated message is one the fingerprint cannot see: 1536 one-byte
     * messages are 256 values six times over, the checksum was all zeros, and
     * a kernel returning nothing would have verified. There is no generator
     * that fixes this -- a one-byte message has 256 values -- so refuse.
     */
    if (vb_batch_messages(&cfg) > vb_distinct_messages(cfg.message_bytes)) {
        fprintf(stderr,
"valubench: --message-bytes %u allows %llu distinct messages, and this\n"
"  working set needs %llu. Repeated messages cancel in the XOR checksum, so\n"
"  verification could not see them. Lower --working-set-kb or lengthen the\n"
"  message.\n",
                cfg.message_bytes,
                (unsigned long long) vb_distinct_messages(cfg.message_bytes),
                (unsigned long long) vb_batch_messages(&cfg));
        return VB_EXIT_USAGE;
    }

    if (cfg.threads > VB_MAX_THREADS) {
        fprintf(stderr, "valubench: --threads must be 0..%d\n", VB_MAX_THREADS);
        return VB_EXIT_USAGE;
    }

    if (expect_arg && !parse_expect(expect_arg, &cfg))
        return VB_EXIT_USAGE;

    if (action == ACT_REFERENCE) {
        emit_reference_ladder(&cfg, ladder, n_ladder);
        return VB_EXIT_OK;
    }

    vb_sysinfo si;
    vb_sysinfo_collect(&si);

    /* Autotune output is progress, not result: keep it off stdout in JSON mode
       so the JSON stays parseable without filtering. */
    if (k) {
        if (!k->available()) {
            fprintf(stderr,
                    "valubench: kernel '%s' needs %s, which this CPU lacks\n",
                    k->name, k->isa);
            return VB_EXIT_USAGE;
        }
        /*
         * Every kernel's lanes x streams must divide the batch, or it cannot
         * cover the corpus and the checksum would be over a different set of
         * messages than every other kernel's.
         *
         * Only reachable with a vector-length-agnostic ISA, where lanes are
         * not known when the table is written: 12 lanes at three streams is
         * 36, and 36 does not divide 768. Autotune skips such a kernel
         * silently, which is right, but naming one explicitly used to report
         * "the hardware did not compute correct digests" -- blaming the
         * silicon for what is a property of the vector length.
         */
        if (!vb_batch_divides(k)) {
            fprintf(stderr,
                    "valubench: kernel '%s' cannot run at this vector length: "
                    "%u lanes x %u streams = %u, which does not divide the "
                    "%u-message batch. Try a different stream count.\n",
                    k->name, k->lanes, k->streams, k->lanes * k->streams,
                    VB_BATCH_LCM);
            return VB_EXIT_USAGE;
        }
        /* Naming a kernel and then excluding where it runs is a contradiction,
           and silently honouring one over the other would mislabel the result.
        */
        if ((cfg.where == VB_WHERE_CPU && k->device) ||
            (cfg.where == VB_WHERE_DEVICE && !k->device)) {
            fprintf(stderr, "valubench: --kernel '%s' runs on the %s, which "
                            "--where %s excludes\n", k->name,
                    k->device ? "device" : "cpu",
                    cfg.where == VB_WHERE_CPU ? "cpu" : "device");
            return VB_EXIT_USAGE;
        }
    } else {
        k = vb_autotune(&cfg, verbose && !as_json);
        if (!k) {
            if (cfg.where != VB_WHERE_ANY) {
                fprintf(stderr,
                        "valubench: no %s kernel is available for %s on this "
                        "machine\n",
                        cfg.where == VB_WHERE_CPU ? "cpu" : "device",
                        cfg.alg->name);
                return VB_EXIT_USAGE;
            }
            fprintf(stderr,
                    "valubench: no kernel passed verification on this machine\n");
            return VB_EXIT_VERIFY_FAILED;
        }
    }

    vb_result r;
    if (vb_measure(k, &cfg, &r) != 0) {
        /* A device that could not be set up at all is a configuration problem,
           not the hardware computing wrong answers. Say which. */
        if (r.device_error[0] && !r.n_samples) {
            fprintf(stderr, "valubench: device kernel '%s' could not run:\n"
                            "  %s\n", k->name, r.device_error);
            return VB_EXIT_USAGE;
        }
        fprintf(stderr,
"valubench: VERIFICATION FAILED for kernel '%s'.\n"
"  The hardware did not compute correct %s digests. No performance number\n"
"  is reported, because a fast wrong answer is not a result. Causes worth\n"
"  checking: overclocking, marginal cooling, unstable memory, or a compiler\n"
"  bug. Run '%s --algorithm %s --kernel %s/scalar-s1 --threads 1' to test\n"
"  the portable path.\n",
                k->name, cfg.alg->name, argv[0], cfg.alg->name,
                cfg.alg->name);
        if (r.device_error[0])
            fprintf(stderr, "  device reported: %s\n", r.device_error);
        return VB_EXIT_VERIFY_FAILED;
    }

    /* Now that the work is done, re-read the things it moves. A clock and a
       temperature taken only at startup describe a machine that has not run
       the benchmark yet. */
    vb_sysinfo_resample(&si);

    if (as_json)
        vb_report_json(stdout, &r, &si, &cfg);
    else
        vb_report_human(stdout, &r, &si, &cfg);

    if (r.cov > cfg.cov_threshold)
        return VB_EXIT_NOISY;

    return VB_EXIT_OK;
}
