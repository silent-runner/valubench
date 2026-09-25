/*
 * report.c -- JSON and human-readable output.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Both renderers read the same structs, so the two views cannot disagree about
 * what was measured. JSON is the primary artifact:
 * nobody should ever have to scrape the human output, which is the failure mode
 * that makes so many Phoronix test profiles fragile (docs/research.md 1.1).
 */

#include "report.h"
#include "power.h"
#include "opencl.h"

#include <stdio.h>
#include <string.h>

static void json_str(FILE *f, const char *s)
{
    fputc('"', f);
    for (; *s; s++) {
        switch (*s) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f);  break;
        case '\r': fputs("\\r", f);  break;
        case '\t': fputs("\\t", f);  break;
        default:
            if ((unsigned char) *s < 0x20)
                fprintf(f, "\\u%04x", (unsigned char) *s);
            else
                fputc(*s, f);
        }
    }
    fputc('"', f);
}

static void json_kv_str(FILE *f, const char *k, const char *v, const char *tail)
{
    fprintf(f, "    ");
    json_str(f, k);
    fprintf(f, ": ");
    json_str(f, v);
    fprintf(f, "%s\n", tail);
}

void vb_report_json(FILE *f, const vb_result *r, const vb_sysinfo *si,
                    const vb_config *cfg)
{
    const char *warn = vb_sysinfo_warnings(si);
    char wid[64];

    vb_workload_id(wid, sizeof wid, r->alg, r->message_bytes, r->iterations);

    fprintf(f, "{\n");

    fprintf(f, "  \"schema\": \"valubench/result/1\",\n");
    fprintf(f, "  \"benchmark\": {\n");
    json_kv_str(f, "name", "valubench", ",");
    json_kv_str(f, "version", VB_VERSION, ",");
    json_kv_str(f, "algorithm", r->alg->name, ",");
    json_kv_str(f, "workload", wid, ",");
    /* Built from the algorithm descriptor rather than written out: these
       strings described MD5 only, and silently misreported every other
       algorithm once the abstraction landed. */
    char desc[512];
    snprintf(desc, sizeof desc,
        "One hash = `iterations` chained full %ss of a %u-byte message. Each "
        "iteration is %u compression%s: all %u steps, real message words, no "
        "constant folding, no step reversal, no early exit. The digest is fed "
        "back as the first %u message bytes so every iteration is equal work.",
        r->alg->name, r->message_bytes, r->blocks, r->blocks == 1 ? "" : "s",
        r->alg->rounds, r->alg->digest_bytes);
    json_kv_str(f, "workload_description", desc, ",");

    /* Sized with headroom: the text plus a long algorithm name overran a
       256-byte buffer and snprintf truncated it mid-word, silently. */
    char comp[320];
    snprintf(comp, sizeof comp,
        "Generic %s throughput: every step of every block runs with real "
        "message words. Not comparable to figures from implementations that "
        "take shortcuts -- constant folding, step reversal, early exit -- "
        "since those perform strictly less work per reported hash.",
        r->alg->name);
    json_kv_str(f, "comparability", comp, "");
    fprintf(f, "  },\n");

    fprintf(f, "  \"parameters\": {\n");
    fprintf(f, "    \"message_bytes\": %u,\n", r->message_bytes);
    fprintf(f, "    \"digest_bytes\": %u,\n", r->alg->digest_bytes);
    fprintf(f, "    \"block_bytes\": %u,\n", r->alg->block_bytes);
    fprintf(f, "    \"blocks_per_message\": %u,\n", r->blocks);
    fprintf(f, "    \"iterations\": %u,\n", r->iterations);
    fprintf(f, "    \"threads\": %u,\n", r->threads);
    fprintf(f, "    \"batch_messages\": %llu,\n",
            (unsigned long long) r->batch_messages);
    fprintf(f, "    \"working_set_bytes\": %llu\n",
            (unsigned long long) r->working_set_bytes);
    fprintf(f, "  },\n");

    fprintf(f, "  \"result\": {\n");
    fprintf(f, "    \"unit\": \"H/s\",\n");
    fprintf(f, "    \"direction\": \"higher_is_better\",\n");
    fprintf(f, "    \"median\": %.6g,\n", r->median);
    fprintf(f, "    \"min\": %.6g,\n", r->min);
    fprintf(f, "    \"max\": %.6g,\n", r->max);
    fprintf(f, "    \"mean\": %.6g,\n", r->mean);
    fprintf(f, "    \"stddev\": %.6g,\n", r->stddev);
    fprintf(f, "    \"cov_percent\": %.4g,\n", r->cov);
    fprintf(f, "    \"stable\": %s,\n",
            (r->cov <= cfg->cov_threshold) ? "true" : "false");
    fprintf(f, "    \"cov_threshold_percent\": %.4g,\n", cfg->cov_threshold);
    fprintf(f, "    \"message_bytes_per_second\": %.6g,\n",
            r->median * (double) r->message_bytes * (double) r->iterations);
    fprintf(f, "    \"compressions_per_second\": %.6g,\n",
            r->median * (double) r->iterations * (double) r->blocks);
    fprintf(f, "    \"samples\": [");
    for (unsigned i = 0; i < r->n_samples; i++)
        fprintf(f, "%s%.6g", i ? ", " : "", r->sample_hps[i]);
    fprintf(f, "],\n");
    fprintf(f, "    \"total_hashes\": %llu,\n",
            (unsigned long long) r->total_hashes);
    fprintf(f, "    \"total_seconds\": %.6g\n", r->total_seconds);
    fprintf(f, "  },\n");

    fprintf(f, "  \"verification\": {\n");
    fprintf(f, "    \"verified\": %s,\n", r->verified ? "true" : "false");
    fprintf(f, "    \"checksum\": \"");
    for (unsigned i = 0; i < r->alg->digest_words; i++) {
        if (r->alg->word_bytes == 8)
            fprintf(f, "%016llx", (unsigned long long) r->checksum[i]);
        else
            fprintf(f, "%08x", (unsigned) r->checksum[i]);
    }
    fprintf(f, "\",\n");
    json_kv_str(f, "method",
        "XOR of every digest computed, compared against the scalar reference "
        "before timing and re-compared on every timed iteration.",
        "");
    fprintf(f, "  },\n");

    fprintf(f, "  \"kernel\": {\n");
    json_kv_str(f, "name", r->kernel->name, ",");
    json_kv_str(f, "isa", r->kernel->isa, ",");
    fprintf(f, "    \"lanes\": %u,\n", r->kernel->lanes);
    fprintf(f, "    \"streams\": %u,\n", r->kernel->streams);
    fprintf(f, "    \"selected_by\": \"%s\",\n",
            cfg->force_kernel ? "user" : "autotune");
    fprintf(f, "    \"runs_on\": \"%s\"\n",
            r->kernel->device ? "device" : "cpu");
    fprintf(f, "  },\n");

    if (r->device_name[0]) {
        fprintf(f, "  \"device\": {\n");
        json_kv_str(f, "name", r->device_name, ",");
        json_kv_str(f, "vendor", r->device_vendor, ",");
        json_kv_str(f, "driver", r->device_driver, ",");
        fprintf(f, "    \"global_work\": %zu,\n", r->device_global);
        fprintf(f, "    \"local_work\": %zu,\n", r->device_local);
        fprintf(f, "    \"corpus_sweeps_per_launch\": %u,\n",
                r->device_repeats);
        fprintf(f, "    \"kernel_busy_fraction\": %.4f,\n", r->device_busy);
        json_kv_str(f, "transfer_mode",
                    r->transfer == VB_TRANSFER_STREAM ? "stream" : "resident",
                    ",");
        if (r->transfer == VB_TRANSFER_STREAM) {
            fprintf(f, "    \"transfer_bytes_per_pass\": %llu,\n",
                    (unsigned long long) r->device_transfer_bytes);
            fprintf(f, "    \"transfer_busy_fraction\": %.4f,\n",
                    r->device_transfer_busy);
            fprintf(f, "    \"transfer_gbytes_per_sec\": %.4f,\n",
                    r->device_transfer_gbps);
            /* Raw per-pass times. Transfer is constant in the iteration
               count and kernel time is linear in it, so a sweep of these two
               solves for the balance point rather than bracketing it. */
            fprintf(f, "    \"kernel_ns_per_pass\": %llu,\n",
                    (unsigned long long) r->device_kernel_ns_per_pass);
            fprintf(f, "    \"transfer_ns_per_pass\": %llu,\n",
                    (unsigned long long) r->device_transfer_ns_per_pass);
            /* The goal 2 figure: >1 compute-limited, <1 link-limited. */
            fprintf(f, "    \"compute_transfer_ratio\": %.4f,\n",
                    r->compute_transfer_ratio);
            json_kv_str(f, "bound_by",
                        r->compute_transfer_ratio >= 1.0 ? "compute"
                                                         : "transfer", ",");
        }
        if (r->gpu_clocks.valid) {
            char why[128];
            fprintf(f, "    \"gpu_clock_mhz\": {\n");
            fprintf(f, "      \"first\": %u,\n", r->gpu_clocks.sm_mhz_first);
            fprintf(f, "      \"last\": %u,\n",  r->gpu_clocks.sm_mhz_last);
            fprintf(f, "      \"min\": %u,\n",   r->gpu_clocks.sm_mhz_min);
            fprintf(f, "      \"max\": %u,\n",   r->gpu_clocks.sm_mhz_max);
            fprintf(f, "      \"samples\": %d\n", r->gpu_clocks.n_samples);
            fprintf(f, "    },\n");
            if (r->gpu_clocks.temp_c_max >= 0) {
                fprintf(f, "    \"gpu_temp_c\": { \"first\": %d, \"last\": %d, "
                        "\"max\": %d },\n",
                        r->gpu_clocks.temp_c_first, r->gpu_clocks.temp_c_last,
                        r->gpu_clocks.temp_c_max);
            }
            json_kv_str(f, "gpu_throttle_reasons",
                        vb_gpu_throttle_str(r->gpu_clocks.throttle_seen,
                                            why, sizeof why), ",");
        }
        fprintf(f, "    \"device_count\": %d\n", r->device_count);
        fprintf(f, "  },\n");
    }

    fprintf(f, "  \"environment\": {\n");
    json_kv_str(f, "cpu", si->cpu_brand, ",");
    fprintf(f, "    \"cpus_online\": %d,\n", si->cpus_online);
    fprintf(f, "    \"smt_active\": %s,\n",
            si->smt_active < 0 ? "null" : (si->smt_active ? "true" : "false"));
    json_kv_str(f, "governor", si->governor, ",");
    fprintf(f, "    \"freq_khz_min\": %ld,\n", si->freq_khz_min);
    fprintf(f, "    \"freq_khz_max\": %ld,\n", si->freq_khz_max);
    fprintf(f, "    \"freq_khz_at_start\": %ld,\n", si->freq_khz_now);    /* Re-read after the timed region. -1 means not taken. A drop between the
       two is thermal or power drift, which a single startup reading cannot
       show. */
    fprintf(f, "    \"freq_khz_at_end\": %ld,\n", si->freq_khz_at_end);
    fprintf(f, "    \"loadavg_1min_at_end\": %.2f,\n", si->loadavg1_at_end);
    fprintf(f, "    \"temp_milli_c\": %ld,\n", si->temp_milli_c);
    fprintf(f, "    \"temp_milli_c_at_end\": %ld,\n", si->temp_milli_c_at_end);
    json_kv_str(f, "temp_source", si->temp_source, ",");
    json_kv_str(f, "governor_at_end", si->governor_at_end, ",");

    if (si->loadavg1 >= 0.0)
        fprintf(f, "    \"loadavg_1min\": %.2f,\n", si->loadavg1);
    else
        fprintf(f, "    \"loadavg_1min\": null,\n");
    json_kv_str(f, "kernel_version", si->kernel, ",");
    json_kv_str(f, "os", si->os, ",");
    json_kv_str(f, "compiler", si->compiler, ",");
    /* Every ISA the dispatcher gates on. Reporting only the x86 three left
       every AArch64 result unable to say which instruction set it ran. */
    fprintf(f, "    \"isa_available\": {\"sse2\": %s, \"avx2\": %s, "
               "\"avx512f\": %s, \"sha_ni\": %s, \"neon\": %s, "
               "\"sve\": %s, \"sve2\": %s},\n",
            si->has_sse2    ? "true" : "false",
            si->has_avx2    ? "true" : "false",
            si->has_avx512f ? "true" : "false",
            si->has_sha_ni  ? "true" : "false",
            si->has_neon    ? "true" : "false",
            si->has_sve     ? "true" : "false",
            si->has_sve2    ? "true" : "false");
    fprintf(f, "    \"threads_used\": %u,\n", r->threads);
    fprintf(f, "    \"pinned_cpus\": %u,\n", r->pinned_cpus);
    /* pinned_cpus is 0 both when a pin was refused and when the platform has
       no affinity API to attempt; can_pin separates the two, so a reader
       grouping results by whether pinning was even possible does not have to
       parse the English warning text for it. */
    fprintf(f, "    \"can_pin\": %s,\n", si->can_pin ? "true" : "false");
    /* The verdict and the evidence behind it, so a reader can disagree.
       "unknown" is a real answer on AArch64, where the x86 hypervisor
       bit has no equivalent and DMI may name nothing. */
    json_kv_str(f, "virtualized",
                si->virtualized == VB_VIRT_YES ? "yes" :
                si->virtualized == VB_VIRT_NO  ? "no"  : "unknown", ",");
    json_kv_str(f, "sys_vendor", si->sys_vendor, ",");
    json_kv_str(f, "product_name", si->product_name, "");
    fprintf(f, "  },\n");

    fprintf(f, "  \"energy\": {\n");
    if (r->power.n > 0) {
        double cpu = vb_power_scope_joules(&r->power, VB_PWR_CPU_PACKAGE);
        double gpu = vb_power_scope_joules(&r->power, VB_PWR_GPU);
        /* Not cpu + gpu. On Intel client parts the GPU figure is the RAPL
           uncore domain, which is inside the package -- adding them counted
           the integrated GPU twice and understated hashes/joule by 15% on the
           development machine. vb_power_total_joules() counts each physical
           domain once. */
        double total = vb_power_total_joules(&r->power);
        int have = total >= 0.0;

        fprintf(f, "    \"available\": true,\n");
        if (cpu >= 0.0) {
            fprintf(f, "    \"cpu_package_joules\": %.4g,\n", cpu);
            fprintf(f, "    \"cpu_package_watts\": %.4g,\n",
                    r->total_seconds > 0 ? cpu / r->total_seconds : 0.0);
        }
        if (gpu >= 0.0) {
            fprintf(f, "    \"gpu_joules\": %.4g,\n", gpu);
            fprintf(f, "    \"gpu_watts\": %.4g,\n",
                    r->total_seconds > 0 ? gpu / r->total_seconds : 0.0);
        }
        if (have && total > 0.0)
            fprintf(f, "    \"hashes_per_joule\": %.6g,\n",
                    (double) r->total_hashes / total);

        /* Count what has been emitted, not what has been iterated. Separating
           on the loop index emits a leading comma whenever the first source is
           invalid and a later one is not -- "sources": [, {...}] -- which no
           parser accepts, and which breaks the machine-readable contract on a
           machine nobody happened to have. */
        fprintf(f, "    \"sources\": [");
        int emitted = 0;
        for (int i = 0; i < r->power.n; i++) {
            if (!r->power.src[i].valid)
                continue;
            fprintf(f, "%s{\"name\": ", emitted++ ? ", " : "");
            json_str(f, r->power.src[i].name);
            fprintf(f, ", \"scope\": \"%s\", \"joules\": %.4g}",
                    vb_power_scope_name(r->power.src[i].scope),
                    r->power.src[i].joules);
        }
        fprintf(f, "]\n");
    } else {
        fprintf(f, "    \"available\": false,\n");
        json_kv_str(f, "reason", r->power.unavailable, "");
    }
    fprintf(f, "  },\n");

    /* Pinning that was asked for and refused belongs here rather than in
       sysinfo: it is a property of this run, not of the machine. */
    const char *pinwarn = r->pin_failed
        ? "thread pinning was requested but at least one CPU was refused; "
          "the process may be confined to a cpuset that does not include the "
          "CPUs chosen. Placement is not what was asked for."
        : NULL;

    fprintf(f, "  \"warnings\": [");
    if (warn) {
        fprintf(f, "\n    ");
        json_str(f, warn);
        if (pinwarn)
            fprintf(f, ",");
    }
    if (pinwarn) {
        fprintf(f, "\n    ");
        json_str(f, pinwarn);
    }
    if (warn || pinwarn)
        fprintf(f, "\n  ");
    fprintf(f, "]\n");

    fprintf(f, "}\n");
}

static void bar(FILE *f, double frac, int width)
{
    int fill = (int) (frac * width + 0.5);
    if (fill < 0) fill = 0;
    if (fill > width) fill = width;
    for (int i = 0; i < width; i++)
        fputc(i < fill ? '#' : '.', f);
}

void vb_report_human(FILE *f, const vb_result *r, const vb_sysinfo *si,
                     const vb_config *cfg)
{
    const char *warn = vb_sysinfo_warnings(si);
    char wid[64];

    vb_workload_id(wid, sizeof wid, r->alg, r->message_bytes, r->iterations);

    fprintf(f, "\n");
    fprintf(f, "valubench %s   workload %s\n", VB_VERSION, wid);
    fprintf(f, "==================================================================\n");
    fprintf(f, "\n");

    fprintf(f, "  %.2f MH/s   (median of %u samples, %u thread%s)\n",
            r->median / 1e6, r->n_samples, r->threads,
            r->threads == 1 ? "" : "s");
    fprintf(f, "  %.2f MC/s   compressions (%u block%s x %u iteration%s)\n",
            r->median * r->iterations * r->blocks / 1e6,
            r->blocks, r->blocks == 1 ? "" : "s",
            r->iterations, r->iterations == 1 ? "" : "s");
    fprintf(f, "  %.2f MB/s   message throughput (%u-byte messages)\n",
            r->median * r->message_bytes * r->iterations / 1e6,
            r->message_bytes);
    fprintf(f, "\n");

    fprintf(f, "  kernel      %s  (%s, %u lanes x %u streams, chosen by %s)\n",
            r->kernel->name, r->kernel->isa, r->kernel->lanes,
            r->kernel->streams, cfg->force_kernel ? "user" : "autotune");
    if (r->device_name[0]) {
        fprintf(f, "  device      %s (%s, driver %s)\n",
                r->device_name, r->device_vendor, r->device_driver);
        if (r->device_count > 1)
            fprintf(f, "  devices     %d, running concurrently over slices of "
                       "the corpus\n", r->device_count);
        fprintf(f, "  launch      %zu work-items x %zu per group, "
                   "%u corpus sweeps%s\n",
                r->device_global, r->device_local, r->device_repeats,
                r->device_count > 1 ? "  (first device)" : "");
        fprintf(f, "  kernel busy %.1f%% of wall time%s\n",
                r->device_busy * 100.0,
                (r->device_busy < 0.9 && r->transfer != VB_TRANSFER_STREAM)
                    ? "  (the rest is launch overhead)" : "");

        if (r->transfer == VB_TRANSFER_STREAM) {
            fprintf(f, "  transfer    %.1f%% of wall, %.2f GB/s host->device "
                       "(%.1f MiB per pass)\n",
                    r->device_transfer_busy * 100.0,
                    r->device_transfer_gbps,
                    (double) r->device_transfer_bytes / 1048576.0);
            fprintf(f, "  bound by    %s  (compute/transfer = %.2f)\n",
                    r->compute_transfer_ratio >= 1.0 ? "COMPUTE" : "TRANSFER",
                    r->compute_transfer_ratio);
        }
    }
    fprintf(f, "  corpus      %llu messages, %.2f MiB working set\n",
            (unsigned long long) r->batch_messages,
            (double) r->working_set_bytes / (1024.0 * 1024.0));
    fprintf(f, "  verified    %s  checksum ", r->verified ? "yes" : "NO");
    for (unsigned i = 0; i < r->alg->digest_words; i++) {
        if (r->alg->word_bytes == 8)
            fprintf(f, "%016llx", (unsigned long long) r->checksum[i]);
        else
            fprintf(f, "%08x", (unsigned) r->checksum[i]);
    }
    fprintf(f, "\n");
    fprintf(f, "\n");

    fprintf(f, "  Distribution\n");
    fprintf(f, "    min       %10.2f MH/s\n", r->min / 1e6);
    fprintf(f, "    median    %10.2f MH/s\n", r->median / 1e6);
    fprintf(f, "    mean      %10.2f MH/s\n", r->mean / 1e6);
    fprintf(f, "    max       %10.2f MH/s\n", r->max / 1e6);
    fprintf(f, "    stddev    %10.2f MH/s\n", r->stddev / 1e6);
    fprintf(f, "    CoV       %10.2f %%   %s\n", r->cov,
            (r->cov <= cfg->cov_threshold) ? "(stable)"
                                           : "(UNSTABLE -- see warnings)");
    fprintf(f, "\n");

    if (r->n_samples > 1) {
        fprintf(f, "  Samples (relative to max)\n");
        for (unsigned i = 0; i < r->n_samples; i++) {
            fprintf(f, "    %2u  ", i + 1);
            bar(f, r->max > 0 ? r->sample_hps[i] / r->max : 0.0, 40);
            fprintf(f, "  %8.2f MH/s\n", r->sample_hps[i] / 1e6);
        }
        fprintf(f, "\n");
    }

    {
        double cpu = vb_power_scope_joules(&r->power, VB_PWR_CPU_PACKAGE);
        double gpu = vb_power_scope_joules(&r->power, VB_PWR_GPU);
        /* Each physical domain once; see the JSON path. */
        double total = vb_power_total_joules(&r->power);

        /* Sources can exist yet produce nothing -- a counter that failed to
           sample, say. Print the section only when there is a number in it. */
        if (cpu >= 0.0 || gpu >= 0.0) {
        fprintf(f, "  Energy\n");
        if (cpu >= 0.0)
            fprintf(f, "    cpu       %8.2f J   %6.2f W\n", cpu,
                    r->total_seconds > 0 ? cpu / r->total_seconds : 0.0);
        if (gpu >= 0.0)
            fprintf(f, "    gpu       %8.2f J   %6.2f W\n", gpu,
                    r->total_seconds > 0 ? gpu / r->total_seconds : 0.0);
        if (total > 0.0)
            fprintf(f, "    efficiency %9.2f kH/J\n",
                    (double) r->total_hashes / total / 1e3);
        fprintf(f, "\n");
        }
    }

    fprintf(f, "  Environment\n");
    fprintf(f, "    cpu       %s\n", si->cpu_brand);
    fprintf(f, "    isa       sse2=%s avx2=%s avx512f=%s\n",
            si->has_sse2 ? "y" : "n",
            si->has_avx2 ? "y" : "n",
            si->has_avx512f ? "y" : "n");
    fprintf(f, "    cpus      %d online, smt %s, threads used %u\n",
            si->cpus_online,
            si->smt_active < 0 ? "unknown" : (si->smt_active ? "on" : "off"),
            r->threads);
    fprintf(f, "    governor  %s\n", si->governor);
    if (si->loadavg1 >= 0.0)
        fprintf(f, "    load      %.2f (1 min)\n", si->loadavg1);
    if (si->freq_khz_max > 0)
        fprintf(f, "    freq      %ld MHz now, %ld MHz max\n",
                si->freq_khz_now / 1000, si->freq_khz_max / 1000);
    fprintf(f, "    os        %s (%s)\n", si->os, si->kernel);
    fprintf(f, "    compiler  %s\n", si->compiler);
    fprintf(f, "\n");

    /*
     * Both views warn about the same things. The pinning warning reached the
     * JSON and not this one, so a human reading a run whose placement was
     * refused saw nothing -- and the schema promises the two views do not
     * disagree about what is worth warning about.
     */
    if (warn || r->pin_failed) {
        fprintf(f, "  Warnings\n");
        if (warn)
            fprintf(f, "    %s\n", warn);
        if (r->pin_failed)
            fprintf(f, "    thread pinning was requested but at least one CPU "
                       "was refused; placement is not what was asked for\n");
        fprintf(f, "\n");
    }

    fprintf(f, "  Each iteration is a full %s: all %u steps, real message\n",
            r->alg->name, r->alg->rounds);
    fprintf(f, "  words, no shortcuts. Not comparable to figures from\n");
    fprintf(f, "  implementations that do less work per hash they report.\n");
    fprintf(f, "\n");
}

/* ---- capability dump ---------------------------------------------------- */
/*
 * `--list --json`: what this binary can do, on this machine.
 *
 * The human `--list` is for a person deciding what to run. This is for the
 * tooling that drives the binary -- tools/sweep.py, tools/run.sh, a
 * Phoronix profile -- and it exists because every fact in it was previously
 * transcribed into the caller by hand, where it drifted. sweep.py carried its
 * own table of per-algorithm minimum message lengths; it was wrong for SHA-512
 * and silently dropped every point that depended on it. Facts the binary knows
 * should come from the binary.
 *
 * Same contract as the result JSON: a versioned schema string, where adding a
 * field is compatible and removing or renaming one is not, so a consumer keys
 * on the schema prefix rather than the exact version. See docs/schema.md.
 */

static void json_alg(FILE *f, const vb_algorithm *a)
{
    fprintf(f, "    { \"name\": ");
    json_str(f, a->name);
    fprintf(f,
            ", \"digest_bytes\": %u, \"digest_words\": %u, "
            "\"block_bytes\": %u, \"word_bytes\": %u, \"length_bytes\": %u, "
            "\"rounds\": %u, \"big_endian\": %s, "
            "\"min_iteration_message_bytes\": %u }",
            a->digest_bytes, a->digest_words, a->block_bytes, a->word_bytes,
            a->length_bytes, a->rounds, a->big_endian ? "true" : "false",
            vb_alg_min_iter_bytes(a));
}

static void json_kernel(FILE *f, const vb_kernel *k)
{
    const vb_algorithm *a = vb_algorithm_by_id(k->alg);

    fprintf(f, "    { \"name\": ");
    json_str(f, k->name);
    fprintf(f, ", \"isa\": ");
    json_str(f, k->isa);
    fprintf(f, ", \"algorithm\": ");
    json_str(f, a ? a->name : "");
    fprintf(f, ", \"lanes\": %u, \"streams\": %u, \"where\": \"%s\", "
               "\"available\": %s }",
            k->lanes, k->streams, k->device ? "device" : "cpu",
            k->available() ? "true" : "false");
}

static void json_device(FILE *f, int index, const vb_ocl_device *d)
{
    fprintf(f, "      { \"index\": %d, \"name\": ", index);
    json_str(f, d->name);
    fprintf(f, ", \"vendor\": ");
    json_str(f, d->vendor);
    fprintf(f, ", \"type\": ");
    json_str(f, vb_ocl_type_name(d->type));
    fprintf(f, ",\n        \"compute_units\": %u, \"clock_mhz\": %u, "
               "\"max_work_group\": %llu,\n"
               "        \"global_mem_bytes\": %llu, "
               "\"max_alloc_bytes\": %llu, \"local_mem_bytes\": %llu,\n"
               "        \"platform_name\": ",
            (unsigned) d->compute_units, (unsigned) d->clock_mhz,
            (unsigned long long) d->max_work_group,
            (unsigned long long) d->global_mem,
            (unsigned long long) d->max_alloc,
            (unsigned long long) d->local_mem);
    json_str(f, d->platform_name);
    fprintf(f, ", \"platform_version\": ");
    json_str(f, d->platform_version);
    fprintf(f, ",\n        \"driver_version\": ");
    json_str(f, d->driver_version);
    fprintf(f, ", \"device_version\": ");
    json_str(f, d->device_version);
    fprintf(f, " }");
}

/*
 * The "opencl" object, indented to sit inside a top-level document. No devices
 * is a normal outcome, not a failure -- the binary runs CPU-only -- so it is
 * reported as an empty list plus the reason, never as an error.
 */
static void json_opencl(FILE *f)
{
    vb_ocl_device d[VB_OCL_MAX_DEVICES];
    int n = vb_ocl_devices(d, VB_OCL_MAX_DEVICES);
    const char *why = n > 0 ? NULL : vb_ocl_error();

    fprintf(f, "  \"opencl\": {\n");
    fprintf(f, "    \"available\": %s,\n", n > 0 ? "true" : "false");
    fprintf(f, "    \"error\": ");
    if (why)
        json_str(f, why);
    else
        fputs("null", f);
    fprintf(f, ",\n    \"devices\": [");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s\n", i ? "," : "");
        json_device(f, i, &d[i]);
    }
    fprintf(f, "%s]\n", n > 0 ? "\n    " : "");
    fprintf(f, "  }\n");
}

void vb_report_capabilities_json(FILE *f)
{
    vb_config def;
    size_t count;
    const vb_kernel *ks = vb_kernels(&count);

    vb_config_defaults(&def);

    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"valubench/capabilities/1\",\n");

    fprintf(f, "  \"benchmark\": {\n");
    json_kv_str(f, "name", "valubench", ",");
    json_kv_str(f, "version", VB_VERSION, "");
    fprintf(f, "  },\n");

    /* Ranges the binary enforces. A driver that respects these never has to
       parse an error message to find out it asked for something impossible. */
    fprintf(f, "  \"limits\": {\n");
    fprintf(f, "    \"message_bytes_min\": %u,\n", VB_MIN_MSG_BYTES);
    fprintf(f, "    \"message_bytes_max\": %u,\n", VB_MAX_MSG_BYTES);
    fprintf(f, "    \"iterations_max\": %u,\n", VB_MAX_ITERS);
    fprintf(f, "    \"samples_max\": %d,\n", VB_MAX_SAMPLES);
    fprintf(f, "    \"threads_max\": %d,\n", VB_MAX_THREADS);
    fprintf(f, "    \"working_set_kb_min\": 1,\n");
    /* Every kernel's group size divides this, so the batch -- and therefore
       the checksum -- is the same whichever kernel ran. It is also the floor
       on the message count, and so on the reachable working set. */
    fprintf(f, "    \"batch_lcm_messages\": %u,\n", VB_BATCH_LCM);
    fprintf(f, "    \"online_cpus\": %u\n", vb_online_cpus());
    fprintf(f, "  },\n");

    fprintf(f, "  \"defaults\": {\n");
    json_kv_str(f, "algorithm", def.alg->name, ",");
    fprintf(f, "    \"threads\": %u,\n", def.threads);
    fprintf(f, "    \"iterations\": %u,\n", def.iterations);
    fprintf(f, "    \"message_bytes\": %u,\n", def.message_bytes);
    fprintf(f, "    \"working_set_kb\": %u,\n", def.working_set_kb);
    fprintf(f, "    \"samples\": %u,\n", def.n_samples);
    fprintf(f, "    \"time_ms\": %u,\n", def.target_ms);
    fprintf(f, "    \"warmup_ms\": %u,\n", def.warmup_ms);
    json_kv_str(f, "transfer",
                def.transfer == VB_TRANSFER_STREAM ? "stream" : "resident",
                ",");
    json_kv_str(f, "where",
                def.where == VB_WHERE_CPU ? "cpu" :
                def.where == VB_WHERE_DEVICE ? "device" : "any", ",");
    fprintf(f, "    \"cov_threshold_percent\": %.4g\n", def.cov_threshold);
    fprintf(f, "  },\n");

    fprintf(f, "  \"exit_codes\": { \"ok\": %d, \"verify_failed\": %d, "
               "\"usage\": %d, \"noisy\": %d },\n",
            VB_EXIT_OK, VB_EXIT_VERIFY_FAILED, VB_EXIT_USAGE, VB_EXIT_NOISY);

    fprintf(f, "  \"transfer_modes\": [\"resident\", \"stream\"],\n");
    /* Autotune restrictions. "cpu" is what makes a CPU baseline measurable on
       a machine whose device kernel would otherwise win every probe. */
    fprintf(f, "  \"where_filters\": [\"any\", \"cpu\", \"device\"],\n");

    fprintf(f, "  \"algorithms\": [\n");
    for (int i = 0; i < VB_ALG_COUNT; i++) {
        const vb_algorithm *a = vb_algorithm_by_id((vb_alg_id) i);
        if (!a)
            continue;
        if (i)
            fprintf(f, ",\n");
        json_alg(f, a);
    }
    fprintf(f, "\n  ],\n");

    /*
     * Every registered kernel, available or not. The unavailable ones are
     * listed on purpose: "this build has AVX-512 kernels but this CPU cannot
     * run them" is a different situation from "this build has none", and a
     * sweep should be able to tell them apart.
     */
    fprintf(f, "  \"kernels\": [\n");
    for (size_t i = 0; i < count; i++) {
        if (i)
            fprintf(f, ",\n");
        json_kernel(f, &ks[i]);
    }
    fprintf(f, "\n  ],\n");

    json_opencl(f);

    fprintf(f, "}\n");
}

void vb_report_devices_json(FILE *f)
{
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"valubench/devices/1\",\n");
    json_opencl(f);
    fprintf(f, "}\n");
}
