/*
 * sysinfo.c -- environment capture from /proc and /sys.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 */

#define _GNU_SOURCE

#include "sysinfo.h"
#include "cpu_features.h"
#include "vb_threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/utsname.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static int read_line_file(const char *path, char *out, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;

    if (!fgets(out, (int) n, f)) {
        fclose(f);
        return 0;
    }
    fclose(f);

    out[strcspn(out, "\r\n")] = '\0';
    return 1;
}

static long read_long_file(const char *path)
{
    char buf[64];
    if (!read_line_file(path, buf, sizeof buf))
        return -1;
    return strtol(buf, NULL, 10);
}

static void collect_governor(vb_sysinfo *si)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    char first[32] = "";
    int mixed = 0;

    snprintf(si->governor, sizeof si->governor, "unknown");

    for (long i = 0; i < n; i++) {
        char path[128], gov[32];
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_governor", i);
        if (!read_line_file(path, gov, sizeof gov))
            continue;

        if (first[0] == '\0')
            snprintf(first, sizeof first, "%s", gov);
        else if (strcmp(first, gov) != 0)
            mixed = 1;
    }

    if (first[0] != '\0')
        snprintf(si->governor, sizeof si->governor, "%s",
                 mixed ? "mixed" : first);
}

static void collect_freq(vb_sysinfo *si)
{
    si->freq_khz_min = read_long_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq");
    si->freq_khz_max = read_long_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    si->freq_khz_now = read_long_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
}

/*
 * CPU package temperature.
 *
 * Zone types are believed selectively. A machine commonly exposes both a
 * package sensor and an ambient one -- 53 C and 28 C on the development box --
 * and taking the hottest, or the first, gets it right there by luck and wrong
 * on the next machine. Only zones that name themselves as CPU or package are
 * used; anything else reports unavailable, which is a true answer where a
 * chassis reading dressed as a core temperature is not.
 */
static long read_cpu_temp(char *source, size_t source_n)
{
    static const char *want[] = {
        "x86_pkg_temp", "coretemp", "cpu", "soc", "Package", "cpu-thermal",
    };
    long best = -1;

    for (int zone = 0; zone < 32; zone++) {
        char path[128], type[32];
        long milli;

        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/type", zone);
        if (!read_line_file(path, type, sizeof type))
            continue;

        int wanted = 0;
        for (size_t i = 0; i < sizeof want / sizeof want[0]; i++)
            if (strstr(type, want[i])) {
                wanted = 1;
                break;
            }
        if (!wanted)
            continue;

        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", zone);
        milli = read_long_file(path);
        if (milli <= 0)                 /* a zone can exist and not answer */
            continue;

        if (milli > best) {
            best = milli;
            if (source && source_n)
                snprintf(source, source_n, "%s", type);
        }
    }
    return best;
}

static double loadavg_1min(void)
{
    double la[3];

    return getloadavg(la, 3) >= 1 ? la[0] : -1.0;
}

void vb_sysinfo_resample(vb_sysinfo *si)
{
    vb_sysinfo tmp;

    /* collect_governor writes into a whole sysinfo; give it a scratch one
       rather than clobbering the startup value we want to compare against. */
    memset(&tmp, 0, sizeof tmp);
    collect_governor(&tmp);
    snprintf(si->governor_at_end, sizeof si->governor_at_end, "%s", tmp.governor);

    si->freq_khz_at_end = read_long_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    si->loadavg1_at_end = loadavg_1min();
    si->temp_milli_c_at_end = read_cpu_temp(NULL, 0);
}

static void collect_os(vb_sysinfo *si)
{
    struct utsname u;

    snprintf(si->kernel, sizeof si->kernel, "unknown");
    snprintf(si->os, sizeof si->os, "unknown");

    if (uname(&u) == 0)
        snprintf(si->kernel, sizeof si->kernel, "%s %s", u.sysname, u.release);

#if defined(__APPLE__)

    /* There is no /etc/os-release. kern.osproductversion is the product
       version a user would recognise -- "15.6" -- as distinct from the Darwin
       release already captured above as the kernel. */
    char ver[64];
    size_t vlen = sizeof ver;
    if (sysctlbyname("kern.osproductversion", ver, &vlen, NULL, 0) == 0)
        snprintf(si->os, sizeof si->os, "macOS %s", ver);
    return;

#else

    FILE *f = fopen("/etc/os-release", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "PRETTY_NAME=", 12) == 0) {
                char *p = line + 12;
                if (*p == '"')
                    p++;
                p[strcspn(p, "\"\r\n")] = '\0';
                snprintf(si->os, sizeof si->os, "%s", p);
                break;
            }
        }
        fclose(f);
    }

#endif
}

static void collect_compiler(vb_sysinfo *si)
{
#if defined(__clang__)
    snprintf(si->compiler, sizeof si->compiler, "clang %d.%d.%d",
             __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    snprintf(si->compiler, sizeof si->compiler, "gcc %d.%d.%d",
             __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    snprintf(si->compiler, sizeof si->compiler, "unknown");
#endif
}

/*
 * Virtual machine or bare metal.
 *
 * Two signals, neither sufficient alone:
 *
 *   The x86 `hypervisor` CPUID bit, which the kernel surfaces as a flag in
 *   /proc/cpuinfo. Definitive where it exists -- present means virtualised,
 *   absent means not -- and it does not exist on AArch64 at all. `lscpu`
 *   relies on it, which is why every Graviton and Grace capture in this
 *   project reports no hypervisor while running on Nitro.
 *
 *   DMI, which both architectures expose. A hypervisor usually names itself
 *   there: QEMU, Bochs, VMware, Xen, KVM, Microsoft (Hyper-V), innotek and
 *   Oracle (VirtualBox), Parallels. EC2 is the awkward case -- it reports
 *   "Amazon EC2" for virtual and bare-metal instances alike -- so the
 *   instance type in product_name decides, a `.metal` suffix meaning the
 *   whole machine.
 *
 * Anything the two cannot settle stays unknown.
 */
static int dmi_field(const char *name, char *out, size_t n)
{
    char path[128];

    snprintf(path, sizeof path, "/sys/class/dmi/id/%s", name);
    if (read_line_file(path, out, n))
        return 1;
    /* Some kernels only expose the virtual path. */
    snprintf(path, sizeof path, "/sys/devices/virtual/dmi/id/%s", name);
    return read_line_file(path, out, n);
}

static int cpuinfo_has_hypervisor_flag(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[4096];
    int found = 0;

    if (!f)
        return -1;                  /* cannot tell */
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "flags", 5) != 0)
            continue;
        if (strstr(line, " hypervisor") || strstr(line, "\thypervisor")) {
            found = 1;
            break;
        }
        found = 0;                  /* a flags line without it */
        break;
    }
    fclose(f);
    return found;
}

/*
 * Pure, so the shapes that matter can be tested without owning the machines
 * that produce them. `hv_flag` is 1, 0, or -1 for "the question does not
 * apply here", which is what AArch64 always passes.
 */
int vb_classify_virt(const char *sys_vendor, const char *product_name,
                     int hv_flag)
{
    static const char *hv[] = {
        "QEMU", "Bochs", "VMware", "Xen", "KVM", "Microsoft Corporation",
        "innotek GmbH", "Oracle Corporation", "Parallels", "Apple Inc.",
        "Red Hat", "Google", "OpenStack", "Alibaba Cloud", "Nutanix",
    };

    if (!sys_vendor) sys_vendor = "";
    if (!product_name) product_name = "";

    /* EC2 first: it also matches nothing in the list, but its product name is
       the discriminator and a later rule must not pre-empt it. */
    if (strstr(sys_vendor, "Amazon EC2")) {
        if (product_name[0])
            return strstr(product_name, "metal") ? VB_VIRT_NO : VB_VIRT_YES;
        return hv_flag >= 0 ? (hv_flag ? VB_VIRT_YES : VB_VIRT_NO)
                            : VB_VIRT_UNKNOWN;
    }

    for (size_t i = 0; i < sizeof hv / sizeof hv[0]; i++)
        if (sys_vendor[0] && strstr(sys_vendor, hv[i]))
            return VB_VIRT_YES;

    /* Absence of the flag is evidence of bare metal only where the flag could
       have appeared. On AArch64 it could not, so nothing is concluded. */
    if (hv_flag >= 0)
        return hv_flag ? VB_VIRT_YES : VB_VIRT_NO;
    return VB_VIRT_UNKNOWN;
}

static void collect_virt(vb_sysinfo *si)
{
    int flag = -1;

    if (!dmi_field("sys_vendor", si->sys_vendor, sizeof si->sys_vendor))
        si->sys_vendor[0] = '\0';
    if (!dmi_field("product_name", si->product_name, sizeof si->product_name))
        si->product_name[0] = '\0';

#if defined(__x86_64__) || defined(__i386__)
    flag = cpuinfo_has_hypervisor_flag();
#endif
    si->virtualized = vb_classify_virt(si->sys_vendor, si->product_name, flag);
}

void vb_sysinfo_collect(vb_sysinfo *si)
{
    memset(si, 0, sizeof *si);

    snprintf(si->cpu_brand, sizeof si->cpu_brand, "%s", vb_cpu_brand());
    si->cpus_online = (int) sysconf(_SC_NPROCESSORS_ONLN);

#if defined(__APPLE__)

    /* No /sys smt/active to read. Comparing logical to physical cores answers
       the same question, and answers it as 0 on Apple silicon (which has no
       SMT) rather than leaving the field unknown, which the warning logic
       would then have to skip. */
    {
        int phys = 0, logical = 0;
        size_t plen = sizeof phys, llen = sizeof logical;
        si->smt_active =
            (sysctlbyname("hw.physicalcpu", &phys, &plen, NULL, 0) == 0 &&
             sysctlbyname("hw.logicalcpu", &logical, &llen, NULL, 0) == 0 &&
             phys > 0)
            ? (logical > phys) : -1;
    }

#else

    char smt[8];
    si->smt_active = read_line_file("/sys/devices/system/cpu/smt/active",
                                    smt, sizeof smt)
                   ? atoi(smt) : -1;

#endif

    si->can_pin = vb_thread_pin_supported();

    si->loadavg1 = -1.0;
    {
        double la[3];
        if (getloadavg(la, 3) >= 1)
            si->loadavg1 = la[0];
    }

    collect_governor(si);
    collect_freq(si);
    collect_os(si);
    collect_virt(si);

    si->temp_milli_c = read_cpu_temp(si->temp_source, sizeof si->temp_source);
    /* Until vb_sysinfo_resample() runs, the end-of-run fields say "not taken"
       rather than repeating the start values, which would read as no drift. */
    si->freq_khz_at_end = -1;
    si->loadavg1_at_end = -1.0;
    si->temp_milli_c_at_end = -1;
    si->governor_at_end[0] = '\0';
    collect_compiler(si);

    si->has_sse2     = vb_cpu_has_sse2();
    si->has_avx2     = vb_cpu_has_avx2();
    si->has_avx512f  = vb_cpu_has_avx512f();
    si->has_sha_ni   = vb_cpu_has_sha_ni();
    si->has_neon     = vb_cpu_has_neon();
    si->has_sve      = vb_cpu_has_sve();
    si->has_sve2     = vb_cpu_has_sve2();
}

const char *vb_sysinfo_warnings(const vb_sysinfo *si)
{
    static char buf[512];
    buf[0] = '\0';

    /*
     * These do not invalidate a result, but they widen its error bars, and a
     * 10%-significance bar is easily swamped by a powersave governor.
     * Report rather than silently tolerate (docs/research.md part 5).
     */
    if (strcmp(si->governor, "performance") != 0 &&
        strcmp(si->governor, "unknown") != 0) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                 "CPU governor is '%s', not 'performance'; ", si->governor);
    }

    /*
     * Competing load is measured against core count because that is what
     * actually matters: 2.0 on a 4-core box means half the machine is already
     * busy and any multi-threaded result will scatter.
     */
    if (si->loadavg1 >= 0.0 && si->cpus_online > 0 &&
        si->loadavg1 > 0.25 * si->cpus_online) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                 "system load average is %.2f on %d cores, so other work is "
                 "competing for the CPU; ", si->loadavg1, si->cpus_online);
    }

    if (si->smt_active == 1) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                 "SMT is enabled, which increases run-to-run variance; ");
    }

    if (!si->can_pin) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                 "this platform has no thread affinity API, so workers run "
                 "unpinned and dispersion is wider than a pinned run; ");
    }

    /*
     * Drift across the timed region, which a startup reading cannot show. A
     * clock that fell means the machine measured at the end is not the one
     * measured at the start, and the samples span both. 5% is well outside the
     * jitter of a settled part and well inside a real thermal or power-limit
     * drop.
     */
    if (si->freq_khz_now > 0 && si->freq_khz_at_end > 0 &&
        si->freq_khz_at_end < si->freq_khz_now * 95 / 100) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                 "clock fell %ld%% during the run (%ld -> %ld kHz), so the "
                 "later samples are not the same machine as the earlier ones; ",
                 100 - (si->freq_khz_at_end * 100 / si->freq_khz_now),
                 si->freq_khz_now, si->freq_khz_at_end);
    }

    /* Temperature is reported rather than judged -- what counts as hot depends
       on the part -- except where it rose enough during one run to explain a
       clock that also fell. */
    if (si->temp_milli_c > 0 && si->temp_milli_c_at_end > 0 &&
        si->temp_milli_c_at_end - si->temp_milli_c >= 10000) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                 "package temperature rose %ld C during the run (%ld -> %ld C); ",
                 (si->temp_milli_c_at_end - si->temp_milli_c) / 1000,
                 si->temp_milli_c / 1000, si->temp_milli_c_at_end / 1000);
    }

    if (si->freq_khz_now > 0 && si->freq_khz_max > 0 &&
        si->freq_khz_now < si->freq_khz_max * 9 / 10) {
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                 "CPU is running below 90%% of max frequency; ");
    }

    return buf[0] ? buf : NULL;
}
