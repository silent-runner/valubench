/*
 * power.c -- energy measurement from powercap, DRM hwmon and NVML.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * See power.h for what is measured and why. Everything here is optional and
 * failure is never fatal: a machine that exposes no energy counters simply
 * reports throughput without energy, and says which source was missing.
 */

#define _GNU_SOURCE

#include "power.h"

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

enum {
    SRC_POWERCAP,        /* energy_uj, microjoules, wraps at max_energy_range */
    SRC_HWMON_ENERGY,    /* energy1_input, microjoules */
    SRC_HWMON_POWER,     /* power1_average, microwatts -- integrated over time */
    SRC_NVML             /* millijoules since driver load */
};

/* ---- NVML, loaded the same way as OpenCL ------------------------------- */

typedef int (*nvml_fn_init)(void);
typedef int (*nvml_fn_shutdown)(void);
typedef int (*nvml_fn_count)(unsigned *);
typedef int (*nvml_fn_handle)(unsigned, void **);
typedef int (*nvml_fn_energy)(void *, unsigned long long *);
typedef int (*nvml_fn_power)(void *, unsigned *);
typedef int (*nvml_fn_name)(void *, char *, unsigned);
typedef int (*nvml_fn_clock)(void *, int, unsigned *);
typedef int (*nvml_fn_temp)(void *, int, unsigned *);
typedef int (*nvml_fn_throttle)(void *, unsigned long long *);
typedef int (*nvml_fn_pciinfo)(void *, void *);

/* NVML enum members used by name rather than by including nvml.h, which is
   not a build dependency: the library is dlopen'd. */
#define VB_NVML_CLOCK_SM        1
#define VB_NVML_TEMPERATURE_GPU 0

static struct {
    void *lib;
    int   ready;
    nvml_fn_shutdown Shutdown;
    nvml_fn_count    Count;
    nvml_fn_handle   Handle;
    nvml_fn_energy   Energy;
    nvml_fn_power    Power;
    nvml_fn_name     Name;
    nvml_fn_clock    Clock;
    nvml_fn_temp     Temp;
    nvml_fn_throttle Throttle;
    nvml_fn_pciinfo  PciInfo;
    void *dev[VB_POWER_MAX_SRC];
    int   n_dev;
} g_nvml;

static int nvml_start(void)
{
    if (g_nvml.ready)
        return 1;
    if (g_nvml.lib)
        return 0;               /* tried and failed */

    g_nvml.lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_nvml.lib)
        g_nvml.lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_nvml.lib)
        return 0;

    /* ISO C has no object-to-function pointer conversion; POSIX dlsym requires
       it. Going through a void* slot is the portable idiom. */
    nvml_fn_init init = NULL;
    *(void **) (&init) = dlsym(g_nvml.lib, "nvmlInit_v2");
    if (!init)
        *(void **) (&init) = dlsym(g_nvml.lib, "nvmlInit");

    *(void **) (&g_nvml.Shutdown) = dlsym(g_nvml.lib, "nvmlShutdown");
    *(void **) (&g_nvml.Count)    = dlsym(g_nvml.lib, "nvmlDeviceGetCount_v2");
    if (!g_nvml.Count)
        *(void **) (&g_nvml.Count) = dlsym(g_nvml.lib, "nvmlDeviceGetCount");
    *(void **) (&g_nvml.Handle)   = dlsym(g_nvml.lib,
                                          "nvmlDeviceGetHandleByIndex_v2");
    if (!g_nvml.Handle)
        *(void **) (&g_nvml.Handle) = dlsym(g_nvml.lib,
                                            "nvmlDeviceGetHandleByIndex");
    /* Volta and later. Preferred: a real counter beats sampling wattage. */
    *(void **) (&g_nvml.Energy) = dlsym(g_nvml.lib,
                                        "nvmlDeviceGetTotalEnergyConsumption");
    *(void **) (&g_nvml.Power)  = dlsym(g_nvml.lib, "nvmlDeviceGetPowerUsage");
    *(void **) (&g_nvml.Name)   = dlsym(g_nvml.lib, "nvmlDeviceGetName");

    /* Telemetry. Optional even when NVML loads: a driver may decline any of
       them, and a missing clock is reported as absent rather than as zero. */
    *(void **) (&g_nvml.Clock)  = dlsym(g_nvml.lib, "nvmlDeviceGetClockInfo");
    *(void **) (&g_nvml.Temp)   = dlsym(g_nvml.lib, "nvmlDeviceGetTemperature");
    *(void **) (&g_nvml.Throttle) =
        dlsym(g_nvml.lib, "nvmlDeviceGetCurrentClocksThrottleReasons");
    /*
     * The PCI address, which is what lets a card be recognised as the same
     * device another provider also sees. nvmlPciInfo_t has grown across
     * versions by appending, so its first member -- char busIdLegacy[16],
     * "0000:01:00.0" -- has been at offset 0 throughout. A generously sized
     * zeroed buffer and a read of that first field is version-independent
     * where naming the struct would not be.
     */
    *(void **) (&g_nvml.PciInfo) = dlsym(g_nvml.lib, "nvmlDeviceGetPciInfo_v3");
    if (!g_nvml.PciInfo)
        *(void **) (&g_nvml.PciInfo) = dlsym(g_nvml.lib, "nvmlDeviceGetPciInfo_v2");
    if (!g_nvml.PciInfo)
        *(void **) (&g_nvml.PciInfo) = dlsym(g_nvml.lib, "nvmlDeviceGetPciInfo");
    if (!g_nvml.Throttle)   /* renamed in newer NVML headers */
        *(void **) (&g_nvml.Throttle) =
            dlsym(g_nvml.lib, "nvmlDeviceGetCurrentClocksEventReasons");

    if (!init || !g_nvml.Count || !g_nvml.Handle || init() != 0)
        return 0;

    g_nvml.ready = 1;
    return 1;
}

/* ---- small helpers ------------------------------------------------------ */

static int read_u64(int fd, uint64_t *out)
{
    char buf[64];
    ssize_t n;

    if (lseek(fd, 0, SEEK_SET) < 0)
        return 0;
    n = read(fd, buf, sizeof buf - 1);
    if (n <= 0)
        return 0;
    buf[n] = '\0';
    *out = strtoull(buf, NULL, 10);
    return 1;
}

static int read_u64_path(const char *path, uint64_t *out)
{
    int fd = open(path, O_RDONLY);
    int ok;

    if (fd < 0)
        return 0;
    ok = read_u64(fd, out);
    close(fd);
    return ok;
}

static void read_str_path(const char *path, char *out, size_t n)
{
    int fd = open(path, O_RDONLY);
    ssize_t got;

    out[0] = '\0';
    if (fd < 0)
        return;
    got = read(fd, out, n - 1);
    close(fd);
    if (got <= 0) {
        out[0] = '\0';
        return;
    }
    out[got] = '\0';
    out[strcspn(out, "\r\n")] = '\0';
}

static vb_power_src *add_src(vb_power *p)
{
    if (p->n >= VB_POWER_MAX_SRC)
        return NULL;
    vb_power_src *s = &p->src[p->n++];
    memset(s, 0, sizeof *s);
    s->fd = -1;
    return s;
}

/* ---- discovery ---------------------------------------------------------- */

static void scan_powercap(vb_power *p, int *denied)
{
    DIR *d = opendir("/sys/class/powercap");
    struct dirent *e;

    if (!d)
        return;

    while ((e = readdir(d)) != NULL) {
        char path[512], name[64];

        /* The powercap RAPL driver registers its control type as "intel-rapl"
           on AMD parts too, so that prefix has covered both so far. "amd-rapl"
           is accepted as well because this path has never run on an AMD part,
           and a naming difference would present as silently absent energy
           rather than as an error -- the worst failure mode for a reading you
           are paying by the hour to take. */
        if (strncmp(e->d_name, "intel-rapl:", 11) != 0 &&
            strncmp(e->d_name, "amd-rapl:", 9) != 0)
            continue;

        snprintf(path, sizeof path, "/sys/class/powercap/%s/name", e->d_name);
        read_str_path(path, name, sizeof name);
        if (!name[0])
            continue;

        snprintf(path, sizeof path, "/sys/class/powercap/%s/energy_uj",
                 e->d_name);
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            *denied = 1;
            continue;
        }

        vb_power_src *s = add_src(p);
        if (!s) {
            close(fd);
            break;
        }

        s->kind = SRC_POWERCAP;
        s->fd = fd;

        snprintf(path, sizeof path,
                 "/sys/class/powercap/%s/max_energy_range_uj", e->d_name);
        if (!read_u64_path(path, &s->wrap_uj) || s->wrap_uj == 0)
            s->wrap_uj = (uint64_t) 1 << 32;

        /*
         * On client Intel parts the `uncore` domain is the integrated GPU, so
         * it is the device reading on a machine with no discrete card.
         */
        if (strstr(name, "package")) {
            s->scope = VB_PWR_CPU_PACKAGE;
            snprintf(s->name, sizeof s->name, "RAPL %s", name);
        } else if (strstr(name, "core") && !strstr(name, "uncore")) {
            s->scope = VB_PWR_CPU_CORES;
            snprintf(s->name, sizeof s->name, "RAPL %s", name);
        } else if (strstr(name, "uncore")) {
            /* The integrated GPU, and already inside the package figure. */
            s->scope = VB_PWR_GPU;
            s->contained = 1;
            snprintf(s->name, sizeof s->name, "RAPL uncore (integrated GPU)");
        } else if (strstr(name, "psys")) {
            /*
             * The whole platform: package, uncore, and everything on the board
             * the firmware can account for. It therefore *contains* the
             * package, exactly as uncore does, and adding it to a total that
             * already has the package counts the same silicon twice. Reported
             * as its own reading and never summed -- the same rule that fixed
             * the uncore double-count, applied to the domain that has it worse.
             */
            s->scope = VB_PWR_OTHER;
            s->contained = 1;
            snprintf(s->name, sizeof s->name, "RAPL psys (whole platform)");
        } else {
            /* dram and friends: genuinely outside the package, so summed. */
            s->scope = VB_PWR_OTHER;
            snprintf(s->name, sizeof s->name, "RAPL %s", name);
        }
    }

    closedir(d);
}

static void scan_drm_hwmon(vb_power *p)
{
    /*
     * Enumerate the cards that exist rather than assuming card0..card7. The
     * benchmark drives up to VB_OCL_MAX_DEVICES accelerators, so a fixed range
     * of eight silently dropped energy for anything numbered above it, and
     * DRM numbering is not guaranteed contiguous in the first place.
     */
    DIR *drm = opendir("/sys/class/drm");
    struct dirent *ce;

    if (!drm)
        return;

    while ((ce = readdir(drm)) != NULL) {
        char base[sizeof "/sys/class/drm//device/hwmon" + 256];
        DIR *d;
        struct dirent *e;
        int card;

        if (sscanf(ce->d_name, "card%d", &card) != 1)
            continue;
        /* card0-DP-1 and friends are connectors, not devices. */
        if (strchr(ce->d_name, '-'))
            continue;

        /*
         * The card's PCI address, which is what makes a device identifiable
         * across providers: /sys/class/drm/cardN/device is a symlink into the
         * PCI tree and its final component is the address. NVML sources carry
         * no id, so they are never deduplicated -- summing an NVIDIA card that
         * some other provider also sees would overstate by at most 2x, where
         * dropping it understates by everything it was doing.
         */
        char link[PATH_MAX], devid[32] = "";
        ssize_t ln;

        snprintf(base, sizeof base, "/sys/class/drm/%s/device", ce->d_name);
        ln = readlink(base, link, sizeof link - 1);
        if (ln > 0) {
            const char *slash;
            link[ln] = '\0';
            slash = strrchr(link, '/');
            /* A PCI address is 12 characters; anything longer is not one, and
               a silent truncation would invent a device id that matches. */
            const char *tail = slash ? slash + 1 : link;
            if (strlen(tail) < sizeof devid)
                snprintf(devid, sizeof devid, "%s", tail);
        }

        snprintf(base, sizeof base, "/sys/class/drm/%s/device/hwmon",
                 ce->d_name);
        d = opendir(base);
        if (!d)
            continue;

        while ((e = readdir(d)) != NULL) {
            char path[600], hname[32];

            if (strncmp(e->d_name, "hwmon", 5) != 0)
                continue;

            snprintf(path, sizeof path, "%s/%s/name", base, e->d_name);
            read_str_path(path, hname, sizeof hname);

            /* Energy counter preferred; average power is a fallback. */
            snprintf(path, sizeof path, "%s/%s/energy1_input", base, e->d_name);
            int fd = open(path, O_RDONLY);
            int kind = SRC_HWMON_ENERGY;

            if (fd < 0) {
                snprintf(path, sizeof path, "%s/%s/power1_average", base,
                         e->d_name);
                fd = open(path, O_RDONLY);
                kind = SRC_HWMON_POWER;
            }
            if (fd < 0)
                continue;

            vb_power_src *s = add_src(p);
            if (!s) {
                close(fd);
                break;
            }
            s->kind = kind;
            s->fd = fd;
            s->scope = VB_PWR_GPU;
            s->provider = VB_PWR_PROV_DRM;
            snprintf(s->dev_id, sizeof s->dev_id, "%s", devid);
            snprintf(s->name, sizeof s->name, "card%d %s%s", card,
                     hname[0] ? hname : "hwmon",
                     kind == SRC_HWMON_POWER ? " (avg power)" : "");
        }
        closedir(d);
    }

    closedir(drm);
}

static void scan_nvml(vb_power *p)
{
    unsigned count = 0;

    if (!nvml_start())
        return;
    if (g_nvml.Count(&count) != 0)
        return;

    for (unsigned i = 0; i < count && i < VB_POWER_MAX_SRC; i++) {
        void *h = NULL;
        if (g_nvml.Handle(i, &h) != 0 || !h)
            continue;
        if (!g_nvml.Energy && !g_nvml.Power)
            continue;

        vb_power_src *s = add_src(p);
        if (!s)
            break;

        s->kind = SRC_NVML;
        s->scope = VB_PWR_GPU;
        s->provider = VB_PWR_PROV_NVML;
        s->nvml_index = i;
        g_nvml.dev[i] = h;
        if ((int) i + 1 > g_nvml.n_dev)
            g_nvml.n_dev = (int) i + 1;

        if (g_nvml.PciInfo) {
            unsigned char info[256];
            memset(info, 0, sizeof info);
            if (g_nvml.PciInfo(h, info) == 0) {
                info[15] = '\0';                /* busIdLegacy is char[16] */
                if (info[0])
                    snprintf(s->dev_id, sizeof s->dev_id, "%s",
                             (const char *) info);
            }
        }

        char nm[80] = "";
        if (g_nvml.Name)
            g_nvml.Name(h, nm, sizeof nm - 1);
        snprintf(s->name, sizeof s->name, "NVML %s%s",
                 nm[0] ? nm : "GPU",
                 g_nvml.Energy ? "" : " (avg power)");
    }
}

void vb_power_open(vb_power *p)
{
    int denied = 0;

    memset(p, 0, sizeof *p);

    scan_powercap(p, &denied);
    scan_drm_hwmon(p);
    scan_nvml(p);

    if (p->n == 0) {
        if (denied)
            snprintf(p->unavailable, sizeof p->unavailable,
                     "RAPL counters exist but are not readable by this user; "
                     "they are root-only on most distributions, as hardening "
                     "against the PLATYPUS side channel. Either run as root, "
                     "or: sudo chmod a+r "
                     "/sys/class/powercap/intel-rapl:*/energy_uj  (note sysfs "
                     "does not support ACLs, so setfacl cannot be used, and "
                     "the mode resets on reboot unless set from a udev rule).");
        else
            snprintf(p->unavailable, sizeof p->unavailable,
                     "no energy counters found (no RAPL, no DRM hwmon power "
                     "node, no NVML)");
    }
}

/*
 * Release handles but keep the measurements. The result is reported after the
 * benchmark has finished with the sources, so clearing the source list here
 * would silently drop the energy block from the output.
 */
void vb_power_close(vb_power *p)
{
    for (int i = 0; i < p->n; i++) {
        if (p->src[i].fd >= 0)
            close(p->src[i].fd);
        p->src[i].fd = -1;
    }

    if (g_nvml.ready && g_nvml.Shutdown)
        g_nvml.Shutdown();
    if (g_nvml.lib)
        dlclose(g_nvml.lib);
    memset(&g_nvml, 0, sizeof g_nvml);
}

/* ---- sampling ----------------------------------------------------------- */

static int sample(vb_power_src *s, uint64_t *out)
{
    switch (s->kind) {
    case SRC_POWERCAP:
    case SRC_HWMON_ENERGY:
    case SRC_HWMON_POWER:
        return read_u64(s->fd, out);

    case SRC_NVML: {
        if (g_nvml.Energy) {
            unsigned long long mj = 0;
            if (g_nvml.Energy(g_nvml.dev[s->nvml_index], &mj) != 0)
                return 0;
            *out = (uint64_t) mj * 1000u;       /* mJ -> uJ */
            return 1;
        }
        if (g_nvml.Power) {
            unsigned mw = 0;
            if (g_nvml.Power(g_nvml.dev[s->nvml_index], &mw) != 0)
                return 0;
            *out = (uint64_t) mw * 1000u;       /* mW -> uW */
            return 1;
        }
        return 0;
    }
    default:
        return 0;
    }
}

void vb_power_begin(vb_power *p)
{
    for (int i = 0; i < p->n; i++) {
        vb_power_src *s = &p->src[i];
        s->valid = sample(s, &s->start_uj);
        s->joules = 0.0;
    }
}

void vb_power_end(vb_power *p, double seconds)
{
    for (int i = 0; i < p->n; i++) {
        vb_power_src *s = &p->src[i];
        uint64_t now = 0;

        if (!s->valid || !sample(s, &now)) {
            s->valid = 0;
            continue;
        }

        if (s->kind == SRC_HWMON_POWER ||
            (s->kind == SRC_NVML && !g_nvml.Energy)) {
            /* Instantaneous wattage: average the two readings and integrate.
               Less trustworthy than a counter, which is why the source name
               says so. */
            double w = ((double) s->start_uj + (double) now) / 2.0 / 1e6;
            s->joules = w * seconds;
        } else {
            if (now >= s->start_uj) {
                s->joules = (double) (now - s->start_uj) / 1e6;
            } else if (s->wrap_uj > 0) {
                /* powercap publishes max_energy_range_uj, so the modulus is
                   known and the counter can be unwrapped. */
                s->joules = (double)
                    ((s->wrap_uj - s->start_uj) + now) / 1e6;
            } else {
                /*
                 * The counter went backwards and nothing told us its range --
                 * hwmon energy and NVML do not publish one. A wrap, a reset, a
                 * driver reload and a device reset all look like this, and
                 * guessing a modulus turns any of them into a plausible
                 * number. Refuse the source for this interval instead.
                 */
                s->valid = 0;
                s->joules = 0.0;
            }
        }
    }
}

/*
 * The best provider present in a scope. RAPL first, then DRM hwmon, then NVML,
 * which is the order the header has always documented and the implementation
 * did not follow: it summed every valid source instead. An NVIDIA card visible
 * through both DRM and NVML was therefore counted twice.
 */
/*
 * Is source `i` a duplicate of an earlier source for the same device?
 *
 * Only a matching, non-empty device id makes a duplicate. An earlier version
 * deduplicated by *provider* -- one provider won a scope and the rest were
 * discarded -- which is right when two providers see one card and badly wrong
 * when they see two. An Intel iGPU visible through RAPL uncore beside an
 * NVIDIA card visible through NVML reported the iGPU's 5 J and dropped the
 * card's 250: the compute device's energy attributed to an idle one, a 50x
 * understatement in an ordinary desktop configuration.
 *
 * Where two sources do name the same device, the lower-numbered provider wins,
 * which is the preference order the header documents.
 */
static int duplicate_of_earlier(const vb_power *p, int i, vb_power_scope scope)
{
    const vb_power_src *s = &p->src[i];

    if (!s->dev_id[0])
        return 0;                   /* unidentified: taken at face value */

    for (int j = 0; j < p->n; j++) {
        const vb_power_src *o = &p->src[j];

        if (j == i || o->scope != scope || !o->valid || !o->dev_id[0])
            continue;
        /* Case-blind: NVML writes the hex of a PCI address in capitals and
           sysfs in lower case, so "0000:A1:00.0" and "0000:a1:00.0" are one
           card that a strcmp counted twice. */
        if (strcasecmp(o->dev_id, s->dev_id) != 0)
            continue;
        if (o->provider < s->provider ||
            (o->provider == s->provider && j < i))
            return 1;
    }
    return 0;
}

double vb_power_scope_joules(const vb_power *p, vb_power_scope scope)
{
    double total = 0.0;
    int found = 0;

    /* Sum every distinct device in the scope. Two sockets legitimately report
       two packages and two cards two GPUs; what must not happen is one device
       counted once per provider that can see it. */
    for (int i = 0; i < p->n; i++) {
        if (p->src[i].scope != scope || !p->src[i].valid)
            continue;
        if (duplicate_of_earlier(p, i, scope))
            continue;
        total += p->src[i].joules;
        found = 1;
    }
    return found ? total : -1.0;
}

double vb_power_total_joules(const vb_power *p)
{
    double total = 0.0;
    int any = 0;

    static const vb_power_scope scopes[] = {
        VB_PWR_CPU_PACKAGE, VB_PWR_CPU_CORES, VB_PWR_GPU, VB_PWR_OTHER
    };

    for (size_t si = 0; si < sizeof scopes / sizeof scopes[0]; si++) {
        vb_power_scope sc = scopes[si];

        for (int i = 0; i < p->n; i++) {
            const vb_power_src *s = &p->src[i];
            /* Contained domains are inside another scope's figure, and
               VB_PWR_CPU_CORES is inside the package. Neither is added. */
            if (s->scope != sc || !s->valid ||
                s->contained || sc == VB_PWR_CPU_CORES)
                continue;
            if (duplicate_of_earlier(p, i, sc))
                continue;
            total += s->joules;
            any = 1;
        }
    }
    return any ? total : -1.0;
}

static int same_device(const char *a, const char *b)
{
    return a && b && a[0] && b[0] && !strcasecmp(a, b);
}

void vb_power_attribute(vb_power *p, int cpu_hashed,
                        const char *const *gpu_pci, int n_gpu)
{
    /* Does every card the run used have an identified source of its own? */
    int unmatched = 0;
    for (int g = 0; g < n_gpu; g++) {
        int hit = 0;
        for (int i = 0; i < p->n; i++)
            if (p->src[i].scope == VB_PWR_GPU &&
                same_device(p->src[i].dev_id, gpu_pci[g]))
                hit = 1;
        if (!hit)
            unmatched = 1;
    }

    for (int i = 0; i < p->n; i++) {
        vb_power_src *s = &p->src[i];
        switch (s->scope) {
        case VB_PWR_CPU_PACKAGE:
        case VB_PWR_CPU_CORES:
            s->counted = cpu_hashed;
            break;
        case VB_PWR_GPU:
            if (s->dev_id[0]) {
                s->counted = 0;
                for (int g = 0; g < n_gpu; g++)
                    if (same_device(s->dev_id, gpu_pci[g]))
                        s->counted = 1;
            } else {
                s->counted = n_gpu > 0 && unmatched;
            }
            /* An integrated GPU's domain is inside the package: with the
               package counted too it would be counted twice. */
            if (s->contained && cpu_hashed)
                s->counted = 0;
            break;
        default:
            /* DRAM, outside the package, is part of what CPU hashing costs;
               the whole-platform domain contains the package and never is. */
            s->counted = cpu_hashed && !s->contained;
            break;
        }
    }
}

double vb_power_hashing_joules(const vb_power *p)
{
    double total = 0.0;
    int any = 0;

    for (int i = 0; i < p->n; i++) {
        const vb_power_src *s = &p->src[i];
        if (!s->valid || !s->counted)
            continue;
        /* Cores are inside the package; a contained domain outside the GPU
           scope is the whole platform. Neither is added. */
        if (s->scope == VB_PWR_CPU_CORES ||
            (s->contained && s->scope != VB_PWR_GPU))
            continue;
        if (duplicate_of_earlier(p, i, s->scope))
            continue;
        total += s->joules;
        any = 1;
    }
    return any ? total : -1.0;
}

const char *vb_power_scope_name(vb_power_scope s)
{
    switch (s) {
    case VB_PWR_CPU_PACKAGE: return "cpu_package";
    case VB_PWR_CPU_CORES:   return "cpu_cores";
    case VB_PWR_GPU:         return "gpu";
    default:                 return "other";
    }
}


/* ---- GPU clock telemetry ------------------------------------------------ */

#define VB_GPU_SELECT_MAX 32
static char g_select[VB_GPU_SELECT_MAX][16];
static int  g_n_select;

void vb_gpu_clocks_select(const char *const *pci, int n)
{
    g_n_select = 0;
    for (int i = 0; i < n && g_n_select < VB_GPU_SELECT_MAX; i++) {
        if (!pci[i] || !pci[i][0]) {
            g_n_select = 0;         /* one unidentified device: sample all */
            return;
        }
        snprintf(g_select[g_n_select++], sizeof g_select[0], "%s", pci[i]);
    }
}

/* Whether NVML device i is one the run uses. */
static int selected(int i)
{
    if (g_n_select == 0)
        return 1;
    if (!g_nvml.PciInfo || !g_nvml.dev[i])
        return 1;
    unsigned char info[256];
    memset(info, 0, sizeof info);
    if (g_nvml.PciInfo(g_nvml.dev[i], info) != 0)
        return 1;
    info[15] = '\0';               /* busIdLegacy, "0000:01:00.0" */
    for (int k = 0; k < g_n_select; k++)
        if (!strcasecmp((const char *) info, g_select[k]))
            return 1;
    return 0;
}

/* Which NVML devices the current run samples, decided once per reset so the
   per-iteration sample does not query PCI info. */
static int g_use[VB_POWER_MAX_SRC];

void vb_gpu_clocks_reset(vb_gpu_clocks *g)
{
    memset(g, 0, sizeof *g);
    g->temp_c_first = g->temp_c_last = g->temp_c_max = -1;
    /* Only meaningful once NVML is up, which vb_power_open() does. Sampling
       before that simply records nothing. */
    for (int i = 0; i < VB_POWER_MAX_SRC; i++) {
        g_use[i] = g_nvml.ready && i < g_nvml.n_dev && g_nvml.dev[i] &&
                   selected(i);
        g->n_devices += g_use[i];
    }
}

void vb_gpu_clocks_sample(vb_gpu_clocks *g)
{
    if (!g_nvml.ready || !g_nvml.Clock || g_nvml.n_dev <= 0)
        return;

    unsigned sm_hi = 0;
    int      temp_hi = -1;
    int      got = 0;

    for (int i = 0; i < g_nvml.n_dev; i++) {
        void *h = g_nvml.dev[i];
        unsigned mhz = 0;

        if (!h || !g_use[i] ||
            g_nvml.Clock(h, VB_NVML_CLOCK_SM, &mhz) != 0 || mhz == 0)
            continue;
        got = 1;
        if (mhz > sm_hi)
            sm_hi = mhz;

        unsigned t = 0;
        if (g_nvml.Temp &&
            g_nvml.Temp(h, VB_NVML_TEMPERATURE_GPU, &t) == 0 && (int) t > temp_hi)
            temp_hi = (int) t;

        unsigned long long why = 0;
        if (g_nvml.Throttle && g_nvml.Throttle(h, &why) == 0)
            g->throttle_seen |= (uint64_t) why;
    }

    if (!got)
        return;

    /*
     * The busiest device in the set, not the average: a run is only as
     * sustained as the part doing the work, and averaging a loaded card with
     * an idle one hides exactly the drop this exists to catch.
     */
    if (!g->valid) {
        g->valid = 1;
        g->sm_mhz_first = g->sm_mhz_min = g->sm_mhz_max = sm_hi;
        g->temp_c_first = g->temp_c_max = temp_hi;
    } else {
        if (sm_hi < g->sm_mhz_min) g->sm_mhz_min = sm_hi;
        if (sm_hi > g->sm_mhz_max) g->sm_mhz_max = sm_hi;
        if (temp_hi > g->temp_c_max) g->temp_c_max = temp_hi;
    }
    g->sm_mhz_last = sm_hi;
    g->temp_c_last = temp_hi;
    g->n_samples++;
}

const char *vb_gpu_throttle_str(uint64_t mask, char *buf, size_t n)
{
    static const struct { uint64_t bit; const char *name; } names[] = {
        { VB_GPU_THROTTLE_POWER,      "power-cap"    },
        { VB_GPU_THROTTLE_HW_SLOW,    "hw-slowdown"  },
        { VB_GPU_THROTTLE_SW_THERMAL, "sw-thermal"   },
        { VB_GPU_THROTTLE_HW_THERMAL, "hw-thermal"   },
        { VB_GPU_THROTTLE_HW_BRAKE,   "hw-power-brake" },
    };
    size_t used = 0;

    if (n == 0)
        return buf;
    buf[0] = '\0';

    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        if (!(mask & names[i].bit))
            continue;
        int w = snprintf(buf + used, n - used, "%s%s",
                         used ? "," : "", names[i].name);
        if (w < 0 || (size_t) w >= n - used)
            break;
        used += (size_t) w;
    }
    if (used == 0)
        snprintf(buf, n, "none");
    return buf;
}
