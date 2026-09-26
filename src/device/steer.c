/*
 * steer.c -- the steer table: one spelling per primitive, per target.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * A row here is a claim that on this vendor's hardware a primitive is best
 * spelled a particular way, and every row other than plain C must cite the
 * idiom probe (tools/idiom_probe.py) evidence that made it. A steer that does
 * not beat or match plain C on its target has no place here: it would be
 * tuning the benchmark to a compiler rather than to the hardware.
 */

#include "device_steer.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* Spelling ids, as device_primitives.h numbers them. */
enum {
    SP_PLAIN  = 0,
    SP_CLB    = 1,
    SP_PTX    = 2,
    SP_AMDGCN = 3
};

static const char *const SPELLING_NAME[] = {
    "plain", "opencl-builtin", "ptx", "amdgcn"
};

/* The primitives the cores use, in the order the table columns follow. */
enum { P_ROTL32, P_ROTR64, P_CH, P_MAJ, P_COUNT };

static const struct {
    const char *name;
    const char *define;
} PRIMITIVE[P_COUNT] = {
    { "rotl32", "VB_STEER_ROTL32" },
    { "rotr64", "VB_STEER_ROTR64" },
    { "ch",     "VB_STEER_CH" },
    { "maj",    "VB_STEER_MAJ" },
};

typedef struct {
    vb_vendor vendor;
    int       spelling[P_COUNT];
} steer_row;

/*
 * The table. A vendor without a row gets the OpenCL builtin spelling under
 * OpenCL -- how the kernels were written before the shared source existed,
 * kept for targets nobody has probed because changing what an unmeasured
 * compiler sees is a regression with nothing measured to set against it --
 * and plain C under any other dialect.
 */
static const steer_row TABLE[] = {
    /*
     * NVIDIA: only the 32-bit rotate is steered, to one funnel shift in inline
     * PTX. Evidence, tools/idiom_probe.py with NVRTC 13.3 and, on an RTX PRO
     * 2000 (sm_120), NVIDIA's OpenCL in driver 610.57.04; 2026-09-26:
     *
     *   - alone, plain C reaches the ideal instruction for every primitive
     *     under both compilers and every architecture from sm_75 to sm_120:
     *     one SHF per 32-bit rotate, two per 64-bit, one LOP3 per Ch or Maj
     *     word;
     *   - in context (--kernels) SHA-1's rotates written plainly are missed
     *     -- shift, shift, or -- by NVIDIA's OpenCL on sm_120 and by NVRTC on
     *     sm_80 and sm_90, costing that kernel a fifth to a quarter more
     *     instructions and more registers at every stream count. The PTX
     *     funnel shift recovers all of it and is neutral where plain was
     *     already found;
     *   - steering Ch or Maj to a PTX lop3 made SHA-512 markedly worse under
     *     NVRTC on sm_120, because asm over split 64-bit words hides them from
     *     the optimiser; rotr64 steered was neutral. All three stay plain.
     *
     * rotate() itself cannot be the steer: CUDA has no such builtin, and the
     * point of a steer is that both APIs compile the same text.
     */
    { VB_VENDOR_NVIDIA, { SP_PTX, SP_PLAIN, SP_PLAIN, SP_PLAIN } },
};

vb_vendor vb_vendor_classify(const char *vendor, unsigned pci_vendor_id)
{
    switch (pci_vendor_id) {
    case 0x10de: return VB_VENDOR_NVIDIA;
    case 0x1002: case 0x1022: return VB_VENDOR_AMD;
    case 0x8086: return VB_VENDOR_INTEL;
    case 0x106b: return VB_VENDOR_APPLE;
    case 0x13b5: return VB_VENDOR_ARM;
    default: break;
    }
    if (!vendor)
        return VB_VENDOR_OTHER;
    if (strstr(vendor, "NVIDIA"))
        return VB_VENDOR_NVIDIA;
    if (strstr(vendor, "Advanced Micro Devices") || strstr(vendor, "AMD"))
        return VB_VENDOR_AMD;
    if (strstr(vendor, "Intel"))
        return VB_VENDOR_INTEL;
    if (strstr(vendor, "Apple"))
        return VB_VENDOR_APPLE;
    if (strstr(vendor, "ARM") || strstr(vendor, "Arm"))
        return VB_VENDOR_ARM;
    return VB_VENDOR_OTHER;
}

const char *vb_vendor_name(vb_vendor v)
{
    switch (v) {
    case VB_VENDOR_NVIDIA: return "nvidia";
    case VB_VENDOR_AMD:    return "amd";
    case VB_VENDOR_INTEL:  return "intel";
    case VB_VENDOR_APPLE:  return "apple";
    case VB_VENDOR_ARM:    return "arm";
    default:               return "other";
    }
}

/* Append to a buffer, truncating rather than overrunning. */
static void append(char *buf, size_t len, size_t *used, const char *fmt, ...)
{
    if (*used >= len)
        return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *used, len - *used, fmt, ap);
    va_end(ap);
    if (n > 0)
        *used += (size_t) n;
}

int vb_device_steers(vb_vendor vendor, vb_dialect dialect, int neutral,
                     char *defines, size_t defines_len,
                     char *record, size_t record_len)
{
    int spelling[P_COUNT];
    int found = 0;

    defines[0] = '\0';
    record[0] = '\0';
    if (neutral)
        return 0;

    for (size_t i = 0; i < sizeof TABLE / sizeof TABLE[0]; i++)
        if (TABLE[i].vendor == vendor) {
            memcpy(spelling, TABLE[i].spelling, sizeof spelling);
            found = 1;
        }
    if (!found)
        for (int p = 0; p < P_COUNT; p++)
            spelling[p] = dialect == VB_DIALECT_OPENCL ? SP_CLB : SP_PLAIN;

    size_t dl = 0, rl = 0;
    for (int p = 0; p < P_COUNT; p++) {
        if (spelling[p] == SP_PLAIN)
            continue;
        if (spelling[p] == SP_CLB && dialect != VB_DIALECT_OPENCL) {
            snprintf(record, record_len, "the %s steer set uses the OpenCL "
                     "builtin spelling for %s, which only OpenCL can compile",
                     vb_vendor_name(vendor), PRIMITIVE[p].name);
            defines[0] = '\0';
            return -1;
        }
        append(defines, defines_len, &dl, "%s-D%s=%d", dl ? " " : "",
               PRIMITIVE[p].define, spelling[p]);
        append(record, record_len, &rl, "%s%s=%s", rl ? "," : "",
               PRIMITIVE[p].name, SPELLING_NAME[spelling[p]]);
    }
    return 0;
}
