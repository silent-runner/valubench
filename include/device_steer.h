/*
 * device_steer.h -- which spelling each device primitive gets on which target.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * src/kernels/gpu/device_primitives.h offers each primitive in several
 * spellings; this decides, per target, which of them a program is compiled
 * with, and says so in a form the result can record. The decision keys on the
 * device's vendor and never on the API compiling for it -- the reasons are in
 * device_primitives.h.
 */

#ifndef VALUBENCH_DEVICE_STEER_H
#define VALUBENCH_DEVICE_STEER_H

#include <stddef.h>

/* Whose hardware. Classified from the vendor string or PCI vendor ID, since
   that is what every API can report. */
typedef enum {
    VB_VENDOR_OTHER = 0,
    VB_VENDOR_NVIDIA,
    VB_VENDOR_AMD,
    VB_VENDOR_INTEL,
    VB_VENDOR_APPLE,
    VB_VENDOR_ARM
} vb_vendor;

/* The language a program is compiled in. */
typedef enum {
    VB_DIALECT_OPENCL = 0,
    VB_DIALECT_CUDA
} vb_dialect;

vb_vendor   vb_vendor_classify(const char *vendor, unsigned pci_vendor_id);
const char *vb_vendor_name(vb_vendor v);

/*
 * The steers for a target: -D options for the device compiler in `defines`,
 * and a record of them for the result in `record` -- "rotl32=ptx,ch=ptx", or
 * empty when every primitive is plain. `neutral` asks for no steers at all.
 *
 * Returns 0, or -1 with `record` explaining why when the target's steer set
 * cannot be expressed in this dialect. That is a table error, not a runtime
 * condition: a steer set is by construction one spelling per target for
 * every API that can reach it.
 */
int vb_device_steers(vb_vendor vendor, vb_dialect dialect, int neutral,
                     char *defines, size_t defines_len,
                     char *record, size_t record_len);

#endif /* VALUBENCH_DEVICE_STEER_H */
