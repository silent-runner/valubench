/*
 * device.h -- the device layer every backend plugs into.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Everything about running a device kernel that does not depend on the API is
 * here: composing the program from the shared kernel sources, choosing the
 * launch geometry, calibrating repeats, folding the per-group partials, and
 * the multi-device split. A backend (src/opencl/backend.c, src/cuda/backend.c)
 * implements only what the API itself does -- enumerate, compile, allocate,
 * upload, launch, read back, time an event -- through the table in
 * vb_dev_backend.
 *
 * That split is what makes a comparison between APIs a comparison of the
 * APIs. The tuner, the repeat count, the timed loop and the verification are
 * the same code whichever backend runs, and so is the program text: the
 * backends differ in how they compile and launch it and in nothing else.
 */

#ifndef VALUBENCH_DEVICE_H
#define VALUBENCH_DEVICE_H

#include "valubench.h"
#include "device_steer.h"

#include <stddef.h>
#include <stdint.h>

#define VB_DEV_MAX 32

typedef enum {
    VB_BACKEND_OPENCL = 0,
    VB_BACKEND_CUDA,
    VB_BACKEND_COUNT
} vb_backend_id;

/* How a CUDA program is compiled. PTX_JIT leaves the last stage -- PTX to
   machine code -- to the driver, as OpenCL does; CUBIN has NVRTC finish it
   with the toolkit's own ptxas, a different compiler version. The two answer
   different questions and are never conflated in a result. */
typedef enum {
    VB_COMPILE_PTX_JIT = 0,
    VB_COMPILE_CUBIN
} vb_compile_mode;

/*
 * One device as one backend sees it. Filled by the backend's enumerate();
 * everything here is reported, and `pci` is what identifies one card seen
 * through two APIs as one device.
 */
typedef struct {
    vb_backend_id backend;
    int      ordinal;           /* the backend's own index */
    char     name[128];
    char     vendor[128];
    unsigned pci_vendor_id;     /* 0 if not reported */
    char     pci[16];           /* "0000:01:00.0", or "" if unknown */
    int      pci_no_domain;     /* the API gave no PCI domain; 0 assumed */
    int      is_gpu;
    unsigned compute_units;
    unsigned clock_mhz;
    size_t   max_work_group;
    uint64_t global_mem;
    uint64_t max_alloc;
    uint64_t local_mem;         /* work-group scratch available */
    char     driver[64];        /* driver version string */
    char     platform[128];     /* OpenCL platform and version; "" for CUDA */
    char     api_version[64];   /* OpenCL device version, or CUDA's */
    char     arch[16];          /* "sm_120"; "" where there is no such name */
    int      handle;            /* backend-private index of the device */
} vb_dev_info;

/*
 * One physical device, with its view through each backend that reaches it.
 * The list is in OpenCL's enumeration order, so --device indices mean what
 * they always did; a device only another API can see comes after.
 */
typedef struct {
    int         present[VB_BACKEND_COUNT];
    vb_dev_info via[VB_BACKEND_COUNT];
} vb_device;

int  vb_devices(vb_device *out, int max);

/* The first backend's view that exists -- for naming a device. */
const vb_dev_info *vb_device_info(const vb_device *d);

/* Why a backend has no devices, for --list-devices and --verbose; or NULL. */
const char *vb_backend_unavailable(vb_backend_id b);

const char *vb_backend_name(vb_backend_id b);

/* Choices that reach the device compiler and the launch. */
typedef struct {
    int             neutral;      /* --primitives neutral: no steers */
    const char     *dump_dir;     /* --dump-device-code, or NULL */
    size_t          pin_global;   /* --device-geometry; 0 means tune */
    size_t          pin_local;
    uint32_t        iterations;   /* the count being measured */
    vb_compile_mode compile_mode; /* CUDA only */
} vb_dev_options;

void vb_dev_options_default(vb_dev_options *o);

typedef struct vb_dev_ctx vb_dev_ctx;

/* What each API implements. Every entry returns 0 or -1 with c->error set. */
typedef struct {
    vb_backend_id id;
    const char   *name;           /* "opencl", "cuda" */
    vb_dialect    dialect;

    int         (*enumerate)(vb_dev_info *out, int max);
    const char *(*unavailable)(void);

    /* Context, queue or stream, and compile `source` -- a complete program
       already composed from the shared sources -- with `defines` appended to
       the compiler's options, then look up `entry`. Records what compiled it
       in c->compiler, c->compiler_version and c->compile_mode, and writes the
       compiled code to o->dump_dir if one is given. */
    int         (*build)(vb_dev_ctx *c, const char *source, const char *entry,
                         const char *defines, const vb_dev_options *o);

    /* The largest work-group this kernel may launch with on this device. */
    size_t      (*max_local)(const vb_dev_ctx *c);

    /* Device buffers for the corpus slice (uploaded now) and the partials. */
    int         (*alloc)(vb_dev_ctx *c, const void *slice, size_t corpus_bytes,
                         size_t partial_bytes);

    /* Pinned host staging for streaming; failure falls back to pageable. */
    void        (*pin_staging)(vb_dev_ctx *c);

    /* Asynchronously: re-upload the slice if streaming, then launch
       global/local work-items with `shared` bytes of work-group scratch. */
    int         (*launch)(vb_dev_ctx *c, uint32_t iterations, size_t shared);

    /* Wait, read `bytes` of partials into c->partials, and set
       c->last_kernel_ns and c->last_transfer_ns from device events. */
    int         (*read)(vb_dev_ctx *c, size_t bytes);

    void        (*destroy)(vb_dev_ctx *c);
} vb_dev_backend;

const vb_dev_backend *vb_backend(vb_backend_id b);

struct vb_dev_ctx {
    const vb_dev_backend *be;
    vb_dev_info dev;
    void       *impl;             /* backend state */

    unsigned  lanes;              /* corpus interleave width */
    /* Digest shape of the work-group partial: 4x4 md5, 5x4 sha1, 8x8 sha512.
       Sizes the readback, the work-group scratch and the final fold. */
    unsigned  partial_words;
    unsigned  partial_word_bytes;
    unsigned  streams;            /* messages each work-item hashes per group */
    uint32_t  blocks;             /* per message */
    uint64_t  first_group;        /* this device's slice within the corpus */
    uint64_t  n_groups;
    uint64_t  n_messages;

    /* Launch geometry, chosen by measurement unless pinned, and independent of
       the corpus: each work-item strides over as many groups as needed. */
    size_t    global_size;
    size_t    local_size;
    int       geometry_pinned;

    /* Sweeps of the corpus per launch, always odd, so a small working set can
       still saturate the device and launch overhead is amortised. */
    uint32_t  repeats;

    uint64_t  n_partials;         /* one digest per work-group in the launch */
    uint64_t  partial_cap;        /* how many the buffers hold */
    void     *partials;           /* host copy of the readback */

    /* Streaming: re-upload the slice before every launch, so the link is
       inside the timed region. */
    int         stream;
    int         host_pinned;      /* streaming reads pinned memory */
    const void *host_slice;       /* not owned; the corpus outlives this */
    size_t      corpus_bytes;

    uint64_t  last_kernel_ns;
    uint64_t  last_transfer_ns;   /* 0 unless streaming */

    char      label[48];          /* "md5-s1": names dumped device code */

    /* What compiled the kernel, for the result. */
    char      compiler[64];       /* the OpenCL platform, or "nvrtc" */
    char      compiler_version[96];
    char      compile_mode[16];   /* "driver", "ptx-jit", "cubin" */
    char      steers[128];        /* e.g. "rotl32=ptx", or "" */

    char      error[512];
};

/*
 * Build the program for this (lanes, streams) pair on `dev` through `be`, upload
 * the slice of the corpus spanning groups [first_group, first_group +
 * n_groups), and choose the launch geometry and repeat count. Passing the whole
 * corpus is the single-device case.
 *
 * Returns 0; on failure ctx->error explains why and nothing leaks.
 */
int  vb_dev_ctx_init(vb_dev_ctx *c, const vb_dev_backend *be,
                     const vb_dev_info *dev, const vb_corpus *corpus,
                     unsigned streams, uint64_t first_group, uint64_t n_groups,
                     const vb_dev_options *o);

/*
 * Streaming on or off, after init. Geometry and repeats are tuned against the
 * kernel alone -- tuning with the upload inside would measure the upload -- so
 * this flips the mode afterwards. On forces repeats to 1: repeats amplify
 * compute without amplifying transfer, so any more would inflate the compute
 * side of the ratio streaming exists to measure.
 */
void vb_dev_ctx_set_stream(vb_dev_ctx *c, int on, int pinned);

/* Launch once and fold the partials. The split lets several devices run
   concurrently: enqueue on every device, then collect from each. */
int  vb_dev_ctx_run(vb_dev_ctx *c, uint32_t iterations,
                    uint64_t checksum[VB_MAX_DIGEST_WORDS]);
int  vb_dev_ctx_enqueue(vb_dev_ctx *c, uint32_t iterations);
int  vb_dev_ctx_collect(vb_dev_ctx *c, uint64_t checksum[VB_MAX_DIGEST_WORDS]);

void vb_dev_ctx_free(vb_dev_ctx *c);

/* The i-th of n contiguous, near-equal slices of total_groups. */
void vb_dev_slice(uint64_t total_groups, int n, int i,
                  uint64_t *first, uint64_t *count);

#endif /* VALUBENCH_DEVICE_H */
