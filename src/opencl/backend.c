/*
 * backend.c -- the OpenCL device backend.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Only what the OpenCL API itself does: enumerate devices, compile a program,
 * allocate and upload, launch, read back, time with events. The program text,
 * the launch geometry, the repeat count, the partial fold and the timed loop
 * are src/device/device.c's, shared with every other backend -- which is what
 * makes an OpenCL figure comparable with any other API's.
 *
 * This file knows nothing about hash functions either: the program arrives
 * composed, and the digest shape comes from the context.
 */

#define _GNU_SOURCE

#include "device.h"
#include "opencl.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    cl_device_id     device;
    cl_mem           d_corpus;
    cl_mem           d_partial;
    cl_mem           h_staging;   /* pinned staging for streaming, or NULL */
    void            *staging;     /* its mapped pointer */
    cl_event         event;       /* in-flight launch */
    cl_event         xfer_event;  /* in-flight upload, streaming only */

    /* Pipelined streaming: uploads on their own queue, since an in-order
       queue would run them strictly between kernels. */
    cl_command_queue copy_queue;
    cl_mem           pbuf[2];     /* chunk buffers, alternating */
    cl_mem           ppart[2];    /* their partials */
    cl_mem           h_ring;      /* pinned readback ring, mapped */
    unsigned char   *ring;
    int              ring_mapped;
    size_t           ring_slot;
    cl_event         up[VB_PIPE_RING], kev[VB_PIPE_RING], rd[VB_PIPE_RING];
} ocl_impl;

static void set_err(vb_dev_ctx *c, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->error, sizeof c->error, fmt, ap);
    va_end(ap);
}

/* ---- devices ------------------------------------------------------------ */

/* Enumerated once: OpenCL device ids are stable for the life of a process,
   and the backend's handle is an index into this. */
static vb_ocl_device g_devs[VB_OCL_MAX_DEVICES];
static int           g_ndevs = -1;

static int ocl_enumerate(vb_dev_info *out, int max)
{
    if (g_ndevs < 0)
        g_ndevs = vb_ocl_devices(g_devs, VB_OCL_MAX_DEVICES);

    int n = 0;
    for (int i = 0; i < g_ndevs && n < max; i++, n++) {
        const vb_ocl_device *d = &g_devs[i];
        vb_dev_info *o = &out[n];

        memset(o, 0, sizeof *o);
        o->backend = VB_BACKEND_OPENCL;
        o->ordinal = i;
        o->handle = i;
        snprintf(o->name, sizeof o->name, "%s", d->name);
        snprintf(o->vendor, sizeof o->vendor, "%s", d->vendor);
        o->pci_vendor_id = d->vendor_id;
        snprintf(o->pci, sizeof o->pci, "%s", d->pci);
        o->pci_no_domain = d->pci_no_domain;
        o->is_gpu = (d->type & CL_DEVICE_TYPE_GPU) != 0;
        o->is_cpu = (d->type & CL_DEVICE_TYPE_CPU) != 0;
        o->compute_units = d->compute_units;
        o->clock_mhz = d->clock_mhz;
        o->max_work_group = d->max_work_group;
        o->global_mem = d->global_mem;
        o->max_alloc = d->max_alloc;
        o->local_mem = d->local_mem;
        snprintf(o->driver, sizeof o->driver, "%s", d->driver_version);
        snprintf(o->platform, sizeof o->platform, "%s (%s)", d->platform_name,
                 d->platform_version);
        snprintf(o->api_version, sizeof o->api_version, "%s",
                 d->device_version);
    }
    return n;
}

static const char *ocl_unavailable(void)
{
    return vb_ocl_error();
}

/* ---- compile ------------------------------------------------------------ */

/* Write `len` bytes to dir/name, best-effort: a dump that cannot be written
   is reported and does not fail the run. */
static void dump_file(const char *dir, const char *name, const void *data,
                      size_t len)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "valubench: cannot write %s\n", path);
        return;
    }
    fwrite(data, 1, len, f);
    fclose(f);
}

/* The program binary, which on NVIDIA is PTX. Caller frees. */
static unsigned char *program_binary(const vb_ocl *cl, cl_program p,
                                     size_t *len)
{
    size_t size = 0;
    *len = 0;
    if (cl->GetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof size, &size,
                           NULL) != CL_SUCCESS || size == 0)
        return NULL;
    unsigned char *bin = malloc(size + 1);
    if (!bin)
        return NULL;
    unsigned char *bins[1] = { bin };
    if (cl->GetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof bins, bins,
                           NULL) != CL_SUCCESS) {
        free(bin);
        return NULL;
    }
    bin[size] = '\0';
    while (size > 0 && bin[size - 1] == '\0')     /* PTX arrives NUL-terminated */
        size--;
    *len = size;
    return bin;
}

static char *build_log(const vb_ocl *cl, cl_program p, cl_device_id d)
{
    size_t n = 0;
    cl->GetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
    char *log = malloc(n + 1);
    if (!log)
        return NULL;
    if (cl->GetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, n, log, NULL)
        != CL_SUCCESS)
        n = 0;
    log[n] = '\0';
    return log;
}

static void dump(const vb_dev_ctx *c, const vb_dev_options *o,
                 const char *source, const unsigned char *bin, size_t bin_len,
                 int is_ptx, const char *opts, const char *log)
{
    char name[256], hdr[768];

    snprintf(name, sizeof name, "%s.opencl.d%d.cl", c->label, c->dev.ordinal);
    dump_file(o->dump_dir, name, source, strlen(source));
    if (bin) {
        snprintf(name, sizeof name, "%s.opencl.d%d.%s", c->label,
                 c->dev.ordinal, is_ptx ? "ptx" : "bin");
        dump_file(o->dump_dir, name, bin, bin_len);
    }
    snprintf(name, sizeof name, "%s.opencl.d%d.log", c->label, c->dev.ordinal);
    int h = snprintf(hdr, sizeof hdr, "options: %s\ncompiler: %s, %s\n\n",
                     opts, c->compiler, c->compiler_version);
    if (h < 0)
        return;
    if ((size_t) h >= sizeof hdr)
        h = (int) sizeof hdr - 1;
    size_t loglen = log ? strlen(log) : 0;
    char *both = malloc((size_t) h + loglen + 1);
    if (!both)
        return;
    memcpy(both, hdr, (size_t) h);
    memcpy(both + h, log ? log : "", loglen + 1);
    dump_file(o->dump_dir, name, both, (size_t) h + loglen);
    free(both);
}

static int ocl_build(vb_dev_ctx *c, const char *source, const char *entry,
                     const char *defines, const vb_dev_options *o)
{
    const vb_ocl *cl = vb_ocl_api();
    cl_int err;

    if (!cl || c->dev.handle < 0 || c->dev.handle >= g_ndevs) {
        set_err(c, "OpenCL not loaded");
        return -1;
    }
    ocl_impl *m = calloc(1, sizeof *m);
    if (!m) {
        set_err(c, "out of memory");
        return -1;
    }
    c->impl = m;
    m->device = g_devs[c->dev.handle].device;

    m->context = cl->CreateContext(NULL, 1, &m->device, NULL, NULL, &err);
    if (!m->context) {
        set_err(c, "clCreateContext: %s", vb_ocl_strerror(err));
        return -1;
    }
    /* Profiling reports device time next to wall time, so launch and
       synchronisation overhead is visible rather than folded into the
       throughput figure. */
    m->queue = cl->CreateCommandQueue(m->context, m->device,
                                      CL_QUEUE_PROFILING_ENABLE, &err);
    if (!m->queue) {
        set_err(c, "clCreateCommandQueue: %s", vb_ocl_strerror(err));
        return -1;
    }

    const char *srcs[1] = { source };
    m->program = cl->CreateProgramWithSource(m->context, 1, srcs, NULL, &err);
    if (!m->program) {
        set_err(c, "clCreateProgramWithSource: %s", vb_ocl_strerror(err));
        return -1;
    }

    /* -cl-nv-verbose puts NVIDIA's register report in the build log and
       changes nothing in the code; only worth asking for when it is kept. */
    int nvidia = vb_vendor_classify(c->dev.vendor, c->dev.pci_vendor_id)
                 == VB_VENDOR_NVIDIA;
    char opts[512];
    snprintf(opts, sizeof opts, "%s -cl-std=CL1.2%s", defines,
             o->dump_dir && nvidia ? " -cl-nv-verbose" : "");

    err = cl->BuildProgram(m->program, 1, &m->device, opts, NULL, NULL);
    char *log = build_log(cl, m->program, m->device);
    if (err != CL_SUCCESS) {
        set_err(c, "kernel build failed (%s): %.400s", vb_ocl_strerror(err),
                log ? log : "");
        free(log);
        return -1;
    }

    size_t bin_len = 0;
    unsigned char *bin = program_binary(cl, m->program, &bin_len);
    int is_ptx = bin && strstr((const char *) bin, ".version") &&
                 strstr((const char *) bin, ".target");

    /* What compiled it. The OpenCL compiler lives in the driver, so the
       platform names it; on NVIDIA the PTX also names the NVVM frontend,
       which is the part that differs from CUDA's. */
    const vb_ocl_device *dv = &g_devs[c->dev.handle];
    snprintf(c->compiler, sizeof c->compiler, "%s", dv->platform_name);
    snprintf(c->compile_mode, sizeof c->compile_mode, "driver");
    const char *nvvm = is_ptx ? strstr((const char *) bin, "Based on NVVM ")
                              : NULL;
    if (nvvm) {
        int len = (int) strcspn(nvvm + 9, "\r\n");
        snprintf(c->compiler_version, sizeof c->compiler_version,
                 "%.*s, driver %s", len, nvvm + 9, dv->driver_version);
    } else {
        snprintf(c->compiler_version, sizeof c->compiler_version,
                 "%s, driver %s", dv->platform_version, dv->driver_version);
    }

    if (o->dump_dir)
        dump(c, o, source, bin, bin_len, is_ptx, opts, log);
    free(bin);
    free(log);

    m->kernel = cl->CreateKernel(m->program, entry, &err);
    if (!m->kernel) {
        set_err(c, "clCreateKernel: %s", vb_ocl_strerror(err));
        return -1;
    }
    return 0;
}

/*
 * What this kernel may launch with, which can be less than the device's
 * general maximum. clGetKernelWorkGroupInfo was once loaded and never called,
 * so a register-poor device that caps SHA-512 s4 below 64 was offered nothing
 * it could accept.
 */
static size_t ocl_max_local(const vb_dev_ctx *c)
{
    const vb_ocl *cl = vb_ocl_api();
    const ocl_impl *m = c->impl;
    size_t kmax = 0;
    if (cl->GetKernelWorkGroupInfo(m->kernel, m->device,
                                   CL_KERNEL_WORK_GROUP_SIZE, sizeof kmax,
                                   &kmax, NULL) != CL_SUCCESS)
        return 0;
    return kmax;
}

/* ---- buffers ------------------------------------------------------------ */

static int ocl_alloc(vb_dev_ctx *c, const void *slice, size_t corpus_bytes,
                     size_t partial_bytes)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    cl_int err;

    m->d_corpus = cl->CreateBuffer(m->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   corpus_bytes, (void *) (uintptr_t) slice,
                                   &err);
    if (!m->d_corpus) {
        set_err(c, "corpus upload (%.1f MiB): %s",
                (double) corpus_bytes / 1048576.0, vb_ocl_strerror(err));
        return -1;
    }
    m->d_partial = cl->CreateBuffer(m->context, CL_MEM_WRITE_ONLY,
                                    partial_bytes, NULL, &err);
    if (!m->d_partial) {
        set_err(c, "partial buffer: %s", vb_ocl_strerror(err));
        return -1;
    }
    return 0;
}

/*
 * Pinned staging for streaming. OpenCL has no pinned allocator as such; the
 * portable route, and the one NVIDIA documents, is a buffer the driver
 * allocates (CL_MEM_ALLOC_HOST_PTR), mapped once for the life of the context.
 * On failure everything it made is released and streaming reads pageable
 * memory, which host_pinned then records.
 */
static void ocl_pin_staging(vb_dev_ctx *c)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    cl_int err;

    m->h_staging = cl->CreateBuffer(m->context,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    c->corpus_bytes, NULL, &err);
    if (!m->h_staging)
        return;
    m->staging = cl->EnqueueMapBuffer(m->queue, m->h_staging, CL_TRUE,
                                      CL_MAP_WRITE, 0, c->corpus_bytes, 0,
                                      NULL, NULL, &err);
    if (!m->staging) {
        cl->ReleaseMemObject(m->h_staging);
        m->h_staging = NULL;
        return;
    }
    memcpy(m->staging, c->host_slice, c->corpus_bytes);
    c->host_pinned = 1;
}

/* ---- launch and read ---------------------------------------------------- */

static int ocl_launch(vb_dev_ctx *c, uint32_t iterations, size_t shared)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    cl_int err;
    size_t global = c->global_size, local = c->local_size;

    cl_uint a = 0;
    err  = cl->SetKernelArg(m->kernel, a++, sizeof m->d_corpus, &m->d_corpus);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof(cl_uint), &c->blocks);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof(cl_uint), &iterations);
    cl_ulong groups = c->n_groups;
    err |= cl->SetKernelArg(m->kernel, a++, sizeof groups, &groups);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof(cl_uint), &c->repeats);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof m->d_partial, &m->d_partial);
    /* The work-group scratch, one digest per work-item. */
    err |= cl->SetKernelArg(m->kernel, a++, shared, NULL);
    if (err != CL_SUCCESS) {
        set_err(c, "clSetKernelArg: %s", vb_ocl_strerror(err));
        return -1;
    }

    if (m->event) {
        cl->ReleaseEvent(m->event);
        m->event = NULL;
    }
    if (m->xfer_event) {
        cl->ReleaseEvent(m->xfer_event);
        m->xfer_event = NULL;
    }

    /*
     * Streaming: put the corpus back across the link before every launch, so
     * the host-to-device copy is inside the timed region. Non-blocking on an
     * in-order queue, so the kernel waits for it without the host blocking,
     * and each is timed by its own event rather than by wall clock.
     */
    if (c->stream) {
        const void *src = m->staging ? m->staging : c->host_slice;
        err = cl->EnqueueWriteBuffer(m->queue, m->d_corpus, CL_FALSE, 0,
                                     c->corpus_bytes, (void *) (uintptr_t) src,
                                     0, NULL, &m->xfer_event);
        if (err != CL_SUCCESS) {
            set_err(c, "corpus upload (%.1f MiB): %s",
                    (double) c->corpus_bytes / 1048576.0,
                    vb_ocl_strerror(err));
            return -1;
        }
    }

    err = cl->EnqueueNDRangeKernel(m->queue, m->kernel, 1, NULL, &global,
                                   &local, 0, NULL, &m->event);
    if (err != CL_SUCCESS) {
        set_err(c, "clEnqueueNDRangeKernel (global=%zu local=%zu): %s",
                global, local, vb_ocl_strerror(err));
        return -1;
    }
    /* Push it to the device now rather than at the blocking read, so several
       devices actually overlap instead of starting one at a time. */
    cl->Flush(m->queue);
    return 0;
}

/* Elapsed device time for a completed event, then release it. 0 if the
   driver declines profiling, which is not an error. */
static uint64_t event_ns(const vb_ocl *cl, cl_event *ev)
{
    uint64_t ns = 0;
    if (*ev) {
        cl_ulong t0 = 0, t1 = 0;
        if (cl->GetEventProfilingInfo(*ev, CL_PROFILING_COMMAND_START,
                                      sizeof t0, &t0, NULL) == CL_SUCCESS &&
            cl->GetEventProfilingInfo(*ev, CL_PROFILING_COMMAND_END,
                                      sizeof t1, &t1, NULL) == CL_SUCCESS &&
            t1 > t0)
            ns = (uint64_t) (t1 - t0);
        cl->ReleaseEvent(*ev);
        *ev = NULL;
    }
    return ns;
}

static int ocl_read(vb_dev_ctx *c, size_t bytes)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    cl_int err = cl->EnqueueReadBuffer(m->queue, m->d_partial, CL_TRUE, 0,
                                       bytes, c->partials, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_err(c, "reading partials: %s", vb_ocl_strerror(err));
        return -1;
    }
    c->last_kernel_ns = event_ns(cl, &m->event);
    c->last_transfer_ns = event_ns(cl, &m->xfer_event);
    return 0;
}

/* ---- pipelined streaming ------------------------------------------------ */

static int ocl_pipe_open(vb_dev_ctx *c, size_t chunk_bytes, size_t read_bytes)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    cl_int err;

    m->copy_queue = cl->CreateCommandQueue(m->context, m->device,
                                           CL_QUEUE_PROFILING_ENABLE, &err);
    if (!m->copy_queue) {
        set_err(c, "second command queue: %s", vb_ocl_strerror(err));
        return -1;
    }
    for (int b = 0; b < 2; b++) {
        m->pbuf[b] = cl->CreateBuffer(m->context, CL_MEM_READ_ONLY, chunk_bytes,
                                      NULL, &err);
        if (!m->pbuf[b]) {
            set_err(c, "chunk buffer (%.1f MiB): %s",
                    (double) chunk_bytes / 1048576.0, vb_ocl_strerror(err));
            return -1;
        }
        m->ppart[b] = cl->CreateBuffer(m->context, CL_MEM_WRITE_ONLY, read_bytes,
                                       NULL, &err);
        if (!m->ppart[b]) {
            set_err(c, "chunk partial buffer: %s", vb_ocl_strerror(err));
            return -1;
        }
    }

    /* The readbacks land in pinned memory too: a non-blocking read into
       pageable memory can stall the host until it completes, and the host
       is what keeps the queues fed. Pageable if pinning is refused. */
    m->ring_slot = read_bytes;
    m->h_ring = cl->CreateBuffer(m->context,
                                 CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                 VB_PIPE_RING * read_bytes, NULL, &err);
    if (m->h_ring)
        m->ring = cl->EnqueueMapBuffer(m->queue, m->h_ring, CL_TRUE,
                                       CL_MAP_READ | CL_MAP_WRITE, 0,
                                       VB_PIPE_RING * read_bytes, 0, NULL, NULL,
                                       &err);
    if (m->ring) {
        m->ring_mapped = 1;
    } else {
        if (m->h_ring)
            cl->ReleaseMemObject(m->h_ring);
        m->h_ring = NULL;
        m->ring = malloc(VB_PIPE_RING * read_bytes);
        if (!m->ring) {
            set_err(c, "out of memory for the readback ring");
            return -1;
        }
    }
    return 0;
}

static int ocl_pipe_enqueue(vb_dev_ctx *c, uint64_t seq, size_t offset,
                            size_t bytes, uint64_t n_groups,
                            uint32_t iterations, size_t shared,
                            size_t read_bytes)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    cl_int err;
    unsigned i = (unsigned) (seq % VB_PIPE_RING);
    int b = (int) (seq % 2);

    /* The buffer's previous user is chunk seq - 2: its kernel must be done
       reading before the upload overwrites it. */
    const unsigned char *src = (const unsigned char *)
        (m->staging ? m->staging : c->host_slice) + offset;
    cl_event wait_kernel[1];
    cl_uint n_wait = 0;
    if (seq >= 2)
        wait_kernel[n_wait++] = m->kev[(seq - 2) % VB_PIPE_RING];
    err = cl->EnqueueWriteBuffer(m->copy_queue, m->pbuf[b], CL_FALSE, 0, bytes,
                                 (void *) (uintptr_t) src, n_wait,
                                 n_wait ? wait_kernel : NULL, &m->up[i]);
    if (err != CL_SUCCESS) {
        set_err(c, "chunk upload: %s", vb_ocl_strerror(err));
        return -1;
    }
    cl->Flush(m->copy_queue);

    /* Arguments are captured when the kernel is enqueued, so resetting them
       for the next chunk is safe while this one waits. */
    cl_uint a = 0;
    cl_ulong groups = n_groups;
    err  = cl->SetKernelArg(m->kernel, a++, sizeof m->pbuf[b], &m->pbuf[b]);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof(cl_uint), &c->blocks);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof(cl_uint), &iterations);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof groups, &groups);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof(cl_uint), &c->repeats);
    err |= cl->SetKernelArg(m->kernel, a++, sizeof m->ppart[b], &m->ppart[b]);
    err |= cl->SetKernelArg(m->kernel, a++, shared, NULL);
    if (err != CL_SUCCESS) {
        set_err(c, "clSetKernelArg: %s", vb_ocl_strerror(err));
        return -1;
    }
    size_t global = c->global_size, local = c->local_size;
    err = cl->EnqueueNDRangeKernel(m->queue, m->kernel, 1, NULL, &global,
                                   &local, 1, &m->up[i], &m->kev[i]);
    if (err != CL_SUCCESS) {
        set_err(c, "clEnqueueNDRangeKernel (chunk): %s", vb_ocl_strerror(err));
        return -1;
    }
    err = cl->EnqueueReadBuffer(m->queue, m->ppart[b], CL_FALSE, 0, read_bytes,
                                m->ring + (size_t) i * m->ring_slot, 0, NULL,
                                &m->rd[i]);
    if (err != CL_SUCCESS) {
        set_err(c, "reading chunk partials: %s", vb_ocl_strerror(err));
        return -1;
    }
    cl->Flush(m->queue);
    return 0;
}

static int ocl_pipe_wait(vb_dev_ctx *c, uint64_t seq, const void **partials,
                         uint64_t *kernel_ns, uint64_t *transfer_ns)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    unsigned i = (unsigned) (seq % VB_PIPE_RING);

    cl_int err = cl->WaitForEvents(1, &m->rd[i]);
    if (err != CL_SUCCESS) {
        set_err(c, "waiting for chunk %llu: %s", (unsigned long long) seq,
                vb_ocl_strerror(err));
        return -1;
    }
    cl->ReleaseEvent(m->rd[i]);
    m->rd[i] = NULL;
    *kernel_ns = event_ns(cl, &m->kev[i]);
    *transfer_ns = event_ns(cl, &m->up[i]);
    *partials = m->ring + (size_t) i * m->ring_slot;
    return 0;
}

static void ocl_destroy(vb_dev_ctx *c)
{
    const vb_ocl *cl = vb_ocl_api();
    ocl_impl *m = c->impl;
    if (!m)
        return;
    if (cl) {
        if (m->copy_queue)
            cl->Finish(m->copy_queue);
        if (m->queue)
            cl->Finish(m->queue);
        for (int i = 0; i < VB_PIPE_RING; i++) {
            if (m->up[i])  cl->ReleaseEvent(m->up[i]);
            if (m->kev[i]) cl->ReleaseEvent(m->kev[i]);
            if (m->rd[i])  cl->ReleaseEvent(m->rd[i]);
        }
        if (m->ring_mapped && m->queue) {
            cl->EnqueueUnmapMemObject(m->queue, m->h_ring, m->ring, 0, NULL,
                                      NULL);
            cl->Finish(m->queue);
        } else if (!m->ring_mapped) {
            free(m->ring);
        }
        if (m->h_ring)     cl->ReleaseMemObject(m->h_ring);
        for (int b = 0; b < 2; b++) {
            if (m->pbuf[b])  cl->ReleaseMemObject(m->pbuf[b]);
            if (m->ppart[b]) cl->ReleaseMemObject(m->ppart[b]);
        }
        if (m->copy_queue) cl->ReleaseCommandQueue(m->copy_queue);
        if (m->event)      cl->ReleaseEvent(m->event);
        if (m->xfer_event) cl->ReleaseEvent(m->xfer_event);
        if (m->staging && m->queue) {
            cl->EnqueueUnmapMemObject(m->queue, m->h_staging, m->staging, 0,
                                      NULL, NULL);
            cl->Finish(m->queue);
        }
        if (m->h_staging) cl->ReleaseMemObject(m->h_staging);
        if (m->kernel)    cl->ReleaseKernel(m->kernel);
        if (m->program)   cl->ReleaseProgram(m->program);
        if (m->d_corpus)  cl->ReleaseMemObject(m->d_corpus);
        if (m->d_partial) cl->ReleaseMemObject(m->d_partial);
        if (m->queue)     cl->ReleaseCommandQueue(m->queue);
        if (m->context)   cl->ReleaseContext(m->context);
    }
    free(m);
    c->impl = NULL;
}

const vb_dev_backend vb_opencl_backend = {
    .id          = VB_BACKEND_OPENCL,
    .name        = "opencl",
    .dialect     = VB_DIALECT_OPENCL,
    .enumerate   = ocl_enumerate,
    .unavailable = ocl_unavailable,
    .build       = ocl_build,
    .max_local   = ocl_max_local,
    .alloc       = ocl_alloc,
    .pin_staging = ocl_pin_staging,
    .launch      = ocl_launch,
    .read        = ocl_read,
    .pipe_open   = ocl_pipe_open,
    .pipe_enqueue = ocl_pipe_enqueue,
    .pipe_wait   = ocl_pipe_wait,
    .destroy     = ocl_destroy,
};
