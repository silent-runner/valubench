/*
 * backend.c -- the CUDA device backend.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * The CUDA counterpart of src/opencl/backend.c, and deliberately no more than
 * that: enumerate, compile, allocate, upload, launch, read back, time with
 * events. The program -- the same algorithm core OpenCL compiles, under
 * dialect_cuda.h -- the launch geometry, the repeat count and the timed loop
 * all come from src/device/device.c, so a CUDA figure differs from an OpenCL
 * one only by what the two APIs themselves do.
 *
 * The program is compiled by NVRTC at run time for the device's own
 * architecture, in one of two modes that must never be conflated:
 *
 *   ptx-jit  NVRTC to PTX, which the driver's JIT turns into machine code --
 *            the same final stage NVIDIA's OpenCL uses, so against OpenCL this
 *            isolates the language frontend.
 *   cubin    NVRTC to machine code with the toolkit's own ptxas, a
 *            different compiler version from the driver's.
 */

#define _GNU_SOURCE

#include "device.h"
#include "cuda_loader.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    vb_CUdevice    device;
    vb_CUcontext   ctx;
    int            ctx_retained;
    vb_CUmodule    module;
    vb_CUfunction  fn;
    vb_CUstream    stream;
    vb_CUevent     k0, k1, x0, x1;
    int            streamed;          /* the last launch recorded x0/x1 */
    vb_CUdeviceptr d_corpus;
    vb_CUdeviceptr d_partial;
    void          *staging;           /* pinned host copy of the slice */

    /* An imported OpenCL kernel takes its work-group scratch as a seventh,
       pointer argument -- the shared-window address of the dynamic region,
       which is what OpenCL's runtime passes for a __local argument. */
    int                ocl_abi;
    unsigned long long shared_base;

    /* Pipelined streaming: uploads on their own stream. */
    vb_CUstream    cstream;
    vb_CUdeviceptr pbuf[2], ppart[2];
    unsigned char *ring;
    size_t         ring_slot;
    int            ring_pinned;
    vb_CUevent     pu0[VB_PIPE_RING], pu1[VB_PIPE_RING];
    vb_CUevent     pk0[VB_PIPE_RING], pk1[VB_PIPE_RING], prd[VB_PIPE_RING];
} cu_impl;

static void set_err(vb_dev_ctx *c, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->error, sizeof c->error, fmt, ap);
    va_end(ap);
}

/* Every call below goes through the device's own context, since one host
   thread drives several devices in turn. */
static int enter(vb_dev_ctx *c)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    vb_CUresult r = cu->CtxSetCurrent(m->ctx);
    if (r != VB_CUDA_SUCCESS) {
        set_err(c, "cuCtxSetCurrent: %s", vb_cuda_strerror(r));
        return -1;
    }
    return 0;
}

#define CU_CHECK(c, call, what)                                            \
    do {                                                                   \
        vb_CUresult r_ = (call);                                           \
        if (r_ != VB_CUDA_SUCCESS) {                                       \
            set_err((c), "%s: %s", (what), vb_cuda_strerror(r_));          \
            return -1;                                                     \
        }                                                                  \
    } while (0)

/* ---- devices ------------------------------------------------------------ */

static int cu_enumerate(vb_dev_info *out, int max)
{
    if (!vb_cuda_load())
        return 0;
    const vb_cuda *cu = vb_cuda_api();

    int count = 0;
    if (cu->DeviceGetCount(&count) != VB_CUDA_SUCCESS)
        return 0;

    int n = 0;
    for (int i = 0; i < count && n < max; i++) {
        vb_CUdevice d;
        if (cu->DeviceGet(&d, i) != VB_CUDA_SUCCESS)
            continue;

        vb_dev_info *o = &out[n];
        memset(o, 0, sizeof *o);
        o->backend = VB_BACKEND_CUDA;
        o->ordinal = i;
        o->handle = d;
        o->is_gpu = 1;
        o->pci_vendor_id = 0x10de;
        snprintf(o->vendor, sizeof o->vendor, "NVIDIA Corporation");
        if (cu->DeviceGetName(o->name, (int) sizeof o->name - 1, d)
            != VB_CUDA_SUCCESS)
            snprintf(o->name, sizeof o->name, "CUDA device %d", i);

        int v = 0, major = 0, minor = 0, dom = 0, bus = 0, dev = 0;
        if (cu->DeviceGetAttribute(&v, VB_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                   d) == VB_CUDA_SUCCESS)
            o->compute_units = (unsigned) v;
        if (cu->DeviceGetAttribute(&v, VB_CU_DEVICE_ATTRIBUTE_CLOCK_RATE, d)
            == VB_CUDA_SUCCESS)
            o->clock_mhz = (unsigned) (v / 1000);
        if (cu->DeviceGetAttribute(&v, VB_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                   d) == VB_CUDA_SUCCESS)
            o->max_work_group = (size_t) v;
        if (cu->DeviceGetAttribute(&v,
                                   VB_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                   d) == VB_CUDA_SUCCESS)
            o->local_mem = (uint64_t) v;
        cu->DeviceGetAttribute(&major,
                               VB_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
        cu->DeviceGetAttribute(&minor,
                               VB_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);
        snprintf(o->arch, sizeof o->arch, "sm_%d%d", major, minor);

        size_t mem = 0;
        if (cu->DeviceTotalMem(&mem, d) == VB_CUDA_SUCCESS)
            o->global_mem = o->max_alloc = (uint64_t) mem;

        /* CUDA reports domain, bus and device but not the function, which is
           0 for every GPU it can enumerate. */
        if (cu->DeviceGetAttribute(&dom, VB_CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, d)
                == VB_CUDA_SUCCESS &&
            cu->DeviceGetAttribute(&bus, VB_CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, d)
                == VB_CUDA_SUCCESS &&
            cu->DeviceGetAttribute(&dev, VB_CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, d)
                == VB_CUDA_SUCCESS)
            snprintf(o->pci, sizeof o->pci, "%04x:%02x:%02x.0",
                     (unsigned) dom & 0xffffu, (unsigned) bus & 0xffu,
                     (unsigned) dev & 0x1fu);

        snprintf(o->driver, sizeof o->driver, "%s",
                 cu->kernel_driver[0] ? cu->kernel_driver : "unknown");
        snprintf(o->api_version, sizeof o->api_version, "CUDA %d.%d",
                 cu->driver_version / 1000, (cu->driver_version % 1000) / 10);
        n++;
    }
    return n;
}

static const char *cu_unavailable(void)
{
    vb_cuda_load();
    return vb_cuda_error();
}

/* ---- compile ------------------------------------------------------------ */

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

/* ---- importing OpenCL's PTX -------------------------------------------- */

/*
 * NVIDIA's OpenCL compiler emits PTX, and the CUDA driver can run it once
 * OpenCL's calling convention is translated -- which makes a 2x2 of frontend
 * against runtime, and lets Nsight Compute, which sees only CUDA, profile the
 * code OpenCL's compiler produced. The translation touches no hash code:
 *
 *   - kernel parameters carry OpenCL's state-space qualifiers (".ptr .global
 *     .align 4"), which the CUDA loader rejects as an invalid image; they are
 *     annotations, and the pointers are used as they are either way;
 *   - OpenCL reads its launch environment from %envreg registers the CUDA
 *     runtime never sets: %envreg0 and %envreg3 are the group and global
 *     offsets, zero for any launch here, and %envreg6 is the group count,
 *     which CUDA calls %nctaid.x. Any other %envreg is refused, not guessed;
 *   - the __local scratch argument stays, and the launch passes it the
 *     address of the dynamic shared region.
 *
 * Checked on an RTX PRO 2000 against NVRTC's kernel on the same data before
 * it was trusted, and the checksum gate checks it on every run after.
 */
static char *ocl_ptx_to_cuda(const char *in, int *ocl_abi, char *err,
                             size_t errn)
{
    size_t n = strlen(in);
    char *out = malloc(n + 64 * 1024);
    if (!out) {
        snprintf(err, errn, "out of memory");
        return NULL;
    }
    size_t o = 0;
    *ocl_abi = strstr(in, ".ptr .shared") != NULL;

    for (const char *p = in; *p; ) {
        if (!strncmp(p, " .ptr .", 7)) {
            /* " .ptr .<space> .align <n>" -- drop it. */
            const char *q = strstr(p, ".align ");
            if (q && q - p < 32) {
                q += 7;
                while (*q >= '0' && *q <= '9')
                    q++;
                p = q;
                continue;
            }
        }
        if (!strncmp(p, "%envreg", 7)) {
            int reg = atoi(p + 7);
            const char *rest = p + 7;
            while (*rest >= '0' && *rest <= '9')
                rest++;
            const char *with = NULL;
            if (reg == 0 || reg == 3)
                with = "0";
            else if (reg == 6)
                with = "%nctaid.x";
            if (!with) {
                snprintf(err, errn, "the PTX reads %%envreg%d, which has no "
                         "CUDA equivalent here", reg);
                free(out);
                return NULL;
            }
            /* mov.b32 from a special register wants the .u32 spelling. */
            if (reg == 6 && o >= 16) {
                char *mv = NULL;
                for (size_t k = o; k > 0 && out[k - 1] != '\n'; k--)
                    if (!strncmp(out + k - 1, "mov.b32", 7))
                        mv = out + k - 1;
                if (mv)
                    memcpy(mv, "mov.u32", 7);
            }
            size_t wl = strlen(with);
            memcpy(out + o, with, wl);
            o += wl;
            p = rest;
            continue;
        }
        out[o++] = *p++;
    }
    out[o] = '\0';
    return out;
}

/* Where the dynamic shared region starts in the shared window: the value
   OpenCL's runtime passes for a __local argument. Not zero -- recent parts
   reserve the first kilobyte -- so it is asked of the device. */
static int shared_base(vb_dev_ctx *c, const char *version_line,
                       unsigned long long *out)
{
    const vb_cuda *cu = vb_cuda_api();
    char ptx[640];
    snprintf(ptx, sizeof ptx,
             "%s\n.target %s\n.address_size 64\n"
             ".extern .shared .align 16 .b8 vb_dyn[];\n"
             ".visible .entry vb_shared_base(.param .u64 out)\n{\n"
             "\t.reg .b64 %%rd<3>;\n\tld.param.u64 %%rd1, [out];\n"
             "\tmov.u64 %%rd2, vb_dyn;\n\tst.global.u64 [%%rd1], %%rd2;\n"
             "\tret;\n}\n", version_line, c->dev.arch);
    vb_CUmodule mod = NULL;
    vb_CUfunction fn = NULL;
    vb_CUdeviceptr d = 0;
    vb_CUresult r = cu->ModuleLoadDataEx(&mod, ptx, 0, NULL, NULL);
    if (r == VB_CUDA_SUCCESS)
        r = cu->ModuleGetFunction(&fn, mod, "vb_shared_base");
    if (r == VB_CUDA_SUCCESS)
        r = cu->MemAlloc(&d, 8);
    void *args[1] = { &d };
    if (r == VB_CUDA_SUCCESS)
        r = cu->LaunchKernel(fn, 1, 1, 1, 1, 1, 1, 16, NULL, args, NULL);
    if (r == VB_CUDA_SUCCESS)
        r = cu->MemcpyDtoHAsync(out, d, 8, NULL);
    if (r == VB_CUDA_SUCCESS)
        r = cu->StreamSynchronize(NULL);
    if (d)
        cu->MemFree(d);
    if (mod)
        cu->ModuleUnload(mod);
    if (r != VB_CUDA_SUCCESS) {
        set_err(c, "finding the shared-memory base: %s", vb_cuda_strerror(r));
        return -1;
    }
    return 0;
}

static char *read_file(const char *path, char *err, size_t errn)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, errn, "cannot read %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char *buf = n >= 0 ? malloc((size_t) n + 1) : NULL;
    if (!buf || fread(buf, 1, (size_t) n, f) != (size_t) n) {
        snprintf(err, errn, "cannot read %s", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

/* NVRTC takes options as an argv, so split the device layer's -D string. */
static int split_options(char *buf, const char **argv, int max)
{
    int n = 0;
    for (char *tok = strtok(buf, " "); tok && n < max; tok = strtok(NULL, " "))
        argv[n++] = tok;
    return n;
}

static int cu_build(vb_dev_ctx *c, const char *source, const char *entry,
                    const char *defines, const vb_dev_options *o)
{
    const vb_cuda *cu = vb_cuda_api();
    if (!cu) {
        set_err(c, "CUDA not loaded");
        return -1;
    }
    cu_impl *m = calloc(1, sizeof *m);
    if (!m) {
        set_err(c, "out of memory");
        return -1;
    }
    c->impl = m;
    m->device = c->dev.handle;

    CU_CHECK(c, cu->DevicePrimaryCtxRetain(&m->ctx, m->device),
             "cuDevicePrimaryCtxRetain");
    m->ctx_retained = 1;
    if (enter(c) != 0)
        return -1;

    int cubin = o->compile_mode == VB_COMPILE_CUBIN && !o->import_ptx;
    const char *sm = c->dev.arch[0] ? c->dev.arch + 3 : "";   /* "120" */
    char opts_buf[512] = "";
    size_t bin_len = 0;
    char *ptx = NULL, *bin = NULL, *log = NULL;

    snprintf(c->compile_mode, sizeof c->compile_mode, "%s",
             cubin ? "cubin" : "ptx-jit");

    if (o->import_ptx) {
        /* Someone else's PTX -- in practice NVIDIA's OpenCL compiler's, from
           --dump-device-code -- finished by the driver as ptx-jit is. */
        char err[256];
        char *raw = read_file(o->import_ptx, err, sizeof err);
        if (!raw) {
            set_err(c, "--import-ptx: %s", err);
            return -1;
        }
        ptx = ocl_ptx_to_cuda(raw, &m->ocl_abi, err, sizeof err);
        free(raw);
        if (!ptx) {
            set_err(c, "--import-ptx %s: %s", o->import_ptx, err);
            return -1;
        }
        const char *base = strrchr(o->import_ptx, '/');
        base = base ? base + 1 : o->import_ptx;
        const char *nvvm = strstr(ptx, "Based on NVVM ");
        snprintf(c->compiler, sizeof c->compiler, "imported");
        /* Whatever steers the importer compiled in, not this run's. */
        snprintf(c->steers, sizeof c->steers, "imported");
        snprintf(c->compiler_version, sizeof c->compiler_version,
                 "%.*s from %.60s",
                 nvvm ? (int) strcspn(nvvm + 9, "\r\n") : 7,
                 nvvm ? nvvm + 9 : "unknown", base);
        snprintf(opts_buf, sizeof opts_buf, "imported from %s", o->import_ptx);
    } else {
        char arch_opt[48];
        snprintf(arch_opt, sizeof arch_opt, "--gpu-architecture=%s_%s",
                 cubin ? "sm" : "compute", sm);
        snprintf(opts_buf, sizeof opts_buf, "%s %s", arch_opt, defines);
        char opts_copy[512];
        memcpy(opts_copy, opts_buf, sizeof opts_copy);
        const char *argv[32];
        int argc = split_options(opts_copy, argv, 32);

        snprintf(c->compiler, sizeof c->compiler, "nvrtc");

        char name[96];
        snprintf(name, sizeof name, "%s.cu", c->label);
        vb_nvrtcProgram prog = NULL;
        if (cu->nvrtcCreateProgram(&prog, source, name, 0, NULL, NULL)
            != VB_NVRTC_SUCCESS) {
            set_err(c, "nvrtcCreateProgram failed");
            return -1;
        }
        vb_nvrtcResult nr = cu->nvrtcCompileProgram(prog, argc, argv);

        size_t log_len = 0;
        cu->nvrtcGetProgramLogSize(prog, &log_len);
        log = calloc(1, log_len + 1);
        if (log && log_len)
            cu->nvrtcGetProgramLog(prog, log);

        if (nr != VB_NVRTC_SUCCESS) {
            set_err(c, "NVRTC %d.%d could not compile for %s (%s): %.300s",
                    cu->nvrtc_major, cu->nvrtc_minor, c->dev.arch,
                    cu->nvrtcGetErrorString(nr), log ? log : "");
            free(log);
            cu->nvrtcDestroyProgram(&prog);
            return -1;
        }

        /* PTX in both modes: it is what ptx-jit loads, and in cubin mode it
           is still the frontend's output worth dumping and naming. */
        size_t ptx_len = 0;
        if (cu->nvrtcGetPTXSize(prog, &ptx_len) == VB_NVRTC_SUCCESS &&
            ptx_len) {
            ptx = malloc(ptx_len);
            if (ptx)
                cu->nvrtcGetPTX(prog, ptx);
        }
        if (cubin && cu->nvrtcGetCUBINSize(prog, &bin_len) == VB_NVRTC_SUCCESS
            && bin_len) {
            bin = malloc(bin_len);
            if (bin)
                cu->nvrtcGetCUBIN(prog, bin);
        }
        cu->nvrtcDestroyProgram(&prog);

        const char *nvvm = ptx ? strstr(ptx, "Based on NVVM ") : NULL;
        if (nvvm)
            snprintf(c->compiler_version, sizeof c->compiler_version,
                     "%d.%d (%.*s)", cu->nvrtc_major, cu->nvrtc_minor,
                     (int) strcspn(nvvm + 9, "\r\n"), nvvm + 9);
        else
            snprintf(c->compiler_version, sizeof c->compiler_version,
                     "%d.%d", cu->nvrtc_major, cu->nvrtc_minor);
    }

    const void *image = cubin ? (const void *) bin : (const void *) ptx;
    if (!image) {
        set_err(c, "NVRTC produced no %s", cubin ? "CUBIN" : "PTX");
        free(ptx);
        free(bin);
        free(log);
        return -1;
    }

    /* The driver's JIT log names the register count and any spill, as
       -cl-nv-verbose does for OpenCL. Only asked for verbosely when kept. */
    char info[4096] = "", errlog[4096] = "";
    int jit_opts[5] = { VB_CU_JIT_INFO_LOG_BUFFER,
                        VB_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                        VB_CU_JIT_ERROR_LOG_BUFFER,
                        VB_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                        VB_CU_JIT_LOG_VERBOSE };
    void *jit_vals[5] = { info, (void *) (uintptr_t) sizeof info, errlog,
                          (void *) (uintptr_t) sizeof errlog,
                          (void *) (uintptr_t) (o->dump_dir ? 1 : 0) };
    vb_CUresult r = cu->ModuleLoadDataEx(&m->module, image, 5, jit_opts,
                                         jit_vals);
    if (r != VB_CUDA_SUCCESS) {
        for (size_t n = strlen(errlog); n > 0 && (errlog[n - 1] == '\n' ||
                                                  errlog[n - 1] == ' '); n--)
            errlog[n - 1] = '\0';
        /* The one mismatch worth naming: NVRTC newer than the driver emits
           PTX the driver's JIT cannot read. */
        set_err(c, "loading the %s module: %s%s%s%s", cubin ? "CUBIN" : "PTX",
                vb_cuda_strerror(r), errlog[0] ? " -- " : "", errlog,
                r == 222 ? " (NVRTC is newer than the driver; try "
                           "--compile-mode cubin or an older NVRTC)" : "");
        free(ptx);
        free(bin);
        free(log);
        return -1;
    }
    r = cu->ModuleGetFunction(&m->fn, m->module, entry);
    if (r != VB_CUDA_SUCCESS) {
        set_err(c, "cuModuleGetFunction(%s): %s", entry, vb_cuda_strerror(r));
        free(ptx);
        free(bin);
        free(log);
        return -1;
    }
    if (m->ocl_abi) {
        char version[32] = ".version 8.7";
        const char *v = strstr(ptx, ".version ");
        if (v)
            snprintf(version, sizeof version, "%.*s",
                     (int) strcspn(v, "\r\n"), v);
        if (shared_base(c, version, &m->shared_base) != 0) {
            free(ptx);
            free(bin);
            free(log);
            return -1;
        }
    }

    if (o->dump_dir) {
        char fname[256];
        const char *mode = o->import_ptx ? "import" : c->compile_mode;
        snprintf(fname, sizeof fname, "%s.cuda-%s.d%d.cu", c->label, mode,
                 c->dev.ordinal);
        if (!o->import_ptx)
            dump_file(o->dump_dir, fname, source, strlen(source));
        if (ptx) {
            snprintf(fname, sizeof fname, "%s.cuda-%s.d%d.ptx", c->label,
                     mode, c->dev.ordinal);
            dump_file(o->dump_dir, fname, ptx, strlen(ptx));
        }
        if (bin) {
            snprintf(fname, sizeof fname, "%s.cuda-%s.d%d.cubin", c->label,
                     mode, c->dev.ordinal);
            dump_file(o->dump_dir, fname, bin, bin_len);
        }
        int regs = 0, local = 0;
        cu->FuncGetAttribute(&regs, VB_CU_FUNC_ATTRIBUTE_NUM_REGS, m->fn);
        cu->FuncGetAttribute(&local, VB_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                             m->fn);
        size_t cap = strlen(opts_buf) + (log ? strlen(log) : 0) +
                     strlen(info) + 512;
        char *text = malloc(cap);
        if (text) {
            int k = snprintf(text, cap,
                             "options: %s\ncompiler: %s %s\n"
                             "registers: %d, local memory: %d bytes\n\n"
                             "nvrtc log:\n%s\n\ndriver JIT log:\n%s\n",
                             opts_buf, c->compiler, c->compiler_version,
                             regs, local,
                             log ? log : "", info);
            if (k > 0) {
                snprintf(fname, sizeof fname, "%s.cuda-%s.d%d.log", c->label,
                         mode, c->dev.ordinal);
                dump_file(o->dump_dir, fname, text,
                          (size_t) k < cap ? (size_t) k : cap - 1);
            }
            free(text);
        }
    }
    free(ptx);
    free(bin);
    free(log);

    CU_CHECK(c, cu->StreamCreate(&m->stream, 0), "cuStreamCreate");
    CU_CHECK(c, cu->EventCreate(&m->k0, 0), "cuEventCreate");
    CU_CHECK(c, cu->EventCreate(&m->k1, 0), "cuEventCreate");
    CU_CHECK(c, cu->EventCreate(&m->x0, 0), "cuEventCreate");
    CU_CHECK(c, cu->EventCreate(&m->x1, 0), "cuEventCreate");
    return 0;
}

/* What this kernel may launch with: its register use caps the block size,
   which CUDA reports per function. */
static size_t cu_max_local(const vb_dev_ctx *c)
{
    const vb_cuda *cu = vb_cuda_api();
    const cu_impl *m = c->impl;
    int v = 0;
    if (cu->FuncGetAttribute(&v, VB_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             m->fn) != VB_CUDA_SUCCESS || v <= 0)
        return 0;
    return (size_t) v;
}

/* ---- buffers ------------------------------------------------------------ */

static int cu_alloc(vb_dev_ctx *c, const void *slice, size_t corpus_bytes,
                    size_t partial_bytes)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    if (enter(c) != 0)
        return -1;
    vb_CUresult r = cu->MemAlloc(&m->d_corpus, corpus_bytes);
    if (r != VB_CUDA_SUCCESS) {
        set_err(c, "corpus allocation (%.1f MiB): %s",
                (double) corpus_bytes / 1048576.0, vb_cuda_strerror(r));
        return -1;
    }
    CU_CHECK(c, cu->MemcpyHtoD(m->d_corpus, slice, corpus_bytes),
             "corpus upload");
    CU_CHECK(c, cu->MemAlloc(&m->d_partial, partial_bytes), "partial buffer");
    return 0;
}

/* Pinned staging: page-locked memory the copy engine reads directly. On
   failure streaming reads pageable memory and host_pinned says so. */
static void cu_pin_staging(vb_dev_ctx *c)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    if (enter(c) != 0)
        return;
    if (cu->MemHostAlloc(&m->staging, c->corpus_bytes, 0) != VB_CUDA_SUCCESS) {
        m->staging = NULL;
        return;
    }
    memcpy(m->staging, c->host_slice, c->corpus_bytes);
    c->host_pinned = 1;
}

/* ---- launch and read ---------------------------------------------------- */

static int cu_launch(vb_dev_ctx *c, uint32_t iterations, size_t shared)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    if (enter(c) != 0)
        return -1;

    /* Streaming: the upload goes on the same stream ahead of the kernel,
       asynchronously, and is timed by its own pair of events -- the shape of
       OpenCL's non-blocking write on an in-order queue. */
    m->streamed = 0;
    if (c->stream) {
        const void *src = m->staging ? m->staging : c->host_slice;
        CU_CHECK(c, cu->EventRecord(m->x0, m->stream), "cuEventRecord");
        vb_CUresult r = cu->MemcpyHtoDAsync(m->d_corpus, src, c->corpus_bytes,
                                            m->stream);
        if (r != VB_CUDA_SUCCESS) {
            set_err(c, "corpus upload (%.1f MiB): %s",
                    (double) c->corpus_bytes / 1048576.0, vb_cuda_strerror(r));
            return -1;
        }
        CU_CHECK(c, cu->EventRecord(m->x1, m->stream), "cuEventRecord");
        m->streamed = 1;
    }

    /* The parameter list the core's signature declares; the work-group
       scratch is dynamic shared memory rather than an argument. */
    unsigned long long groups = c->n_groups;
    void *args[7] = { &m->d_corpus, &c->blocks, &iterations, &groups,
                      &c->repeats, &m->d_partial, &m->shared_base };

    unsigned grid = (unsigned) (c->global_size / c->local_size);
    CU_CHECK(c, cu->EventRecord(m->k0, m->stream), "cuEventRecord");
    vb_CUresult r = cu->LaunchKernel(m->fn, grid, 1, 1,
                                     (unsigned) c->local_size, 1, 1,
                                     (unsigned) shared, m->stream, args, NULL);
    if (r != VB_CUDA_SUCCESS) {
        set_err(c, "cuLaunchKernel (global=%zu local=%zu): %s",
                c->global_size, c->local_size, vb_cuda_strerror(r));
        return -1;
    }
    CU_CHECK(c, cu->EventRecord(m->k1, m->stream), "cuEventRecord");
    return 0;
}

static uint64_t elapsed_ns(const vb_cuda *cu, vb_CUevent a, vb_CUevent b)
{
    float ms = 0.0f;
    if (cu->EventElapsedTime(&ms, a, b) != VB_CUDA_SUCCESS || ms <= 0.0f)
        return 0;
    return (uint64_t) ((double) ms * 1e6);
}

static int cu_read(vb_dev_ctx *c, size_t bytes)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    if (enter(c) != 0)
        return -1;
    CU_CHECK(c, cu->MemcpyDtoHAsync(c->partials, m->d_partial, bytes,
                                    m->stream), "reading partials");
    CU_CHECK(c, cu->StreamSynchronize(m->stream), "cuStreamSynchronize");
    c->last_kernel_ns = elapsed_ns(cu, m->k0, m->k1);
    c->last_transfer_ns = m->streamed ? elapsed_ns(cu, m->x0, m->x1) : 0;
    return 0;
}

/* ---- pipelined streaming ------------------------------------------------ */

static int cu_pipe_open(vb_dev_ctx *c, size_t chunk_bytes, size_t read_bytes)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    if (enter(c) != 0)
        return -1;
    CU_CHECK(c, cu->StreamCreate(&m->cstream, 0), "cuStreamCreate");
    for (int i = 0; i < VB_PIPE_RING; i++) {
        CU_CHECK(c, cu->EventCreate(&m->pu0[i], 0), "cuEventCreate");
        CU_CHECK(c, cu->EventCreate(&m->pu1[i], 0), "cuEventCreate");
        CU_CHECK(c, cu->EventCreate(&m->pk0[i], 0), "cuEventCreate");
        CU_CHECK(c, cu->EventCreate(&m->pk1[i], 0), "cuEventCreate");
        CU_CHECK(c, cu->EventCreate(&m->prd[i], 0), "cuEventCreate");
    }
    for (int b = 0; b < 2; b++) {
        vb_CUresult r = cu->MemAlloc(&m->pbuf[b], chunk_bytes);
        if (r != VB_CUDA_SUCCESS) {
            set_err(c, "chunk buffer (%.1f MiB): %s",
                    (double) chunk_bytes / 1048576.0, vb_cuda_strerror(r));
            return -1;
        }
        CU_CHECK(c, cu->MemAlloc(&m->ppart[b], read_bytes),
                 "chunk partial buffer");
    }
    /* Pinned, because an asynchronous copy to pageable memory is not
       asynchronous: the host would wait out every kernel before it could
       queue the next chunk. */
    m->ring_slot = read_bytes;
    if (cu->MemHostAlloc((void **) &m->ring, VB_PIPE_RING * read_bytes, 0)
        == VB_CUDA_SUCCESS) {
        m->ring_pinned = 1;
    } else {
        m->ring = malloc(VB_PIPE_RING * read_bytes);
        if (!m->ring) {
            set_err(c, "out of memory for the readback ring");
            return -1;
        }
    }
    return 0;
}

static int cu_pipe_enqueue(vb_dev_ctx *c, uint64_t seq, size_t offset,
                           size_t bytes, uint64_t n_groups,
                           uint32_t iterations, size_t shared,
                           size_t read_bytes)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    unsigned i = (unsigned) (seq % VB_PIPE_RING);
    int b = (int) (seq % 2);
    if (enter(c) != 0)
        return -1;

    /* Upload, once chunk seq - 2's kernel has finished reading the buffer. */
    const unsigned char *src = (const unsigned char *)
        (m->staging ? m->staging : c->host_slice) + offset;
    if (seq >= 2)
        CU_CHECK(c, cu->StreamWaitEvent(m->cstream,
                                        m->pk1[(seq - 2) % VB_PIPE_RING], 0),
                 "cuStreamWaitEvent");
    CU_CHECK(c, cu->EventRecord(m->pu0[i], m->cstream), "cuEventRecord");
    vb_CUresult r = cu->MemcpyHtoDAsync(m->pbuf[b], src, bytes, m->cstream);
    if (r != VB_CUDA_SUCCESS) {
        set_err(c, "chunk upload: %s", vb_cuda_strerror(r));
        return -1;
    }
    CU_CHECK(c, cu->EventRecord(m->pu1[i], m->cstream), "cuEventRecord");

    /* Kernel, once the upload has landed; then the partials back. */
    CU_CHECK(c, cu->StreamWaitEvent(m->stream, m->pu1[i], 0),
             "cuStreamWaitEvent");
    unsigned long long groups = n_groups;
    void *args[7] = { &m->pbuf[b], &c->blocks, &iterations, &groups,
                      &c->repeats, &m->ppart[b], &m->shared_base };
    unsigned grid = (unsigned) (c->global_size / c->local_size);
    CU_CHECK(c, cu->EventRecord(m->pk0[i], m->stream), "cuEventRecord");
    r = cu->LaunchKernel(m->fn, grid, 1, 1, (unsigned) c->local_size, 1, 1,
                         (unsigned) shared, m->stream, args, NULL);
    if (r != VB_CUDA_SUCCESS) {
        set_err(c, "cuLaunchKernel (chunk): %s", vb_cuda_strerror(r));
        return -1;
    }
    CU_CHECK(c, cu->EventRecord(m->pk1[i], m->stream), "cuEventRecord");
    CU_CHECK(c, cu->MemcpyDtoHAsync(m->ring + (size_t) i * m->ring_slot,
                                    m->ppart[b], read_bytes, m->stream),
             "reading chunk partials");
    CU_CHECK(c, cu->EventRecord(m->prd[i], m->stream), "cuEventRecord");
    return 0;
}

static int cu_pipe_wait(vb_dev_ctx *c, uint64_t seq, const void **partials,
                        uint64_t *kernel_ns, uint64_t *transfer_ns)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    unsigned i = (unsigned) (seq % VB_PIPE_RING);
    if (enter(c) != 0)
        return -1;
    CU_CHECK(c, cu->EventSynchronize(m->prd[i]), "waiting for a chunk");
    *kernel_ns = elapsed_ns(cu, m->pk0[i], m->pk1[i]);
    *transfer_ns = elapsed_ns(cu, m->pu0[i], m->pu1[i]);
    *partials = m->ring + (size_t) i * m->ring_slot;
    return 0;
}

static void cu_destroy(vb_dev_ctx *c)
{
    const vb_cuda *cu = vb_cuda_api();
    cu_impl *m = c->impl;
    if (!m)
        return;
    if (cu && m->ctx && cu->CtxSetCurrent(m->ctx) == VB_CUDA_SUCCESS) {
        if (m->stream)    cu->StreamSynchronize(m->stream);
        if (m->cstream)   cu->StreamSynchronize(m->cstream);
        for (int i = 0; i < VB_PIPE_RING; i++) {
            if (m->pu0[i]) cu->EventDestroy(m->pu0[i]);
            if (m->pu1[i]) cu->EventDestroy(m->pu1[i]);
            if (m->pk0[i]) cu->EventDestroy(m->pk0[i]);
            if (m->pk1[i]) cu->EventDestroy(m->pk1[i]);
            if (m->prd[i]) cu->EventDestroy(m->prd[i]);
        }
        for (int b = 0; b < 2; b++) {
            if (m->pbuf[b])  cu->MemFree(m->pbuf[b]);
            if (m->ppart[b]) cu->MemFree(m->ppart[b]);
        }
        if (m->ring_pinned)  cu->MemFreeHost(m->ring);
        else                 free(m->ring);
        if (m->cstream)   cu->StreamDestroy(m->cstream);
        if (m->k0)        cu->EventDestroy(m->k0);
        if (m->k1)        cu->EventDestroy(m->k1);
        if (m->x0)        cu->EventDestroy(m->x0);
        if (m->x1)        cu->EventDestroy(m->x1);
        if (m->stream)    cu->StreamDestroy(m->stream);
        if (m->staging)   cu->MemFreeHost(m->staging);
        if (m->d_corpus)  cu->MemFree(m->d_corpus);
        if (m->d_partial) cu->MemFree(m->d_partial);
        if (m->module)    cu->ModuleUnload(m->module);
    }
    if (cu && m->ctx_retained)
        cu->DevicePrimaryCtxRelease(m->device);
    free(m);
    c->impl = NULL;
}

const vb_dev_backend vb_cuda_backend = {
    .id          = VB_BACKEND_CUDA,
    .name        = "cuda",
    .dialect     = VB_DIALECT_CUDA,
    .enumerate   = cu_enumerate,
    .unavailable = cu_unavailable,
    .build       = cu_build,
    .max_local   = cu_max_local,
    .alloc       = cu_alloc,
    .pin_staging = cu_pin_staging,
    .launch      = cu_launch,
    .read        = cu_read,
    .pipe_open   = cu_pipe_open,
    .pipe_enqueue = cu_pipe_enqueue,
    .pipe_wait   = cu_pipe_wait,
    .destroy     = cu_destroy,
};
