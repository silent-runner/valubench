/*
 * loader.c -- dlopen libcuda and libnvrtc and resolve entry points.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * The same technique as src/opencl/loader.c: no toolkit to build, and a
 * machine without the driver or without NVRTC runs the same binary with CUDA
 * reported unavailable.
 */

#define _GNU_SOURCE

#include "cuda_loader.h"

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vb_cuda g_cu;
static int     g_state;             /* 0 untried, 1 loaded, -1 unavailable */
static char    g_error[320];

const char *vb_cuda_error(void)
{
    return g_error[0] ? g_error : NULL;
}

const vb_cuda *vb_cuda_api(void)
{
    return g_state == 1 ? &g_cu : NULL;
}

static int fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof g_error, fmt, ap);
    va_end(ap);
    if (g_cu.nvrtc_lib)
        dlclose(g_cu.nvrtc_lib);
    if (g_cu.lib)
        dlclose(g_cu.lib);
    memset(&g_cu, 0, sizeof g_cu);
    g_state = -1;
    return 0;
}

/* The first name that resolves. Versioned names first: libcuda keeps the
   unversioned symbol for binary compatibility with the oldest behaviour, so
   cuMemAlloc and cuMemAlloc_v2 both exist and only the latter takes a 64-bit
   size_t. */
static void *resolve(void *lib, const char *const *names)
{
    for (; *names; names++) {
        void *p = dlsym(lib, *names);
        if (p)
            return p;
    }
    return NULL;
}

#define NAMES(...) ((const char *const[]) { __VA_ARGS__, NULL })

#define LOAD_CU(field, ...)                                                  \
    do {                                                                     \
        *(void **) (&g_cu.field) = resolve(g_cu.lib, NAMES(__VA_ARGS__));    \
        if (!g_cu.field)                                                     \
            return fail("libcuda is missing %s", NAMES(__VA_ARGS__)[0]);     \
    } while (0)

#define LOAD_NVRTC(field, name)                                              \
    do {                                                                     \
        *(void **) (&g_cu.field) = dlsym(g_cu.nvrtc_lib, name);              \
        if (!g_cu.field)                                                     \
            return fail("libnvrtc is missing %s", name);                     \
    } while (0)

/*
 * NVRTC's soname carries its major version, and a toolkit install is not on
 * the loader's path by default. So: the majors this could plausibly meet,
 * newest first, then the unversioned name, through the normal search
 * (LD_LIBRARY_PATH, the cache), then the same in the conventional toolkit
 * directories. The pip wheel (nvidia-cuda-nvrtc) ships only the versioned
 * name, in nvidia/cu13/lib, and is reached by putting that directory on
 * LD_LIBRARY_PATH -- which is why versioned names come first: an unversioned
 * toolkit symlink found through the cache would otherwise win over it.
 */
static void *open_nvrtc(void)
{
    static const char *const names[] = {
        "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so",
    };
    const char *dirs[4] = { getenv("CUDA_HOME"), getenv("CUDA_PATH"),
                            "/usr/local/cuda", "/opt/cuda" };

    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        void *h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (h)
            return h;
    }
    for (size_t d = 0; d < 4; d++) {
        if (!dirs[d] || !dirs[d][0])
            continue;
        for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
            char path[512];
            snprintf(path, sizeof path, "%s/lib64/%s", dirs[d], names[i]);
            void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (h)
                return h;
        }
    }
    return NULL;
}

/* The kernel driver's release, which is what a person recognises ("610.57.04")
   and what OpenCL reports; CUDA's own version number is the API level. */
static void read_kernel_driver(char *out, size_t n)
{
    out[0] = '\0';
    FILE *f = fopen("/proc/driver/nvidia/version", "r");
    if (!f)
        return;
    char line[512];
    if (fgets(line, sizeof line, f)) {
        /* "NVRM version: NVIDIA UNIX ... Kernel Module for x86_64  610.57.04
           Release Build ..." -- the first token that looks like a release. */
        char *tok = strtok(line, " \t\n");
        while (tok) {
            if (tok[0] >= '0' && tok[0] <= '9' && strchr(tok, '.')) {
                snprintf(out, n, "%s", tok);
                break;
            }
            tok = strtok(NULL, " \t\n");
        }
    }
    fclose(f);
}

int vb_cuda_load(void)
{
    if (g_state != 0)
        return g_state == 1;

    g_cu.lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_cu.lib)
        g_cu.lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_cu.lib)
        return fail("no CUDA driver (libcuda.so.1 not found)");

    LOAD_CU(Init,                   "cuInit");
    LOAD_CU(DriverGetVersion,       "cuDriverGetVersion");
    LOAD_CU(DeviceGetCount,         "cuDeviceGetCount");
    LOAD_CU(DeviceGet,              "cuDeviceGet");
    LOAD_CU(DeviceGetName,          "cuDeviceGetName");
    LOAD_CU(DeviceGetAttribute,     "cuDeviceGetAttribute");
    LOAD_CU(DeviceTotalMem,         "cuDeviceTotalMem_v2");
    LOAD_CU(DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    LOAD_CU(DevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2",
                                     "cuDevicePrimaryCtxRelease");
    LOAD_CU(CtxSetCurrent,          "cuCtxSetCurrent");
    LOAD_CU(ModuleLoadDataEx,       "cuModuleLoadDataEx");
    LOAD_CU(ModuleUnload,           "cuModuleUnload");
    LOAD_CU(ModuleGetFunction,      "cuModuleGetFunction");
    LOAD_CU(FuncGetAttribute,       "cuFuncGetAttribute");
    LOAD_CU(MemAlloc,               "cuMemAlloc_v2");
    LOAD_CU(MemFree,                "cuMemFree_v2");
    LOAD_CU(MemHostAlloc,           "cuMemHostAlloc");
    LOAD_CU(MemFreeHost,            "cuMemFreeHost");
    LOAD_CU(MemcpyHtoD,             "cuMemcpyHtoD_v2");
    LOAD_CU(MemcpyHtoDAsync,        "cuMemcpyHtoDAsync_v2");
    LOAD_CU(MemcpyDtoHAsync,        "cuMemcpyDtoHAsync_v2");
    LOAD_CU(StreamCreate,           "cuStreamCreate");
    LOAD_CU(StreamDestroy,          "cuStreamDestroy_v2", "cuStreamDestroy");
    LOAD_CU(StreamSynchronize,      "cuStreamSynchronize");
    LOAD_CU(EventCreate,            "cuEventCreate");
    LOAD_CU(EventDestroy,           "cuEventDestroy_v2", "cuEventDestroy");
    LOAD_CU(EventRecord,            "cuEventRecord");
    LOAD_CU(EventElapsedTime,       "cuEventElapsedTime_v2",
                                    "cuEventElapsedTime");
    LOAD_CU(LaunchKernel,           "cuLaunchKernel");
    *(void **) (&g_cu.GetErrorName) = dlsym(g_cu.lib, "cuGetErrorName");

    vb_CUresult r = g_cu.Init(0);
    if (r != VB_CUDA_SUCCESS)
        return fail("cuInit failed (%s) -- the driver is installed but the "
                    "device is not usable", vb_cuda_strerror(r));
    g_cu.DriverGetVersion(&g_cu.driver_version);

    g_cu.nvrtc_lib = open_nvrtc();
    if (!g_cu.nvrtc_lib)
        return fail("the CUDA driver is present but NVRTC is not: CUDA "
                    "kernels are compiled at run time and need libnvrtc, from "
                    "a CUDA toolkit or the nvidia-cuda-nvrtc package");

    LOAD_NVRTC(nvrtcVersion,           "nvrtcVersion");
    LOAD_NVRTC(nvrtcCreateProgram,     "nvrtcCreateProgram");
    LOAD_NVRTC(nvrtcDestroyProgram,    "nvrtcDestroyProgram");
    LOAD_NVRTC(nvrtcCompileProgram,    "nvrtcCompileProgram");
    LOAD_NVRTC(nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
    LOAD_NVRTC(nvrtcGetProgramLog,     "nvrtcGetProgramLog");
    LOAD_NVRTC(nvrtcGetPTXSize,        "nvrtcGetPTXSize");
    LOAD_NVRTC(nvrtcGetPTX,            "nvrtcGetPTX");
    LOAD_NVRTC(nvrtcGetCUBINSize,      "nvrtcGetCUBINSize");
    LOAD_NVRTC(nvrtcGetCUBIN,          "nvrtcGetCUBIN");
    LOAD_NVRTC(nvrtcGetErrorString,    "nvrtcGetErrorString");
    g_cu.nvrtcVersion(&g_cu.nvrtc_major, &g_cu.nvrtc_minor);

    read_kernel_driver(g_cu.kernel_driver, sizeof g_cu.kernel_driver);

    g_state = 1;
    g_error[0] = '\0';
    return 1;
}

const char *vb_cuda_strerror(vb_CUresult r)
{
    const char *name = NULL;
    if (g_cu.GetErrorName && g_cu.GetErrorName(r, &name) == VB_CUDA_SUCCESS &&
        name)
        return name;
    static char buf[32];
    snprintf(buf, sizeof buf, "CUDA error %d", r);
    return buf;
}
