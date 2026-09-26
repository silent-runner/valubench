/*
 * loader.c -- dlopen libOpenCL and resolve entry points at runtime.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * The whole point is that no OpenCL SDK is needed to build valubench, and a
 * machine with no GPU runs the same binary without complaint. Absence of OpenCL
 * is a normal outcome reported through vb_ocl_error(), never a fatal error.
 */

#define _GNU_SOURCE

#include "opencl.h"

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static vb_ocl g_ocl;
static int    g_state;          /* 0 = untried, 1 = loaded, -1 = unavailable */
static char   g_error[256];

const char *vb_ocl_error(void)
{
    return g_error[0] ? g_error : NULL;
}

static int fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof g_error, fmt, ap);
    va_end(ap);

    if (g_ocl.lib) {
        dlclose(g_ocl.lib);
        memset(&g_ocl, 0, sizeof g_ocl);
    }
    g_state = -1;
    return 0;
}

/*
 * SONAME first, then the development symlink. Preferring libOpenCL.so.1 means
 * we work on a machine that has a runtime but no -dev package, which is the
 * normal case for a bare deployment box.
 */
static const char *const SONAMES[] = {
    "libOpenCL.so.1",
    "libOpenCL.so",
};

#define LOAD(field, symbol)                                                   \
    do {                                                                      \
        *(void **) (&g_ocl.field) = dlsym(g_ocl.lib, symbol);                 \
        if (!g_ocl.field)                                                     \
            return fail("libOpenCL is missing %s -- the ICD loader looks "    \
                        "incomplete", symbol);                                \
    } while (0)

int vb_ocl_load(void)
{
    if (g_state != 0)
        return g_state == 1;

    for (size_t i = 0; i < sizeof SONAMES / sizeof SONAMES[0]; i++) {
        g_ocl.lib = dlopen(SONAMES[i], RTLD_NOW | RTLD_LOCAL);
        if (g_ocl.lib)
            break;
    }

    if (!g_ocl.lib)
        return fail("no OpenCL loader found (tried libOpenCL.so.1 and "
                    "libOpenCL.so)");

    LOAD(GetPlatformIDs,          "clGetPlatformIDs");
    LOAD(GetPlatformInfo,         "clGetPlatformInfo");
    LOAD(GetDeviceIDs,            "clGetDeviceIDs");
    LOAD(GetDeviceInfo,           "clGetDeviceInfo");
    LOAD(CreateContext,           "clCreateContext");
    LOAD(ReleaseContext,          "clReleaseContext");
    LOAD(CreateCommandQueue,      "clCreateCommandQueue");
    LOAD(ReleaseCommandQueue,     "clReleaseCommandQueue");
    LOAD(Finish,                  "clFinish");
    LOAD(Flush,                   "clFlush");
    LOAD(CreateBuffer,            "clCreateBuffer");
    LOAD(ReleaseMemObject,        "clReleaseMemObject");
    LOAD(EnqueueWriteBuffer,      "clEnqueueWriteBuffer");
    LOAD(EnqueueReadBuffer,       "clEnqueueReadBuffer");
    LOAD(EnqueueMapBuffer,        "clEnqueueMapBuffer");
    LOAD(EnqueueUnmapMemObject,   "clEnqueueUnmapMemObject");
    LOAD(CreateProgramWithSource, "clCreateProgramWithSource");
    LOAD(BuildProgram,            "clBuildProgram");
    LOAD(GetProgramBuildInfo,     "clGetProgramBuildInfo");
    LOAD(ReleaseProgram,          "clReleaseProgram");
    LOAD(GetProgramInfo,          "clGetProgramInfo");
    LOAD(CreateKernel,            "clCreateKernel");
    LOAD(SetKernelArg,            "clSetKernelArg");
    LOAD(GetKernelWorkGroupInfo,  "clGetKernelWorkGroupInfo");
    LOAD(ReleaseKernel,           "clReleaseKernel");
    LOAD(EnqueueNDRangeKernel,    "clEnqueueNDRangeKernel");
    LOAD(GetEventProfilingInfo,   "clGetEventProfilingInfo");
    LOAD(ReleaseEvent,            "clReleaseEvent");

    g_state = 1;
    g_error[0] = '\0';
    return 1;
}

const vb_ocl *vb_ocl_api(void)
{
    return (g_state == 1) ? &g_ocl : NULL;
}

const char *vb_ocl_strerror(cl_int err)
{
    switch (err) {
    case 0:   return "CL_SUCCESS";
    case -1:  return "CL_DEVICE_NOT_FOUND";
    case -2:  return "CL_DEVICE_NOT_AVAILABLE";
    case -3:  return "CL_COMPILER_NOT_AVAILABLE";
    case -4:  return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:  return "CL_OUT_OF_RESOURCES";
    case -6:  return "CL_OUT_OF_HOST_MEMORY";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    /* Returned by the ICD loader itself, not by any driver, when no vendor
       ICD could be loaded -- the usual symptom of a driver that is installed
       but unreachable. */
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    default:  return "unknown OpenCL error";
    }
}

const char *vb_ocl_type_name(cl_device_type t)
{
    if (t & CL_DEVICE_TYPE_GPU)         return "GPU";
    if (t & CL_DEVICE_TYPE_CPU)         return "CPU";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "accelerator";
    return "other";
}

static void get_str(const vb_ocl *cl, cl_device_id d, cl_device_info what,
                    char *out, size_t n)
{
    size_t got = 0;
    out[0] = '\0';
    if (cl->GetDeviceInfo(d, what, n - 1, out, &got) != CL_SUCCESS) {
        snprintf(out, n, "unknown");
        return;
    }
    out[got < n ? got : n - 1] = '\0';
}

static void get_plat_str(const vb_ocl *cl, cl_platform_id p,
                         cl_platform_info what, char *out, size_t n)
{
    size_t got = 0;
    out[0] = '\0';
    if (cl->GetPlatformInfo(p, what, n - 1, out, &got) != CL_SUCCESS) {
        snprintf(out, n, "unknown");
        return;
    }
    out[got < n ? got : n - 1] = '\0';
}

int vb_ocl_devices(vb_ocl_device *out, int max)
{
    if (!vb_ocl_load())
        return 0;

    const vb_ocl *cl = vb_ocl_api();
    /*
     * GetPlatformIDs reports the number of platforms that *exist*, not the
     * number it wrote. Looping to the reported count read past the array on a
     * host with more than sixteen platforms; the count is now clamped to what
     * was actually filled in.
     */
    enum { MAX_PLATFORMS = 16 };
    cl_platform_id platforms[MAX_PLATFORMS];
    cl_uint n_platforms = 0;
    int n = 0;

    if (max < 0)
        max = 0;
    if (max > VB_OCL_MAX_DEVICES)
        max = VB_OCL_MAX_DEVICES;

    cl_int err = cl->GetPlatformIDs(MAX_PLATFORMS, platforms, &n_platforms);
    if (n_platforms > MAX_PLATFORMS)
        n_platforms = MAX_PLATFORMS;

    if (err != CL_SUCCESS || n_platforms == 0) {
        snprintf(g_error, sizeof g_error,
                 "OpenCL loaded but reports no platforms (%s). A driver may be "
                 "installed without permission to reach the device -- check "
                 "access to /dev/dri/render*.", vb_ocl_strerror(err));
        return 0;
    }

    for (cl_uint p = 0; p < n_platforms && n < max; p++) {
        cl_device_id devices[VB_OCL_MAX_DEVICES];
        cl_uint n_devices = 0;

        if (cl->GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                             VB_OCL_MAX_DEVICES, devices,
                             &n_devices) != CL_SUCCESS)
            continue;                       /* platform with no usable devices */

        for (cl_uint d = 0; d < n_devices && n < max; d++) {
            vb_ocl_device *o = &out[n];

            memset(o, 0, sizeof *o);
            o->platform = platforms[p];
            o->device = devices[d];

            get_plat_str(cl, platforms[p], CL_PLATFORM_NAME,
                         o->platform_name, sizeof o->platform_name);
            get_plat_str(cl, platforms[p], CL_PLATFORM_VERSION,
                         o->platform_version, sizeof o->platform_version);

            get_str(cl, devices[d], CL_DEVICE_NAME, o->name, sizeof o->name);
            get_str(cl, devices[d], CL_DEVICE_VENDOR, o->vendor,
                    sizeof o->vendor);
            get_str(cl, devices[d], CL_DRIVER_VERSION, o->driver_version,
                    sizeof o->driver_version);
            get_str(cl, devices[d], CL_DEVICE_VERSION, o->device_version,
                    sizeof o->device_version);

            cl->GetDeviceInfo(devices[d], CL_DEVICE_TYPE,
                              sizeof o->type, &o->type, NULL);
            cl->GetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof o->compute_units, &o->compute_units, NULL);
    /*
     * A device that reports zero compute units hangs the geometry tuner: the
     * candidate global size is local * mult * compute_units, which stays 0
     * however far mult grows, so it never reaches the cap that ends the loop.
     * The query is not required to succeed and some ICDs leave the value
     * untouched, so a floor of one is cheaper than trusting it.
     */
    if (o->compute_units == 0)
        o->compute_units = 1;
            cl->GetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                              sizeof o->clock_mhz, &o->clock_mhz, NULL);
            cl->GetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                              sizeof o->max_work_group, &o->max_work_group,
                              NULL);
            cl->GetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE,
                              sizeof o->global_mem, &o->global_mem, NULL);
            cl->GetDeviceInfo(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                              sizeof o->max_alloc, &o->max_alloc, NULL);
            cl->GetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE,
                              sizeof o->local_mem, &o->local_mem, NULL);
            n++;
        }
    }

    if (n == 0)
        snprintf(g_error, sizeof g_error,
                 "OpenCL reports %u platform(s) but no usable devices",
                 n_platforms);

    return n;
}
