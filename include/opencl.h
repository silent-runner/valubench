/*
 * opencl.h -- runtime-loaded OpenCL backend.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Nothing here is linked at build time. libOpenCL is dlopen'd on first use and
 * every entry point resolved into a table, so valubench builds with no SDK and
 * runs unchanged on a machine with no GPU, no driver, or no OpenCL at all --
 * the technique described in docs/research.md 2.7.
 */

#ifndef VALUBENCH_OPENCL_H
#define VALUBENCH_OPENCL_H

#include "vb_cl.h"

#include <stddef.h>

/* Resolved entry points. Valid only after vb_ocl_load() returns 1. */
typedef struct {
    void *lib;

    cl_fn_GetPlatformIDs          GetPlatformIDs;
    cl_fn_GetPlatformInfo         GetPlatformInfo;
    cl_fn_GetDeviceIDs            GetDeviceIDs;
    cl_fn_GetDeviceInfo           GetDeviceInfo;
    cl_fn_CreateContext           CreateContext;
    cl_fn_ReleaseContext          ReleaseContext;
    cl_fn_CreateCommandQueue      CreateCommandQueue;
    cl_fn_ReleaseCommandQueue     ReleaseCommandQueue;
    cl_fn_Finish                  Finish;
    cl_fn_Flush                   Flush;
    cl_fn_CreateBuffer            CreateBuffer;
    cl_fn_ReleaseMemObject        ReleaseMemObject;
    cl_fn_EnqueueWriteBuffer      EnqueueWriteBuffer;
    cl_fn_EnqueueReadBuffer       EnqueueReadBuffer;
    cl_fn_EnqueueMapBuffer        EnqueueMapBuffer;
    cl_fn_EnqueueUnmapMemObject   EnqueueUnmapMemObject;
    cl_fn_CreateProgramWithSource CreateProgramWithSource;
    cl_fn_BuildProgram            BuildProgram;
    cl_fn_GetProgramBuildInfo     GetProgramBuildInfo;
    cl_fn_ReleaseProgram          ReleaseProgram;
    cl_fn_GetProgramInfo          GetProgramInfo;
    cl_fn_CreateKernel            CreateKernel;
    cl_fn_SetKernelArg            SetKernelArg;
    cl_fn_GetKernelWorkGroupInfo  GetKernelWorkGroupInfo;
    cl_fn_ReleaseKernel           ReleaseKernel;
    cl_fn_EnqueueNDRangeKernel    EnqueueNDRangeKernel;
    cl_fn_GetEventProfilingInfo   GetEventProfilingInfo;
    cl_fn_ReleaseEvent            ReleaseEvent;
} vb_ocl;

/*
 * Load libOpenCL and resolve every required symbol. Returns 1 on success, 0 if
 * OpenCL is unavailable -- which is a normal outcome, not an error. Idempotent;
 * the library is loaded at most once per process.
 *
 * On failure vb_ocl_error() explains why, for --verbose reporting.
 */
int vb_ocl_load(void);

/* NULL until vb_ocl_load() has succeeded. */
const vb_ocl *vb_ocl_api(void);

/* Last load or device-enumeration failure, or NULL. Static storage. */
const char *vb_ocl_error(void);

/* Human-readable name for an OpenCL error code. */
const char *vb_ocl_strerror(cl_int err);

#define VB_OCL_MAX_DEVICES 32

typedef struct {
    cl_platform_id platform;
    cl_device_id   device;

    char     platform_name[128];
    char     platform_version[128];
    char     name[128];
    char     vendor[128];
    char     driver_version[64];
    char     device_version[64];

    cl_device_type type;
    cl_uint  compute_units;
    cl_uint  clock_mhz;
    size_t   max_work_group;
    cl_ulong global_mem;
    cl_ulong max_alloc;

    /* Work-group scratch available. The reduction needs one digest per
       work-item, so this bounds the work-group size for wide digests --
       SHA-512 at 512 work-items wants 32 KiB, which not every device has. */
    cl_ulong local_mem;
} vb_ocl_device;

/*
 * Enumerate every device on every platform. Returns the count (0 if none, which
 * is not an error). Never returns more than VB_OCL_MAX_DEVICES.
 */
int vb_ocl_devices(vb_ocl_device *out, int max);

/* "GPU", "CPU", "accelerator" or "other". */
const char *vb_ocl_type_name(cl_device_type t);

#endif /* VALUBENCH_OPENCL_H */
