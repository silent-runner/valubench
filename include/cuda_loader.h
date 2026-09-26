/*
 * cuda_loader.h -- the CUDA driver API and NVRTC, loaded at run time.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * The counterpart of opencl.h. Nothing is linked: libcuda.so.1 (the driver)
 * and libnvrtc (the runtime compiler) are dlopen'd on first use and every
 * entry point resolved into a table, so the binary builds with no CUDA
 * toolkit and starts on a machine with no NVIDIA driver at all. Either
 * library missing is a normal outcome -- CUDA kernels are then unavailable,
 * as OpenCL ones are without an ICD -- reported through vb_cuda_error().
 *
 * NVRTC is required, not optional. The kernels are compiled at run time for
 * the device in front of them, as OpenCL's are, rather than for a list of
 * architectures fixed when the binary was built -- a build-time list would be
 * the device form of -march=native, baking the build machine into the result.
 * A driver-only machine therefore reports CUDA unavailable, and says it is
 * NVRTC that is missing.
 */

#ifndef VALUBENCH_CUDA_LOADER_H
#define VALUBENCH_CUDA_LOADER_H

#include "vb_cuda.h"

typedef struct {
    void *lib;          /* libcuda */
    void *nvrtc_lib;    /* libnvrtc */

    cu_fn_Init                     Init;
    cu_fn_DriverGetVersion         DriverGetVersion;
    cu_fn_DeviceGetCount           DeviceGetCount;
    cu_fn_DeviceGet                DeviceGet;
    cu_fn_DeviceGetName            DeviceGetName;
    cu_fn_DeviceGetAttribute       DeviceGetAttribute;
    cu_fn_DeviceTotalMem           DeviceTotalMem;
    cu_fn_DevicePrimaryCtxRetain   DevicePrimaryCtxRetain;
    cu_fn_DevicePrimaryCtxRelease  DevicePrimaryCtxRelease;
    cu_fn_CtxSetCurrent            CtxSetCurrent;
    cu_fn_ModuleLoadDataEx         ModuleLoadDataEx;
    cu_fn_ModuleUnload             ModuleUnload;
    cu_fn_ModuleGetFunction        ModuleGetFunction;
    cu_fn_FuncGetAttribute         FuncGetAttribute;
    cu_fn_MemAlloc                 MemAlloc;
    cu_fn_MemFree                  MemFree;
    cu_fn_MemHostAlloc             MemHostAlloc;
    cu_fn_MemFreeHost              MemFreeHost;
    cu_fn_MemcpyHtoD               MemcpyHtoD;
    cu_fn_MemcpyHtoDAsync          MemcpyHtoDAsync;
    cu_fn_MemcpyDtoHAsync          MemcpyDtoHAsync;
    cu_fn_StreamCreate             StreamCreate;
    cu_fn_StreamDestroy            StreamDestroy;
    cu_fn_StreamSynchronize        StreamSynchronize;
    cu_fn_EventCreate              EventCreate;
    cu_fn_EventDestroy             EventDestroy;
    cu_fn_EventRecord              EventRecord;
    cu_fn_EventElapsedTime         EventElapsedTime;
    cu_fn_LaunchKernel             LaunchKernel;
    cu_fn_GetErrorName             GetErrorName;        /* optional */

    nvrtc_fn_Version               nvrtcVersion;
    nvrtc_fn_CreateProgram         nvrtcCreateProgram;
    nvrtc_fn_DestroyProgram        nvrtcDestroyProgram;
    nvrtc_fn_CompileProgram        nvrtcCompileProgram;
    nvrtc_fn_GetSize               nvrtcGetProgramLogSize;
    nvrtc_fn_GetData               nvrtcGetProgramLog;
    nvrtc_fn_GetSize               nvrtcGetPTXSize;
    nvrtc_fn_GetData               nvrtcGetPTX;
    nvrtc_fn_GetSize               nvrtcGetCUBINSize;
    nvrtc_fn_GetData               nvrtcGetCUBIN;
    nvrtc_fn_GetErrorString        nvrtcGetErrorString;

    int  driver_version;        /* cuDriverGetVersion, e.g. 13030 */
    int  nvrtc_major, nvrtc_minor;
    char kernel_driver[32];     /* the NVIDIA driver release, e.g. 610.57.04 */
} vb_cuda;

/* Load both libraries and initialise the driver. 1 on success, 0 if CUDA is
   unavailable, which is not an error. Idempotent. */
int vb_cuda_load(void);

/* NULL until vb_cuda_load() has succeeded. */
const vb_cuda *vb_cuda_api(void);

/* Why CUDA is unavailable, or NULL. Static storage. */
const char *vb_cuda_error(void);

/* The driver's name for an error code, e.g. "CUDA_ERROR_INVALID_VALUE". */
const char *vb_cuda_strerror(vb_CUresult r);

#endif /* VALUBENCH_CUDA_LOADER_H */
