/*
 * vb_cuda.h -- the CUDA driver API and NVRTC interface valubench uses.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Declared here rather than taken from <cuda.h> and <nvrtc.h>, so the build
 * needs no CUDA toolkit: libcuda and libnvrtc are dlopen'd by
 * src/cuda/loader.c and every entry point is resolved into a table, exactly
 * as OpenCL is. A machine without the toolkit builds the same binary, and one
 * without the driver runs it and reports CUDA unavailable.
 *
 * Only the driver API (libcuda), never the runtime (libcudart): the runtime
 * is a convenience layer that would add a second library to find and a
 * second version to report, and the kernels are compiled at run time anyway.
 *
 * The enumerant values are ABI, not authorship -- the driver reads
 * CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as 16 whoever wrote the 16 -- and
 * were checked against CUDA 13.3's headers. With VB_HAVE_CUDA_HEADERS defined
 * (pass the toolkit's include directory in CFLAGS_EXTRA) the real headers are
 * included as well and every value below is checked against them at compile
 * time, the way building with and without <CL/cl.h> cross-checks vb_cl.h.
 * (cuda.h itself draws -Wpedantic warnings; the assertions are what matter.)
 */

#ifndef VALUBENCH_VB_CUDA_H
#define VALUBENCH_VB_CUDA_H

#include <stddef.h>

/* ---- the driver API ------------------------------------------------------ */

typedef int                 vb_CUresult;
typedef int                 vb_CUdevice;
typedef unsigned long long  vb_CUdeviceptr;
typedef struct vb_CUctx_st     *vb_CUcontext;
typedef struct vb_CUmod_st     *vb_CUmodule;
typedef struct vb_CUfunc_st    *vb_CUfunction;
typedef struct vb_CUstream_st  *vb_CUstream;
typedef struct vb_CUevent_st   *vb_CUevent;

#define VB_CUDA_SUCCESS                                0

#define VB_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK         1
#define VB_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK   8
#define VB_CU_DEVICE_ATTRIBUTE_CLOCK_RATE                    13
#define VB_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT          16
#define VB_CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                    33
#define VB_CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                 34
#define VB_CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                 50
#define VB_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR      75
#define VB_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR      76

#define VB_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK           0
#define VB_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES                3
#define VB_CU_FUNC_ATTRIBUTE_NUM_REGS                        4

#define VB_CU_JIT_INFO_LOG_BUFFER                            3
#define VB_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES                 4
#define VB_CU_JIT_ERROR_LOG_BUFFER                           5
#define VB_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES                6
#define VB_CU_JIT_LOG_VERBOSE                                12

typedef vb_CUresult (*cu_fn_Init)(unsigned);
typedef vb_CUresult (*cu_fn_DriverGetVersion)(int *);
typedef vb_CUresult (*cu_fn_DeviceGetCount)(int *);
typedef vb_CUresult (*cu_fn_DeviceGet)(vb_CUdevice *, int);
typedef vb_CUresult (*cu_fn_DeviceGetName)(char *, int, vb_CUdevice);
typedef vb_CUresult (*cu_fn_DeviceGetAttribute)(int *, int, vb_CUdevice);
typedef vb_CUresult (*cu_fn_DeviceTotalMem)(size_t *, vb_CUdevice);
typedef vb_CUresult (*cu_fn_DevicePrimaryCtxRetain)(vb_CUcontext *,
                                                    vb_CUdevice);
typedef vb_CUresult (*cu_fn_DevicePrimaryCtxRelease)(vb_CUdevice);
typedef vb_CUresult (*cu_fn_CtxSetCurrent)(vb_CUcontext);
typedef vb_CUresult (*cu_fn_ModuleLoadDataEx)(vb_CUmodule *, const void *,
                                              unsigned, int *, void **);
typedef vb_CUresult (*cu_fn_ModuleUnload)(vb_CUmodule);
typedef vb_CUresult (*cu_fn_ModuleGetFunction)(vb_CUfunction *, vb_CUmodule,
                                               const char *);
typedef vb_CUresult (*cu_fn_FuncGetAttribute)(int *, int, vb_CUfunction);
typedef vb_CUresult (*cu_fn_MemAlloc)(vb_CUdeviceptr *, size_t);
typedef vb_CUresult (*cu_fn_MemFree)(vb_CUdeviceptr);
typedef vb_CUresult (*cu_fn_MemHostAlloc)(void **, size_t, unsigned);
typedef vb_CUresult (*cu_fn_MemFreeHost)(void *);
typedef vb_CUresult (*cu_fn_MemcpyHtoD)(vb_CUdeviceptr, const void *, size_t);
typedef vb_CUresult (*cu_fn_MemcpyHtoDAsync)(vb_CUdeviceptr, const void *,
                                             size_t, vb_CUstream);
typedef vb_CUresult (*cu_fn_MemcpyDtoHAsync)(void *, vb_CUdeviceptr, size_t,
                                             vb_CUstream);
typedef vb_CUresult (*cu_fn_StreamCreate)(vb_CUstream *, unsigned);
typedef vb_CUresult (*cu_fn_StreamDestroy)(vb_CUstream);
typedef vb_CUresult (*cu_fn_StreamSynchronize)(vb_CUstream);
typedef vb_CUresult (*cu_fn_EventCreate)(vb_CUevent *, unsigned);
typedef vb_CUresult (*cu_fn_EventDestroy)(vb_CUevent);
typedef vb_CUresult (*cu_fn_EventRecord)(vb_CUevent, vb_CUstream);
typedef vb_CUresult (*cu_fn_EventElapsedTime)(float *, vb_CUevent, vb_CUevent);
typedef vb_CUresult (*cu_fn_EventSynchronize)(vb_CUevent);
typedef vb_CUresult (*cu_fn_StreamWaitEvent)(vb_CUstream, vb_CUevent, unsigned);
typedef vb_CUresult (*cu_fn_LaunchKernel)(vb_CUfunction, unsigned, unsigned,
                                          unsigned, unsigned, unsigned,
                                          unsigned, unsigned, vb_CUstream,
                                          void **, void **);
typedef vb_CUresult (*cu_fn_GetErrorName)(vb_CUresult, const char **);

/* ---- NVRTC --------------------------------------------------------------- */

typedef int                      vb_nvrtcResult;
typedef struct vb_nvrtcProg_st  *vb_nvrtcProgram;

#define VB_NVRTC_SUCCESS 0

typedef vb_nvrtcResult (*nvrtc_fn_Version)(int *, int *);
typedef vb_nvrtcResult (*nvrtc_fn_CreateProgram)(vb_nvrtcProgram *,
                                                 const char *, const char *,
                                                 int, const char *const *,
                                                 const char *const *);
typedef vb_nvrtcResult (*nvrtc_fn_DestroyProgram)(vb_nvrtcProgram *);
typedef vb_nvrtcResult (*nvrtc_fn_CompileProgram)(vb_nvrtcProgram, int,
                                                  const char *const *);
typedef vb_nvrtcResult (*nvrtc_fn_GetSize)(vb_nvrtcProgram, size_t *);
typedef vb_nvrtcResult (*nvrtc_fn_GetData)(vb_nvrtcProgram, char *);
typedef const char *   (*nvrtc_fn_GetErrorString)(vb_nvrtcResult);

/* ---- cross-check against the real headers, when they are present -------- */

#ifdef VB_HAVE_CUDA_HEADERS
#include <cuda.h>
#include <nvrtc.h>
_Static_assert(CUDA_SUCCESS == VB_CUDA_SUCCESS, "CUDA_SUCCESS");
_Static_assert(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK ==
               VB_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK ==
               VB_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_CLOCK_RATE ==
               VB_CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT ==
               VB_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID ==
               VB_CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID ==
               VB_CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID ==
               VB_CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ==
               VB_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "attr");
_Static_assert(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR ==
               VB_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "attr");
_Static_assert(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK ==
               VB_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "func attr");
_Static_assert(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES ==
               VB_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, "func attr");
_Static_assert(CU_FUNC_ATTRIBUTE_NUM_REGS ==
               VB_CU_FUNC_ATTRIBUTE_NUM_REGS, "func attr");
_Static_assert(CU_JIT_INFO_LOG_BUFFER == VB_CU_JIT_INFO_LOG_BUFFER, "jit");
_Static_assert(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES ==
               VB_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, "jit");
_Static_assert(CU_JIT_ERROR_LOG_BUFFER == VB_CU_JIT_ERROR_LOG_BUFFER, "jit");
_Static_assert(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES ==
               VB_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, "jit");
_Static_assert(CU_JIT_LOG_VERBOSE == VB_CU_JIT_LOG_VERBOSE, "jit");
_Static_assert(NVRTC_SUCCESS == VB_NVRTC_SUCCESS, "nvrtc");
_Static_assert(sizeof(CUdeviceptr) == sizeof(vb_CUdeviceptr), "CUdeviceptr");
_Static_assert(sizeof(CUdevice) == sizeof(vb_CUdevice), "CUdevice");
#endif

#endif /* VALUBENCH_VB_CUDA_H */
