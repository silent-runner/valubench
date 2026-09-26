/*
 * vb_cl.h -- the OpenCL interface valubench uses.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 *
 * TWO WAYS TO GET THE DECLARATIONS
 * ================================
 *
 * If the build finds <CL/cl.h>, we use it: authoritative types, and any future
 * enumerant is there for free. If not, the block below declares the subset we
 * call. Either way the *runtime* behaviour is identical, because nothing is
 * ever linked -- libOpenCL is dlopen'd and every entry point resolved by
 * src/opencl/loader.c.
 *
 * Why not simply require the headers: the project asks for a benchmark that builds
 * on a bare Linux box. Requiring an OpenCL SDK to compile a binary that may
 * never see a GPU trades that away for nothing. Why not simply use our own:
 * when the real headers are present they are more authoritative than anything
 * hand-written, and building both ways cross-checks the two.
 *
 * Why not link -lOpenCL: that would give the binary a hard runtime dependency
 * on libOpenCL.so.1, so it would fail to *start* on a machine without OpenCL --
 * the opposite of "GPU kernels when the system has OpenCL, CPU-only when it
 * does not".
 *
 * Note on licensing: including a system header is not redistributing it, so the
 * Apache-2.0 terms on the Khronos headers create no obligation here. Vendoring
 * a copy into the repository would, which is why we do not.
 *
 * The enumerant values in the fallback are fixed by the OpenCL specification
 * and are ABI, not authorship -- CL_DEVICE_NAME has to be 0x102B for the call
 * to work. Each was checked against the published specification.
 */

#ifndef VALUBENCH_VB_CL_H
#define VALUBENCH_VB_CL_H

#include <stddef.h>
#include <stdint.h>

#ifdef VB_HAVE_CL_HEADERS

/* Pin the API level so the headers do not warn about 1.2 entry points we use
   deliberately (clCreateCommandQueue has no portable 2.0 replacement). */
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#include <CL/cl.h>

#else  /* ---- fallback declarations ---- */

typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef uint64_t  cl_ulong;
typedef cl_uint   cl_bool;
typedef cl_ulong  cl_bitfield;

typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_map_flags;
typedef cl_uint     cl_program_info;
typedef cl_uint     cl_platform_info;
typedef cl_uint     cl_device_info;
typedef cl_uint     cl_program_build_info;
typedef cl_uint     cl_kernel_work_group_info;
typedef cl_uint     cl_profiling_info;

/* Opaque handles. The runtime only ever hands these back to us. */
typedef struct _cl_platform_id   *cl_platform_id;
typedef struct _cl_device_id     *cl_device_id;
typedef struct _cl_context       *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem           *cl_mem;
typedef struct _cl_program       *cl_program;
typedef struct _cl_kernel        *cl_kernel;
typedef struct _cl_event         *cl_event;

typedef intptr_t cl_context_properties;

/* On Linux the OpenCL calling convention is the platform default. */
#define CL_API_CALL
#define CL_CALLBACK

#define CL_SUCCESS                       0
#define CL_FALSE                         0
#define CL_TRUE                          1

#define CL_DEVICE_TYPE_CPU               (1 << 1)
#define CL_DEVICE_TYPE_GPU               (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR       (1 << 3)
#define CL_DEVICE_TYPE_ALL               0xFFFFFFFFu

#define CL_QUEUE_PROFILING_ENABLE        (1 << 1)

#define CL_PLATFORM_VERSION              0x0901
#define CL_PLATFORM_NAME                 0x0902
#define CL_PLATFORM_VENDOR               0x0903

#define CL_DEVICE_TYPE                   0x1000
#define CL_DEVICE_VENDOR_ID              0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS      0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE    0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY    0x100C
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE     0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE        0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE         0x1023
#define CL_DEVICE_NAME                   0x102B
#define CL_DEVICE_VENDOR                 0x102C
#define CL_DRIVER_VERSION                0x102D
#define CL_DEVICE_VERSION                0x102F
#define CL_DEVICE_EXTENSIONS             0x1030

#define CL_MEM_READ_WRITE                (1 << 0)
#define CL_MEM_WRITE_ONLY                (1 << 1)
#define CL_MEM_READ_ONLY                 (1 << 2)
#define CL_MEM_USE_HOST_PTR              (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR            (1 << 4)
#define CL_MEM_COPY_HOST_PTR             (1 << 5)

#define CL_MAP_READ                      (1 << 0)
#define CL_MAP_WRITE                     (1 << 1)

#define CL_PROGRAM_BINARY_SIZES          0x1165
#define CL_PROGRAM_BINARIES              0x1166
#define CL_PROGRAM_BUILD_LOG             0x1183

#define CL_KERNEL_WORK_GROUP_SIZE                     0x11B0
#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE  0x11B3

#define CL_PROFILING_COMMAND_START       0x1282
#define CL_PROFILING_COMMAND_END         0x1283

#endif /* VB_HAVE_CL_HEADERS */

/*
 * Vendor queries for a device's PCI address, which is what lets one card seen
 * through two APIs be recognised as one device. Not in every <CL/cl.h>, so
 * defined here in both modes; the values are fixed by each extension's
 * registry entry. cl_khr_pci_bus_info is preferred because it carries the PCI
 * domain; the NVIDIA and AMD queries do not, which is ambiguous on a
 * multi-domain server and is reported as such rather than guessed.
 */
#ifndef CL_DEVICE_PCI_BUS_INFO_KHR
#define CL_DEVICE_PCI_BUS_INFO_KHR       0x410F
#endif
#ifndef CL_DEVICE_PCI_BUS_ID_NV
#define CL_DEVICE_PCI_BUS_ID_NV          0x4008
#endif
#ifndef CL_DEVICE_PCI_SLOT_ID_NV
#define CL_DEVICE_PCI_SLOT_ID_NV         0x4009
#endif
#ifndef CL_DEVICE_TOPOLOGY_AMD
#define CL_DEVICE_TOPOLOGY_AMD           0x4037
#endif

typedef struct {
    cl_uint pci_domain;
    cl_uint pci_bus;
    cl_uint pci_device;
    cl_uint pci_function;
} vb_cl_pci_bus_info_khr;

/* cl_device_topology_amd: a 24-byte union whose PCIe member is a type word,
   17 bytes of padding, then bus, device and function as single bytes. */
typedef union {
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; char unused[17]; char bus; char device;
             char function; } pcie;
} vb_cl_topology_amd;

/* ---- entry points we resolve at runtime --------------------------------- */
/*
 * Declared here in both modes: with the real headers these mirror the upstream
 * prototypes (and the compiler checks nothing, since we only ever dlsym), and
 * without them they are the only declaration.
 */

typedef cl_int (CL_API_CALL *cl_fn_GetPlatformIDs)(
    cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (CL_API_CALL *cl_fn_GetPlatformInfo)(
    cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *cl_fn_GetDeviceIDs)(
    cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (CL_API_CALL *cl_fn_GetDeviceInfo)(
    cl_device_id, cl_device_info, size_t, void *, size_t *);

typedef cl_context (CL_API_CALL *cl_fn_CreateContext)(
    const cl_context_properties *, cl_uint, const cl_device_id *,
    void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
    void *, cl_int *);
typedef cl_int (CL_API_CALL *cl_fn_ReleaseContext)(cl_context);

/* Deprecated in OpenCL 2.0 but present in every ICD; the 2.0 replacement is
   not, so this is the portable choice. */
typedef cl_command_queue (CL_API_CALL *cl_fn_CreateCommandQueue)(
    cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
typedef cl_int (CL_API_CALL *cl_fn_ReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CL_API_CALL *cl_fn_Finish)(cl_command_queue);
typedef cl_int (CL_API_CALL *cl_fn_Flush)(cl_command_queue);

typedef cl_mem (CL_API_CALL *cl_fn_CreateBuffer)(
    cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_int (CL_API_CALL *cl_fn_ReleaseMemObject)(cl_mem);
typedef cl_int (CL_API_CALL *cl_fn_EnqueueWriteBuffer)(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *,
    cl_uint, const cl_event *, cl_event *);
typedef cl_int (CL_API_CALL *cl_fn_EnqueueReadBuffer)(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *,
    cl_uint, const cl_event *, cl_event *);
typedef void * (CL_API_CALL *cl_fn_EnqueueMapBuffer)(
    cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t,
    cl_uint, const cl_event *, cl_event *, cl_int *);
typedef cl_int (CL_API_CALL *cl_fn_EnqueueUnmapMemObject)(
    cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);

typedef cl_program (CL_API_CALL *cl_fn_CreateProgramWithSource)(
    cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (CL_API_CALL *cl_fn_BuildProgram)(
    cl_program, cl_uint, const cl_device_id *, const char *,
    void (CL_CALLBACK *)(cl_program, void *), void *);
typedef cl_int (CL_API_CALL *cl_fn_GetProgramBuildInfo)(
    cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *cl_fn_ReleaseProgram)(cl_program);
typedef cl_int (CL_API_CALL *cl_fn_GetProgramInfo)(
    cl_program, cl_program_info, size_t, void *, size_t *);

typedef cl_kernel (CL_API_CALL *cl_fn_CreateKernel)(
    cl_program, const char *, cl_int *);
typedef cl_int (CL_API_CALL *cl_fn_SetKernelArg)(
    cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (CL_API_CALL *cl_fn_GetKernelWorkGroupInfo)(
    cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *,
    size_t *);
typedef cl_int (CL_API_CALL *cl_fn_ReleaseKernel)(cl_kernel);

typedef cl_int (CL_API_CALL *cl_fn_EnqueueNDRangeKernel)(
    cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
    const size_t *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (CL_API_CALL *cl_fn_GetEventProfilingInfo)(
    cl_event, cl_profiling_info, size_t, void *, size_t *);
typedef cl_int (CL_API_CALL *cl_fn_ReleaseEvent)(cl_event);

#endif /* VALUBENCH_VB_CL_H */
