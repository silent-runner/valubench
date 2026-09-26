/*
 * dialect_opencl.h -- the OpenCL C spelling of the device kernel language.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Every device kernel is one algorithm core (md5_device_impl.h and siblings)
 * written against the macros below, compiled under one of two dialects: this
 * one, and dialect_cuda.h for CUDA and HIP. A program is the concatenation
 * dialect + device_primitives.h + core, assembled at run time from embedded
 * copies of these files, so nothing is located on disk and the core is the
 * same text whichever API compiles it. That is what keeps an OpenCL-against-
 * CUDA comparison a comparison of toolchains rather than of two ports.
 *
 * The dialect maps the language, not the hardware. What the kernels need that
 * OpenCL and CUDA spell differently is small and fixed: address spaces, the
 * kernel qualifier, work-item indices, the work-group barrier, the work-group
 * scratch buffer and a constant table. How a *primitive* is spelled -- rotate,
 * select -- is a question about the hardware, and lives in
 * device_primitives.h, keyed by vendor rather than by API.
 */

#define VB_DIALECT_OPENCL 1

typedef uint  vb_u32;
typedef ulong vb_u64;

#define VB_KERNEL   __kernel void
#define VB_GLOBAL   __global

/*
 * A constant table. OpenCL's __constant is immutable, so every compiler is
 * free to fold an entry indexed by a literal into an immediate -- which is
 * what happens, and what the CUDA spelling has to match (see dialect_cuda.h).
 */
#define VB_CONST_TABLE __constant

/* Only the NVIDIA inline-PTX spellings in device_primitives.h use functions;
   everything else is a macro, as the kernels always were. */
#define VB_INLINE static inline

#define VB_GLOBAL_ID()    get_global_id(0)
#define VB_GLOBAL_SIZE()  get_global_size(0)
#define VB_LOCAL_ID()     get_local_id(0)
#define VB_LOCAL_SIZE()   get_local_size(0)
#define VB_GROUP_ID()     get_group_id(0)

#define VB_BARRIER()      barrier(CLK_LOCAL_MEM_FENCE)

/*
 * The work-group scratch for the reduction. OpenCL passes it as a __local
 * pointer argument sized by the host (clSetKernelArg with a NULL value), so it
 * is a trailing parameter and there is nothing to declare in the body.
 */
#define VB_SCRATCH_PARAM(T, name)  , __local T *name
#define VB_SCRATCH_DECL(T, name)
