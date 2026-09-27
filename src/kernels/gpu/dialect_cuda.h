/*
 * dialect_cuda.h -- the CUDA C++ spelling of the device kernel language.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * The counterpart of dialect_opencl.h; read that one first. Compiled by NVRTC
 * at run time, so it may include nothing: NVRTC has no system headers, and the
 * program is assembled from embedded text. hiprtc accepts the same spelling,
 * which is why a HIP backend needs no third dialect.
 */

#define VB_DIALECT_CUDA 1

typedef unsigned int       vb_u32;
typedef unsigned long long vb_u64;

/* size_t needs no declaration: NVRTC and hiprtc both provide it built in. */

/* extern "C" so the entry point keeps its name through C++ mangling, and the
   host can look it up as "vb_md5" exactly as OpenCL does. */
#define VB_KERNEL   extern "C" __global__ void
#define VB_GLOBAL

/*
 * A constant table, spelled so the compiler may fold it.
 *
 * Not __constant__, although that is the literal translation of OpenCL's
 * __constant. A __constant__ variable is writable from the host
 * (cuMemcpyHtoD to its symbol), so the compiler must load every entry at run
 * time, where OpenCL -- whose __constant is immutable -- folds each one into an
 * immediate. On an RTX PRO 2000 with CUDA 13.3 (2026-09-26), SHA-512's round
 * constants as __constant__ cost about two-thirds more registers at one stream
 * and more than doubled the spill at three -- most of what first looked like a
 * toolchain gap. static const __device__ is immutable and folds exactly as
 * OpenCL's does.
 */
#define VB_CONST_TABLE static const __device__

#define VB_INLINE static __device__ __forceinline__

#define VB_GLOBAL_ID()   ((size_t) blockIdx.x * blockDim.x + threadIdx.x)
#define VB_GLOBAL_SIZE() ((size_t) gridDim.x * blockDim.x)
#define VB_LOCAL_ID()    ((size_t) threadIdx.x)
#define VB_LOCAL_SIZE()  ((size_t) blockDim.x)
#define VB_GROUP_ID()    ((size_t) blockIdx.x)
#define VB_GROUP_COUNT() ((size_t) gridDim.x)

#define VB_BARRIER()     __syncthreads()

/* The capacity probe's atomics (device_capacity.h). CUDA's take a pointer
   that is not volatile; the probe's is, for its plain reads. */
#define VB_ATOMIC_INC(p)     atomicAdd((unsigned int *) (p), 1u)
#define VB_ATOMIC_MAX(p, v)  atomicMax((unsigned int *) (p), (v))

/*
 * The work-group scratch. CUDA has no pointer-to-shared argument; the host
 * sizes dynamic shared memory at launch instead, and the kernel declares it.
 * One program holds one kernel, so one extern __shared__ declaration per
 * program cannot collide with another of a different type.
 */
#define VB_SCRATCH_PARAM(T, name)
#define VB_SCRATCH_DECL(T, name)   extern __shared__ T name[];
