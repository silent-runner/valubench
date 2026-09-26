/*
 * cuda_ptx.h -- running NVIDIA OpenCL's PTX through the CUDA driver.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * Pure text in, text out, so it is its own translation unit: a test can check
 * the translation on a machine with no NVIDIA driver, which is every CI runner.
 */

#ifndef VALUBENCH_CUDA_PTX_H
#define VALUBENCH_CUDA_PTX_H

#include <stddef.h>

/*
 * Translate PTX from NVIDIA's OpenCL compiler to what the CUDA driver loads:
 * parameter state-space qualifiers dropped, and OpenCL's launch-environment
 * registers replaced by their CUDA values. *ocl_abi is set when the kernel
 * takes a __local argument, which the launch must then pass the dynamic
 * shared base. Returns a malloc'd string, or NULL with `err` saying why.
 */
char *vb_ptx_from_opencl(const char *in, int *ocl_abi, char *err, size_t errn);

#endif /* VALUBENCH_CUDA_PTX_H */
