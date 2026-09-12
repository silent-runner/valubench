# valubench -- integer SIMD microbenchmark
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, The valubench authors. See LICENSE.
#
# Deliberately plain. No configure step, no generated build files, no network.
# Building needs only a C11 compiler, POSIX threads, and make.
#
#   make                     build with the default compiler
#   make CC=gcc              build with GCC
#   make CC=clang            build with Clang/LLVM
#   make config              show what this toolchain can build
#   make check               run every correctness test
#
# Two deliberate absences:
#
#   -march=native  would make the binary's behaviour depend on the machine that
#                  compiled it, which is exactly the hidden variable that makes
#                  benchmark results incomparable. Each SIMD path is compiled
#                  into its own object with its own -m flags and selected at
#                  runtime by CPUID instead. See docs/research.md 4.3.
#
#   -flto          link-time optimization could let the compiler prove that
#                  the message corpus is effectively constant and fold message
#                  words into the round constants. That shortcut is exactly what
#                  this workload excludes, because it does less work per hash.
#                  Do not enable it.

CC      ?= cc
# embed_cl runs on the machine doing the building, not on the target, so it gets
# its own compiler. It defaults to $(CC), which is right for a normal build and
# wrong for a cross build -- pass HOSTCC there.
HOSTCC  ?= $(CC)
CFLAGS  ?= -O2
CFLAGS  += -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Iinclude -Isrc
# The embedded OpenCL kernels are generated into $(BUILD) on every build.
CFLAGS  += -I$(BUILD)
LDFLAGS ?=
LDLIBS  ?=

# pthreads. -pthread is understood by both GCC and Clang and sets both the
# compile-time and link-time requirements.
CFLAGS  += -pthread

# A hook for build-time instrumentation -- sanitizers in CI, mainly. Appended
# last so it can override, and empty by default so a normal build is untouched.
CFLAGS  += $(CFLAGS_EXTRA)
LDFLAGS += $(CFLAGS_EXTRA)
LDLIBS  += -pthread

BUILD := build
$(shell mkdir -p $(BUILD))

# ---- toolchain probing --------------------------------------------------
#
# Ask the compiler what it can actually generate rather than assuming, so the
# same Makefile works for GCC, Clang, and cross-compilers with differing ISA
# support. A path is compiled in only if it builds; the runtime CPUID check is
# separate and always applies, because "compiled in" never implies "safe to
# run here".

probe = $(shell printf 'int main(void){return 0;}' > $(BUILD)/.probe.c 2>/dev/null && \
          $(CC) $(1) -c -o $(BUILD)/.probe.o $(BUILD)/.probe.c >/dev/null 2>&1 && \
          echo yes || echo no)

# The target architecture, not the host's. `uname -m` answers the wrong question
# under cross-compilation -- it would report x86_64 while building for aarch64,
# and the build would try to hand SSE2 sources to an ARM compiler. Ask the
# compiler what it targets, and fall back to uname only if it cannot say.
ARCH := $(shell $(CC) -dumpmachine 2>/dev/null | cut -d- -f1)
ifeq ($(ARCH),)
  ARCH := $(shell uname -m 2>/dev/null)
endif

# Use the real Khronos headers when the system has them, our own declarations
# when it does not. Runtime behaviour is identical either way -- libOpenCL is
# always dlopen'd, never linked -- so this only decides where the types come
# from at compile time. Install `opencl-headers` to switch it on.
HAVE_CL_HEADERS := $(shell printf '#include <CL/cl.h>\nint main(void){return 0;}' \
  > $(BUILD)/.clprobe.c 2>/dev/null && \
  $(CC) -DCL_TARGET_OPENCL_VERSION=120 -c -o $(BUILD)/.clprobe.o \
    $(BUILD)/.clprobe.c >/dev/null 2>&1 && echo yes || echo no)

ifeq ($(HAVE_CL_HEADERS),yes)
  CFLAGS += -DVB_HAVE_CL_HEADERS=1
endif

# Does CC exist at all? Without this check a missing compiler makes every ISA
# probe answer "no", and the build quietly succeeds as a scalar-only binary that
# under-reports the machine several-fold with no indication anything is wrong.
CC_OK := $(shell $(CC) --version >/dev/null 2>&1 && echo yes || echo no)

ifeq ($(CC_OK),no)
  ifeq ($(filter clean config,$(MAKECMDGOALS)),)
    $(error C compiler '$(CC)' not found or not runnable.       Install it, or pass one: make CC=gcc / make CC=clang.       Note that the 'llvm' package does not provide the clang driver --       that is the 'clang' package.)
  endif
endif

# AArch64: Advanced SIMD is architecturally mandatory, so the probe is really
# asking whether this toolchain has the intrinsics header rather than whether
# the target supports the instructions. No -m flag is needed or wanted.
HAVE_NEON := no
ifneq ($(filter aarch64 arm64,$(ARCH)),)
  HAVE_NEON := $(shell printf '#include <arm_neon.h>\nint main(void){uint32x4_t v=vdupq_n_u32(1);return (int)vgetq_lane_u32(v,0)-1;}' \
    > $(BUILD)/.neonprobe.c 2>/dev/null && \
    $(CC) -c -o $(BUILD)/.neonprobe.o $(BUILD)/.neonprobe.c >/dev/null 2>&1 && echo yes || echo no)

  # SVE and SVE2 are optional on AArch64 and need an -march flag, so unlike
  # NEON they are real probes. Probing compiles a vector operation rather than
  # just the header: a toolchain can ship arm_sve.h and still reject +sve.
  #
  # This asks what the *compiler* can build, never what the machine can run --
  # the kernels are gated at run time by HWCAP_SVE. A cross build for a machine
  # we cannot execute is exactly the case that has to work.
  HAVE_SVE := $(shell printf '#include <arm_sve.h>\nint main(void){svuint32_t a=svdup_n_u32(1);return (int)svaddv_u32(svptrue_b32(),a)-1;}' \
    > $(BUILD)/.sveprobe.c 2>/dev/null && \
    $(CC) -march=armv8-a+sve -c -o $(BUILD)/.sveprobe.o $(BUILD)/.sveprobe.c >/dev/null 2>&1 && echo yes || echo no)

  HAVE_SVE2 := $(shell printf '#include <arm_sve.h>\nint main(void){svuint32_t a=svdup_n_u32(1);a=svbsl_u32(a,a,a);return (int)svaddv_u32(svptrue_b32(),a)-1;}' \
    > $(BUILD)/.sve2probe.c 2>/dev/null && \
    $(CC) -march=armv8-a+sve2 -c -o $(BUILD)/.sve2probe.o $(BUILD)/.sve2probe.c >/dev/null 2>&1 && echo yes || echo no)
endif

HAVE_SVE  ?= no
HAVE_SVE2 ?= no

ifeq ($(filter x86_64 i686 i386,$(ARCH)),)
  HAVE_SSE2   := no
  HAVE_AVX2   := no
  HAVE_AVX512 := no
  HAVE_SHANI  := no
else
  HAVE_SSE2   := $(call probe,-msse2)
  HAVE_AVX2   := $(call probe,-mavx2)
  HAVE_AVX512 := $(call probe,-mavx512f)
  HAVE_SHANI  := $(call probe,-msha)

  # SSE2 is part of the x86-64 baseline; any working compiler targeting it can
  # emit SSE2. If the probe says otherwise the toolchain is broken, and a build
  # that continues would silently produce scalar-only results.
  ifeq ($(HAVE_SSE2),no)
    ifeq ($(filter clean config,$(MAKECMDGOALS)),)
      $(error compiler '$(CC)' cannot build SSE2 on $(ARCH), which is part of         the x86-64 baseline. The toolchain looks broken; run 'make config'.)
    endif
  endif
endif

# ---- kernels ------------------------------------------------------------
#
# Every CPU kernel lives in src/kernels/cpu/ and is one translation unit
# compiled with its own ISA flags. To add one:
#
#   1. write src/kernels/cpu/<name>.c defining the OPS_* macros and
#      instantiating the template (see src/kernels/cpu/README.md)
#   2. add a KFLAGS_<name> line below, and a HAVE_ guard if it needs one
#   3. add it to KERNELS
#   4. add a block to src/kernels/cpu/matrix.h and a token to
#      VB_FOR_EACH_KERNEL
#
# src/registry.c is *not* touched: its declarations and rows are expanded from
# the matrix. (This comment used to say otherwise, which was true only before
# the matrix existed.)
#
# Nothing else in the build sees those flags, so no ISA can leak into the
# dispatcher or the harness.

# GCC auto-vectorises the independent streams in this translation unit: at -O2
# it turns scalar-s2 into 88% SSE2 on x86 and 79% NEON on AArch64, using two of
# four lanes. That makes the scalar rung secretly a vector one, which is the
# baseline every ISA ratio divides by -- and it is not even
# consistent, since clang does not do it at all. Both spellings are accepted by
# gcc and clang; SLP is the one that fuses the streams.
KFLAGS_scalar := -fno-tree-vectorize -fno-tree-slp-vectorize
KFLAGS_sse2   := -msse2
KFLAGS_avx2   := -mavx2
KFLAGS_avx512 := -mavx512f
KFLAGS_shani  := -msha
KFLAGS_neon   :=            # Advanced SIMD is the AArch64 baseline
KFLAGS_sve    := -march=armv8-a+sve
KFLAGS_sve2   := -march=armv8-a+sve2

KERNELS := scalar
KERNEL_DEFS :=

ifeq ($(HAVE_SSE2),yes)
  KERNELS     += sse2
  KERNEL_DEFS += -DVB_HAVE_SSE2=1
else
  KERNEL_DEFS += -DVB_HAVE_SSE2=0
endif

ifeq ($(HAVE_AVX2),yes)
  KERNELS     += avx2
  KERNEL_DEFS += -DVB_HAVE_AVX2=1
else
  KERNEL_DEFS += -DVB_HAVE_AVX2=0
endif

ifeq ($(HAVE_AVX512),yes)
  KERNELS     += avx512
  KERNEL_DEFS += -DVB_HAVE_AVX512=1
else
  KERNEL_DEFS += -DVB_HAVE_AVX512=0
endif

ifeq ($(HAVE_SHANI),yes)
  KERNELS     += shani
  KERNEL_DEFS += -DVB_HAVE_SHANI=1
else
  KERNEL_DEFS += -DVB_HAVE_SHANI=0
endif

ifeq ($(HAVE_NEON),yes)
  KERNELS     += neon
  KERNEL_DEFS += -DVB_HAVE_NEON=1
else
  KERNEL_DEFS += -DVB_HAVE_NEON=0
endif

ifeq ($(HAVE_SVE),yes)
  KERNELS     += sve
  KERNEL_DEFS += -DVB_HAVE_SVE=1
else
  KERNEL_DEFS += -DVB_HAVE_SVE=0
endif

ifeq ($(HAVE_SVE2),yes)
  KERNELS     += sve2
  KERNEL_DEFS += -DVB_HAVE_SVE2=1
else
  KERNEL_DEFS += -DVB_HAVE_SVE2=0
endif

KERNEL_OBJS := $(addprefix $(BUILD)/kernel_,$(addsuffix .o,$(KERNELS)))

# The vector length in lanes, which the registry needs and which requires
# arm_sve.h -- so it cannot live in registry.c, which must stay free of any
# ISA flag. Its own translation unit, built with +sve, doing nothing else.
# Either ISA needs it: the SVE2 rows in matrix.h call the same two functions.
ifneq ($(filter yes,$(HAVE_SVE) $(HAVE_SVE2)),)
  KERNEL_OBJS += $(BUILD)/sve_lanes.o
endif

# OpenCL is dlopen'd, never linked: -ldl is the only addition, and the binary
# runs unchanged on a machine with no GPU or no OpenCL at all.
OCL_OBJS := $(BUILD)/ocl_loader.o $(BUILD)/ocl_backend.o
LDLIBS   += -ldl

# The scalar reference implementations -- the correctness oracles every kernel
# is validated against. Discovered rather than listed: one per algorithm in
# src/reference/, so adding an algorithm does not also mean remembering to edit
# two lists down here. A reference that is not linked is a kernel with nothing
# to validate against.
REF_OBJS := $(patsubst src/reference/%.c,$(BUILD)/ref_%.o,\
                       $(wildcard src/reference/*.c))

CORE_OBJS := $(REF_OBJS) \
             $(BUILD)/algorithm.o $(BUILD)/workload.o $(BUILD)/cpu_features.o \
             $(BUILD)/bench.o $(BUILD)/sysinfo.o $(BUILD)/report.o \
             $(BUILD)/registry.o $(BUILD)/power.o $(KERNEL_OBJS) $(OCL_OBJS)

HDRS := include/hashes.h include/sha512_const.h \
        include/algorithm.h include/valubench.h \
        include/bench.h include/sysinfo.h include/report.h \
        include/cpu_features.h include/vb_cl.h include/opencl.h \
        include/opencl_backend.h include/power.h
# Every kernel translation unit depends on the whole template set and on the
# matrix, so any of them changing rebuilds all of them.
KHDRS := $(wildcard src/kernels/cpu/*.h)


# Before any target, so a rule added above `all` cannot silently become the
# default goal. One did: the registry.o dependency below sat four lines higher
# and made `make` build a single object and exit 0, so CI's build step passed
# in zero seconds and the next step died on a binary that was never linked.
.DEFAULT_GOAL := all

.PHONY: all test check check-kernels clean config

all: $(BUILD)/valubench $(BUILD)/test_hashes $(BUILD)/test_kernels

config:
	@echo "arch        $(ARCH)"
	@echo "CC          $(CC)"
	@echo "found       $(CC_OK)"
	@echo "version     $$($(CC) --version 2>/dev/null | head -1)"
	@echo "sse2        $(HAVE_SSE2)"
	@echo "avx2        $(HAVE_AVX2)"
	@echo "avx512f     $(HAVE_AVX512)"
	@echo "sha-ni      $(HAVE_SHANI)"
	@echo "neon        $(HAVE_NEON)"
	@echo "sve         $(HAVE_SVE)"
	@echo "sve2        $(HAVE_SVE2)"
	@echo "kernels     $(KERNELS)"
	@echo "CL headers  $(HAVE_CL_HEADERS) $(if $(filter no,$(HAVE_CL_HEADERS)),(using built-in declarations; install opencl-headers to use the real ones),)"

# ---- generated sources --------------------------------------------------
#
# Constant tables are transcribed from their specifications and checked by the
# known-answer vectors, so there is nothing to generate for them. The only
# generated artifact is the embedded OpenCL kernel below.

# The OpenCL kernels are real .cl files carrying their own constants and round
# schedules, so embedding one is a single step: embed_cl turns the file into a
# byte array the binary compiles in, and nothing has to be installed or located
# at run time.
#
# Generated into $(BUILD), not committed. The output is a pure function of the
# .cl file and embed_cl needs nothing but the C compiler the build already
# requires, so a checked-in copy could only be a second source of truth to keep
# in sync -- which it did not: a `check-embed` target existed solely to catch
# drift, and the headers were once found truncated in a working tree with a
# green build behind them.
#
# Adding a device kernel is adding a .cl file. The symbol and include guard are
# derived from its name.
CL_SOURCES := $(wildcard src/kernels/gpu/*.cl)
CL_HEADERS := $(patsubst src/kernels/gpu/%.cl,$(BUILD)/%_kernel.h,$(CL_SOURCES))

$(BUILD)/embed_cl: tools/embed_cl.c
	$(HOSTCC) -O2 -std=c11 -Iinclude -o $@ $<

CL_UC = $(shell echo $(1) | tr a-z A-Z)

# Without this make treats the headers as intermediates of a pattern rule and
# deletes them once the objects are built, leaving nothing to read when a device
# kernel misbehaves.
.SECONDARY: $(CL_HEADERS)

$(BUILD)/%_kernel.h: src/kernels/gpu/%.cl $(BUILD)/embed_cl
	$(BUILD)/embed_cl VB_OCL_$(call CL_UC,$*)_SOURCE \
	    VALUBENCH_OPENCL_$(call CL_UC,$*)_KERNEL_H < $< > $@

# ---- objects ------------------------------------------------------------

$(BUILD)/%.o: src/%.c $(HDRS)
	$(CC) $(CFLAGS) $(KERNEL_DEFS) -c -o $@ $<

$(BUILD)/ref_%.o: src/reference/%.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/kernel_%.o: src/kernels/cpu/%.c $(KHDRS) $(HDRS)
	$(CC) $(CFLAGS) $(KFLAGS_$*) -c -o $@ $<

# Not a kernel, but it needs +sve for arm_sve.h and must not be built without
# it. KERNEL_DEFS so VB_HAVE_SVE means the same thing here as everywhere else.
$(BUILD)/sve_lanes.o: src/kernels/cpu/sve_lanes.c $(HDRS)
	$(CC) $(CFLAGS) $(KERNEL_DEFS) $(KFLAGS_sve) -c -o $@ $<

# KERNEL_DEFS here too: backend.c includes the kernel matrix for the device
# list, and without the ISA defines the matrix would mean something different in
# this translation unit than in every other one.
$(BUILD)/ocl_%.o: src/opencl/%.c $(CL_HEADERS) $(HDRS) $(KHDRS)
	$(CC) $(CFLAGS) $(KERNEL_DEFS) -c -o $@ $<

$(BUILD)/test_hashes.o: tests/test_hashes.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

# registry.c includes kernels/cpu/matrix.h, but is built by the generic
# src/%.c rule whose prerequisites are $(HDRS) alone. Editing the kernel matrix
# therefore left a stale registry.o behind -- the one object whose contents are
# generated from that header. Named explicitly rather than folded into $(HDRS),
# which would rebuild every object for a kernel-only change.
$(BUILD)/registry.o: $(KHDRS)

$(BUILD)/test_report_json.o: tests/test_report_json.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/test_config.o: tests/test_config.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/test_checkpoints.o: tests/test_checkpoints.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/test_kernels.o: tests/test_kernels.c $(HDRS)
	$(CC) $(CFLAGS) $(KERNEL_DEFS) -c -o $@ $<

# ---- binaries -----------------------------------------------------------

$(BUILD)/valubench: $(BUILD)/main.o $(CORE_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_hashes: $(BUILD)/test_hashes.o $(REF_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_kernels: $(BUILD)/test_kernels.o $(CORE_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ---- checks -------------------------------------------------------------

test: $(BUILD)/test_hashes
	$(BUILD)/test_hashes

check-kernels: $(BUILD)/test_kernels
	$(BUILD)/test_kernels

# The scalar rung is the denominator of every ISA ratio reported, so verify it
# is scalar rather than trusting KFLAGS_scalar to have been honoured. Uses the
# toolchain's own objdump so it works under cross-compilation, and skips itself
# if there is none.
# Strip any -NN version suffix first: gcc-15 and clang-20 are ordinary
# spellings and the old pattern matched neither, so OBJDUMP became the
# compiler itself and the scalar-purity guard silently found no kernels.
#
# sed -E because \? is a GNU extension that BSD sed (and thus macOS) takes
# literally, which reproduced that same failure for a different reason: plain
# `cc` went through unchanged and the guard was handed the compiler to
# disassemble with. ERE spells the optional `g` portably.
OBJDUMP ?= $(shell echo $(CC) | sed -E 's/-[0-9]+$$//; s/g?cc$$/objdump/; s/clang/objdump/')

check-scalar: $(BUILD)/kernel_scalar.o
	@sh tests/check_scalar_is_scalar.sh $(BUILD)/kernel_scalar.o $(OBJDUMP)

$(BUILD)/test_checkpoints: $(BUILD)/test_checkpoints.o $(BUILD)/workload.o \
                           $(BUILD)/algorithm.o $(REF_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_report_json: $(BUILD)/test_report_json.o $(CORE_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_power_model: tests/test_power_model.c $(BUILD)/power.o
	$(CC) $(CFLAGS) -o $@ $^ -pthread

# Energy domains must be counted once each: an iGPU inside a package, a card
# seen by two providers, two sockets, two cards. Constructed rather than
# measured, because no one machine has all of these shapes.
check-power: $(BUILD)/test_power_model
	@$(BUILD)/test_power_model

check-report: $(BUILD)/test_report_json
	@$(BUILD)/test_report_json

$(BUILD)/test_virt: tests/test_virt.c $(BUILD)/sysinfo.o $(BUILD)/cpu_features.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^

# Virtual or bare metal, and the shapes where the honest answer is neither.
check-virt: $(BUILD)/test_virt
	@$(BUILD)/test_virt

$(BUILD)/test_config: $(BUILD)/test_config.o $(CORE_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Nothing in a freshly defaulted config may be indeterminate.
check-config: $(BUILD)/test_config
	@$(BUILD)/test_config

# A multi-threaded pool must span more than one CPU.
#
# This is the check that was missing when autotune pinned every worker to one
# core: pool_create() pins the calling thread, vb_allowed_cpus() read the mask
# back from that same thread, and after the first probe every later pool put
# all of its workers on one CPU -- reporting the thread count it was asked for
# and running at a third of the speed. Nothing in the output could express
# "eight threads, one core", so review, sanitizers and five hardware sessions
# all missed it.
#
# No baseline needed, which is the point: this is an invariant, not a
# comparison against a recorded figure. A single-CPU machine cannot test it and
# says so rather than passing quietly.
check-pinning: $(BUILD)/valubench
	@n=$$(nproc 2>/dev/null || echo 1); \
	 if [ "$$n" -lt 2 ]; then \
	   echo "  skip  pinning       (needs >1 cpu; this machine has $$n)"; \
	 else \
	   t=$$(if [ "$$n" -gt 4 ]; then echo 4; else echo "$$n"; fi); \
	   out=$$($(BUILD)/valubench --json --algorithm md5 --where cpu \
	            --threads $$t --samples 3 --time-ms 40 --warmup-ms 40 \
	          2>/dev/null); \
	   got=$$(printf '%s' "$$out" | python3 -c \
	     'import json,sys; e=json.load(sys.stdin)["environment"]; \
print(e["threads_used"], e["pinned_cpus"], int(e.get("can_pin", True)))' 2>/dev/null); \
	   used=$$(echo "$$got" | cut -d" " -f1); \
	   cpus=$$(echo "$$got" | cut -d" " -f2); \
	   canpin=$$(echo "$$got" | cut -d" " -f3); \
	   if [ -z "$$cpus" ]; then \
	     echo "  FAIL  pinning       no pinned_cpus in the result"; exit 1; \
	   elif [ "$$canpin" = 0 ]; then \
	     echo "  skip  pinning       (no thread affinity API on this platform)"; \
	   elif [ "$$cpus" -lt 2 ]; then \
	     echo "  FAIL  pinning       $$used threads pinned onto $$cpus cpu"; exit 1; \
	   else \
	     echo "  ok    pinning       ($$used threads across $$cpus cpus)"; \
	   fi; \
	 fi

check-contract: $(BUILD)/valubench
	@sh tests/check_output_contract.sh $(BUILD)/valubench .

$(BUILD)/fail_pthread_create.so: tests/fail_pthread_create.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $< -ldl

$(BUILD)/test_thread_failure: tests/test_thread_failure.c $(BUILD)/workload.o \
                              $(BUILD)/algorithm.o $(REF_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -pthread

# A thread that fails to start must not corrupt the reference. Injected at each
# index in turn, because the defect this covers only appears when the failure is
# not the last one -- a contiguous prefix of successes was always handled.
check-threadfail: $(BUILD)/test_thread_failure $(BUILD)/fail_pthread_create.so \
                  $(BUILD)/valubench
	@for n in 1 2 3 4; do \
	   LD_PRELOAD=$(BUILD)/fail_pthread_create.so VB_FAIL_CREATE=$$n \
	     $(BUILD)/test_thread_failure > $(BUILD)/.tf.log 2>&1 || \
	     { echo "  FAIL  thread-failure with create $$n failing"; \
	       cat $(BUILD)/.tf.log; exit 1; }; \
	 done; \
	 EXP=$$($(BUILD)/valubench --reference-ladder 1 --algorithm md5 \
	          --message-bytes 55 --working-set-kb 1024 \
	        | grep -o '"checksum": "[0-9a-f]*"' | cut -d'"' -f4); \
	 for n in 1 2 3 4 5 6; do \
	   out=$$(LD_PRELOAD=$(BUILD)/fail_pthread_create.so VB_FAIL_CREATE=$$n \
	          timeout -s KILL 60 \
	          $(BUILD)/valubench --json --kernel md5/scalar-s1 --threads 4 \
	          --expect $$EXP --samples 2 --time-ms 20 --warmup-ms 20 2>/dev/null) \
	         && rc=0 || rc=$$?; \
	   if [ "$$rc" = 137 ] || [ "$$rc" = 124 ]; then \
	     echo "  FAIL  threadfail  create $$n hung; a stalled pool is not a pass"; \
	     exit 1; \
	   fi; \
	   if [ "$$rc" != 0 ]; then continue; fi; \
	   used=$$(printf '%s' "$$out" | python3 -c \
	     'import json,sys; print(json.load(sys.stdin)["environment"]["threads_used"])' \
	     2>/dev/null); \
	   if [ "$$used" != 4 ]; then \
	     echo "  FAIL  threadfail  a pool of $$used reported a result (wanted 4)"; \
	     exit 1; \
	   fi; \
	 done
	 $(BUILD)/test_thread_failure | sed 's/^/  ok    threadfail  /'

check-checkpoints: $(BUILD)/test_checkpoints
	@$(BUILD)/test_checkpoints

check-working-set: $(BUILD)/valubench
	@sh tests/check_working_set.sh $(BUILD)/valubench

check: test check-kernels check-scalar check-checkpoints check-threadfail \
       check-power \
       check-pinning \
       check-virt \
       check-config \
       check-working-set \
       check-report check-contract

clean:
	rm -rf $(BUILD)
