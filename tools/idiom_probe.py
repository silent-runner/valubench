#!/usr/bin/env python3
"""idiom_probe.py -- does a device toolchain find each hash primitive's idiom?

SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026, The valubench authors. See LICENSE.

Usage: idiom_probe.py [--kernels] [--json] [--sm LIST] [--gfx LIST]
                      [--only TOOLCHAINS]

Whether a compiler turns a rotate written in plain C into one funnel shift is a
property of that compiler version, not of the hardware, so it does not belong
in a benchmark result. It still matters: it is the performance left on the
table by code that does not steer, and it is the evidence every steer in
src/device/steer.c has to cite.

For each primitive -- 32-bit rotate, 64-bit rotate, select (Ch), majority,
three-way XOR, byte swap -- and each spelling device_primitives.h offers, this
compiles sixteen independent uses and the same kernel with the primitive
removed, and reports the difference in machine instructions per use.
The chain is composed exactly as the backends compose a kernel: a dialect
header, then src/kernels/gpu/device_primitives.h, then the probe body. So a
spelling is measured in the form the kernels use it.

A primitive measured alone is not the whole story, and was once wrong: NVIDIA's
OpenCL compiler finds the 32-bit funnel shift in every isolated rotate but
misses most of SHA-1's in context, which cost that kernel about a quarter more
instructions. So --kernels also compiles each real kernel with one primitive
steered at a time and reports instructions and registers against all-plain.
That in-context figure is what decides a steer; the isolated one explains it.

Static, so there is no timing noise, and mostly offline:

  nvrtc     NVRTC to PTX, then ptxas, for any NVIDIA architecture (--sm).
            Needs libnvrtc and ptxas; no device.
  opencl    NVIDIA's OpenCL compiler, which lives in the driver, so it needs
            an NVIDIA device. Its PTX is assembled by the same ptxas.
  amdgpu    clang's AMDGPU backend on the OpenCL dialect (--gfx), offline.
            Without ROCm's device libraries OpenCL builtins are unresolved
            calls, so only the plain and amdgcn spellings are probed there.

Toolchains that are not present are skipped and say why. Standard library
only; libnvrtc and libOpenCL are reached through ctypes.
"""
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.path.join(os.path.dirname(HERE), "src", "kernels", "gpu")
N = 16

# Spelling ids, as device_primitives.h numbers them.
PLAIN, CLB, PTX, AMDGCN = 0, 1, 2, 3
SPELLING = {PLAIN: "plain", CLB: "opencl-builtin", PTX: "ptx", AMDGCN: "amdgcn"}

# Rotate counts the kernels actually use, so a toolchain that special-cases a
# count is measured on the counts that matter.
MD5_S = (7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21)
SHA512_S = (28, 34, 39, 14, 18, 41, 1, 8, 19, 61, 28, 34, 39, 14, 18, 41)


def primitives():
    """(name, word type, steer define or None, expression of x y z j, spellings)."""
    return [
        ("rotl32", "vb_u32", "VB_STEER_ROTL32",
         lambda j: "VB_ROTL32(x, %du)" % MD5_S[j % 16], (PLAIN, CLB, PTX, AMDGCN)),
        ("rotr64", "vb_u64", "VB_STEER_ROTR64",
         lambda j: "VB_ROTR64(x, %du)" % SHA512_S[j % 16], (PLAIN, CLB, PTX, AMDGCN)),
        ("ch32", "vb_u32", "VB_STEER_CH", lambda j: "VB_CH32(x, y, z)", (PLAIN, CLB, PTX)),
        ("maj32", "vb_u32", "VB_STEER_MAJ", lambda j: "VB_MAJ32(x, y, z)", (PLAIN, CLB, PTX)),
        ("ch64", "vb_u64", "VB_STEER_CH", lambda j: "VB_CH64(x, y, z)", (PLAIN, CLB, PTX)),
        ("maj64", "vb_u64", "VB_STEER_MAJ", lambda j: "VB_MAJ64(x, y, z)", (PLAIN, CLB, PTX)),
        ("xor3_32", "vb_u32", None, lambda j: "(x ^ y ^ z)", (PLAIN,)),
        ("bswap32", "vb_u32", None,
         lambda j: ("((x >> 24) | ((x >> 8) & 0xff00u) | "
                    "((x << 8) & 0xff0000u) | (x << 24))"), (PLAIN,)),
    ]


def probe_source(dialect, word, expr, with_primitive):
    """
    N independent uses of the primitive, or the same kernel without it.

    Each use loads three words per work-item and stores three back, and the
    only difference between the two kernels is whether the first of them is
    replaced by the primitive's result. So loads, stores and addressing are
    identical, nothing can fuse with a neighbouring use, and the difference in
    machine instructions is the primitive's own cost. Addressing by work-item
    keeps the values per-lane, as they are in the kernels -- a probe whose
    inputs are uniform measures a GPU's scalar datapath instead.
    """
    lines = ["    const size_t b = (size_t) VB_PROBE_LANE * %d;" % (3 * N)]
    for j in range(N):
        e = expr(j) if with_primitive else "x"
        lines.append("    { %s x = in[b + %d], y = in[b + %d], z = in[b + %d];\n"
                     "      out[b + %d] = %s; out[b + %d] = y; out[b + %d] = z; }"
                     % (word, 3 * j, 3 * j + 1, 3 * j + 2, 3 * j, e,
                        3 * j + 1, 3 * j + 2))
    return (dialect + open(os.path.join(GPU, "device_primitives.h")).read() +
            "\nVB_KERNEL vb_probe(VB_GLOBAL const %s *in, VB_GLOBAL %s *out)\n{\n"
            % (word, word) + "\n".join(lines) + "\n}\n")


# ---- counting --------------------------------------------------------------

SASS_LINE = re.compile(r"^\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_.]+)?)")


def sass_ops(cuobjdump, cubin):
    out = subprocess.run([cuobjdump, "-sass", cubin], capture_output=True,
                         text=True, check=True).stdout
    ops = []
    for line in out.splitlines():
        m = SASS_LINE.match(line)
        if m and not m.group(1).startswith("NOP"):
            ops.append(m.group(1))
    return ops


def gcn_ops(asm):
    """The kernel's own instructions. Recent clang also emits a callable copy
    of every OpenCL kernel (__clang_ocl_kern_imp_*), which would double every
    count; only the entry point, up to its s_endpgm, is the kernel."""
    ops, inside = [], False
    for line in asm.splitlines():
        if re.match(r"^vb_probe:", line):
            inside = True
            continue
        if not inside:
            continue
        m = re.match(r"^\s+([sv]_[a-z0-9_]+)", line)
        if not m:
            continue
        if m.group(1) == "s_endpgm":
            break
        if m.group(1) != "s_nop":
            ops.append(m.group(1))
    return ops


def delta(with_ops, without_ops):
    """Per-use instruction cost, and the mnemonics that account for it."""
    counts = {}
    for op in with_ops:
        counts[op] = counts.get(op, 0) + 1
    for op in without_ops:
        counts[op] = counts.get(op, 0) - 1
    mix = {k: v for k, v in counts.items() if v}
    return (len(with_ops) - len(without_ops)) / N, mix


# ---- toolchains ------------------------------------------------------------

def find_tool(name):
    found = shutil.which(name)
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"),
                 "/usr/local/cuda", "/opt/cuda"):
        if root and os.access(os.path.join(root, "bin", name), os.X_OK):
            return os.path.join(root, "bin", name)
    return None


class Nvrtc:
    """libnvrtc through ctypes, found the way the CUDA backend finds it."""

    NAMES = ("libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12")

    def __init__(self):
        self.lib, err = None, None
        dirs = [""] + [os.path.join(r, "lib64") + "/" for r in
                       (os.environ.get("CUDA_HOME"), "/usr/local/cuda",
                        "/opt/cuda") if r]
        for d in dirs:
            for n in self.NAMES:
                try:
                    self.lib = ctypes.CDLL(d + n)
                    break
                except OSError as e:
                    err = e
            if self.lib:
                break
        if not self.lib:
            raise RuntimeError("libnvrtc not found (%s)" % err)
        major, minor = ctypes.c_int(), ctypes.c_int()
        self.lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        self.version = "%d.%d" % (major.value, minor.value)

    def ptx(self, src, opts):
        prog = ctypes.c_void_p()
        L = self.lib
        L.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"probe.cu",
                             0, None, None)
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        rc = L.nvrtcCompileProgram(prog, len(opts), arr)
        if rc != 0:
            size = ctypes.c_size_t()
            L.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            L.nvrtcGetProgramLog(prog, log)
            L.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(log.value.decode(errors="replace")[:400])
        size = ctypes.c_size_t()
        L.nvrtcGetPTXSize(prog, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        L.nvrtcGetPTX(prog, buf)
        L.nvrtcDestroyProgram(ctypes.byref(prog))
        return buf.value.decode()


class NvidiaOpenCL:
    """NVIDIA's OpenCL compiler, which only exists inside the driver."""

    def __init__(self):
        c = ctypes
        self.cl = cl = c.CDLL("libOpenCL.so.1")
        cl.clCreateContext.restype = c.c_void_p
        cl.clCreateProgramWithSource.restype = c.c_void_p
        plats, n = (c.c_void_p * 16)(), c.c_uint()
        cl.clGetPlatformIDs(16, plats, c.byref(n))
        self.dev = None
        for p in plats[:n.value]:
            name = c.create_string_buffer(256)
            cl.clGetPlatformInfo(c.c_void_p(p), 0x0902, 256, name, None)
            if b"NVIDIA" not in name.value:
                continue
            devs, nd = (c.c_void_p * 8)(), c.c_uint()
            if cl.clGetDeviceIDs(c.c_void_p(p), c.c_ulonglong(1 << 2), 8, devs,
                                 c.byref(nd)) == 0 and nd.value:
                self.dev = c.c_void_p(devs[0])
                break
        if self.dev is None:
            raise RuntimeError("no NVIDIA OpenCL device")
        err = c.c_int()
        self.ctx = c.c_void_p(cl.clCreateContext(None, 1, c.byref(self.dev),
                                                 None, None, c.byref(err)))
        drv = c.create_string_buffer(64)
        cl.clGetDeviceInfo(self.dev, 0x102D, 64, drv, None)
        self.version = "driver " + drv.value.decode()

    def ptx(self, src, opts):
        c, cl = ctypes, self.cl
        err = c.c_int()
        s = c.c_char_p(src.encode())
        prog = c.c_void_p(cl.clCreateProgramWithSource(self.ctx, 1, c.byref(s),
                                                       None, c.byref(err)))
        if cl.clBuildProgram(prog, 1, c.byref(self.dev),
                             " ".join(opts).encode(), None, None) != 0:
            log = c.create_string_buffer(8192)
            cl.clGetProgramBuildInfo(prog, self.dev, 0x1183, 8192, log, None)
            cl.clReleaseProgram(prog)
            raise RuntimeError(log.value.decode(errors="replace")[:400])
        size = c.c_size_t()
        cl.clGetProgramInfo(prog, 0x1165, c.sizeof(size), c.byref(size), None)
        buf = c.create_string_buffer(size.value)
        ptrs = (c.c_char_p * 1)(c.cast(buf, c.c_char_p))
        cl.clGetProgramInfo(prog, 0x1166, c.sizeof(ptrs), ptrs, None)
        cl.clReleaseProgram(prog)
        return buf.raw.rstrip(b"\0").decode()


def assemble(ptxas, cuobjdump, ptx, sm, tmp, want_regs=False):
    src = os.path.join(tmp, "p.ptx")
    obj = os.path.join(tmp, "p.cubin")
    open(src, "w").write(ptx)
    p = subprocess.run([ptxas, "-arch=sm_%s" % sm, "-v", "-o", obj, src],
                       check=True, capture_output=True, text=True)
    ops = sass_ops(cuobjdump, obj)
    if not want_regs:
        return ops
    m = re.search(r"Used (\d+) registers", p.stderr)
    sp = re.search(r"(\d+) bytes spill stores", p.stderr)
    return ops, (int(m.group(1)) if m else None), (int(sp.group(1)) if sp else 0)


# The steerable primitives and the spellings each toolchain can compile.
KERNEL_STEERS = (("rotl32", "VB_STEER_ROTL32"), ("rotr64", "VB_STEER_ROTR64"),
                 ("ch", "VB_STEER_CH"), ("maj", "VB_STEER_MAJ"))


def kernel_rows(tool, target, dialect, build, spellings_ok):
    """Each real kernel all-plain, then with one primitive steered at a time."""
    rows = []
    for alg in ("md5", "sha1", "sha512"):
        core = open(os.path.join(GPU, "%s_device_impl.h" % alg)).read()
        src = dialect + open(os.path.join(GPU, "device_primitives.h")).read() + core
        for streams in (1, 2, 3, 4):
            base = ["-DLANES=64", "-DSTREAMS=%d" % streams]
            try:
                ops, regs, spill = build(src, base)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                rows.append({"toolchain": tool, "target": target,
                             "kernel": "%s-s%d" % (alg, streams),
                             "error": str(e)[:160]})
                continue
            row = {"toolchain": tool, "target": target,
                   "kernel": "%s-s%d" % (alg, streams),
                   "plain": {"insns": len(ops), "regs": regs, "spill": spill},
                   "steered": {}}
            for prim, define in KERNEL_STEERS:
                for sp in spellings_ok:
                    if sp == PLAIN:
                        continue
                    try:
                        o, r, spl = build(src, base + ["-D%s=%d" % (define, sp)])
                    except (RuntimeError, subprocess.CalledProcessError):
                        continue
                    row["steered"]["%s=%s" % (prim, SPELLING[sp])] = {
                        "insns": len(o), "regs": r, "spill": spl}
            rows.append(row)
    return rows


def run_kernels(args):
    dialect_cl = open(os.path.join(GPU, "dialect_opencl.h")).read()
    dialect_cu = open(os.path.join(GPU, "dialect_cuda.h")).read()
    rows, notes = [], []
    ptxas, cuobjdump = find_tool("ptxas"), find_tool("cuobjdump")
    if not (ptxas and cuobjdump):
        return rows, ["kernel mode needs ptxas and cuobjdump"]
    with tempfile.TemporaryDirectory() as tmp:
        if "nvrtc" in args["only"]:
            try:
                nv = Nvrtc()
                for sm in args["sm"]:
                    rows += kernel_rows(
                        "nvrtc %s" % nv.version, "sm_%s" % sm, dialect_cu,
                        lambda s, d, sm=sm: assemble(
                            ptxas, cuobjdump,
                            nv.ptx(s, ["-arch=compute_%s" % sm] + d), sm, tmp,
                            True),
                        (PLAIN, PTX))
            except RuntimeError as e:
                notes.append("nvrtc skipped: %s" % e)
        if "opencl" in args["only"]:
            try:
                ocl = NvidiaOpenCL()
                probe = ocl.ptx(dialect_cl + "__kernel void k(__global uint *a){a[0]=1;}",
                                ["-cl-std=CL1.2"])
                m = re.search(r"\.target\s+sm_(\d+)", probe)
                sm = m.group(1) if m else "120"
                rows += kernel_rows(
                    "nvidia-opencl %s" % ocl.version, "sm_%s" % sm, dialect_cl,
                    lambda s, d: assemble(ptxas, cuobjdump,
                                          ocl.ptx(s, ["-cl-std=CL1.2"] + d),
                                          sm, tmp, True),
                    (PLAIN, CLB, PTX))
            except (RuntimeError, OSError) as e:
                notes.append("nvidia opencl skipped: %s" % e)
    return rows, notes


def run(args):
    dialect_cl = open(os.path.join(GPU, "dialect_opencl.h")).read()
    dialect_cu = open(os.path.join(GPU, "dialect_cuda.h")).read()
    rows, notes = [], []
    ptxas, cuobjdump = find_tool("ptxas"), find_tool("cuobjdump")
    want = set(args["only"])

    def each(tool, target, dialect, compile_ops, spellings_ok, lane):
        for name, word, define, step, spellings in primitives():
            for sp in spellings:
                if sp not in spellings_ok:
                    continue
                defs = ["-D%s=%d" % (define, sp)] if define and sp else []
                if sp and not define:
                    continue
                try:
                    with_ops = compile_ops(probe_source(dialect, word, step, True),
                                           defs + [lane])
                    base_ops = compile_ops(probe_source(dialect, word, step, False),
                                           defs + [lane])
                except (RuntimeError, subprocess.CalledProcessError) as e:
                    msg = getattr(e, "stderr", None) or str(e)
                    if isinstance(msg, bytes):
                        msg = msg.decode(errors="replace")
                    rows.append({"toolchain": tool, "target": target,
                                 "primitive": name, "spelling": SPELLING[sp],
                                 "error": msg.strip().splitlines()[0][:160]})
                    continue
                per, mix = delta(with_ops, base_ops)
                rows.append({"toolchain": tool, "target": target,
                             "primitive": name, "spelling": SPELLING[sp],
                             "per_use": per, "mix": mix})

    with tempfile.TemporaryDirectory() as tmp:
        if "nvrtc" in want:
            try:
                if not (ptxas and cuobjdump):
                    raise RuntimeError("ptxas or cuobjdump not found")
                nv = Nvrtc()
                for sm in args["sm"]:
                    tool = "nvrtc %s" % nv.version
                    each(tool, "sm_%s" % sm, dialect_cu,
                         lambda s, d, sm=sm: assemble(
                             ptxas, cuobjdump,
                             nv.ptx(s, ["-arch=compute_%s" % sm] + d), sm, tmp),
                         (PLAIN, PTX), "-DVB_PROBE_LANE=threadIdx.x")
            except RuntimeError as e:
                notes.append("nvrtc skipped: %s" % e)
        if "opencl" in want:
            try:
                if not (ptxas and cuobjdump):
                    raise RuntimeError("ptxas or cuobjdump not found")
                ocl = NvidiaOpenCL()
                sm = None
                probe = ocl.ptx(dialect_cl + "__kernel void k(__global uint *a){a[0]=1;}",
                                ["-cl-std=CL1.2"])
                m = re.search(r"\.target\s+sm_(\d+)", probe)
                sm = m.group(1) if m else "120"
                each("nvidia-opencl %s" % ocl.version, "sm_%s" % sm, dialect_cl,
                     lambda s, d: assemble(ptxas, cuobjdump,
                                           ocl.ptx(s, ["-cl-std=CL1.2"] + d),
                                           sm, tmp),
                     (PLAIN, CLB, PTX), "-DVB_PROBE_LANE=get_local_id(0)")
            except (RuntimeError, OSError) as e:
                notes.append("nvidia opencl skipped: %s" % e)
        if "amdgpu" in want:
            clang = shutil.which("clang")
            if not clang:
                notes.append("amdgpu skipped: no clang")
            else:
                ver = subprocess.run([clang, "--version"], capture_output=True,
                                     text=True).stdout.splitlines()[0]
                for gfx in args["gfx"]:
                    def gcn(s, d, gfx=gfx):
                        p = subprocess.run(
                            [clang, "-x", "cl", "-cl-std=CL1.2",
                             "-target", "amdgcn-amd-amdhsa", "-mcpu=" + gfx,
                             "-nogpulib", "-O3", "-S", "-o", "-", "-"] + d,
                            input=s, capture_output=True, text=True)
                        if p.returncode != 0:
                            raise RuntimeError(p.stderr)
                        return gcn_ops(p.stdout)
                    # get_local_id is a device-library call without ROCm's
                    # libraries; the builtin is what it lowers to.
                    each("clang " + ver.split("version")[-1].strip().split()[0],
                         gfx, dialect_cl, gcn, (PLAIN, AMDGCN),
                         "-DVB_PROBE_LANE=__builtin_amdgcn_workitem_id_x()")
    return rows, notes


def main():
    argv = sys.argv[1:]
    args = {"sm": ["75", "80", "86", "89", "90", "120"],
            "gfx": ["gfx906", "gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1201"],
            "only": ["nvrtc", "opencl", "amdgpu"]}
    as_json = "--json" in argv
    for flag in ("--sm", "--gfx", "--only"):
        if flag in argv:
            args[flag[2:]] = argv[argv.index(flag) + 1].split(",")
    if "--kernels" in argv:
        rows, notes = run_kernels(args)
        if as_json:
            json.dump({"schema": "valubench/idiom-probe-kernels/1",
                       "rows": rows, "notes": notes}, sys.stdout, indent=1)
            print()
            return 0
        for n in notes:
            print("note: " + n)
        print("instructions (registers/spill bytes), all-plain, then each single "
              "steer as a change from it")
        for r in rows:
            if "error" in r:
                print("%-30s %-7s %-10s error %s" % (r["toolchain"], r["target"],
                                                    r["kernel"], r["error"]))
                continue
            p = r["plain"]
            steer = "  ".join("%s %+d (%s/%s)" % (k, v["insns"] - p["insns"],
                                                 v["regs"], v["spill"])
                              for k, v in r["steered"].items())
            print("%-30s %-7s %-10s %6d (%s/%s)  %s"
                  % (r["toolchain"], r["target"], r["kernel"], p["insns"],
                     p["regs"], p["spill"], steer))
        return 0
    rows, notes = run(args)
    if as_json:
        json.dump({"schema": "valubench/idiom-probe/1", "uses_per_chain": N,
                   "rows": rows, "notes": notes}, sys.stdout, indent=1)
        print()
        return 0
    for n in notes:
        print("note: " + n)
    print("%-28s %-8s %-8s %-15s %8s  %s"
          % ("toolchain", "target", "prim", "spelling", "insn/use", "mix per chain"))
    for r in rows:
        if "error" in r:
            print("%-28s %-8s %-8s %-15s %8s  %s" % (r["toolchain"], r["target"],
                  r["primitive"], r["spelling"], "error", r["error"]))
            continue
        mix = " ".join("%s%+d" % (k, v) for k, v in
                       sorted(r["mix"].items(), key=lambda kv: -abs(kv[1]))[:5])
        print("%-28s %-8s %-8s %-15s %8.2f  %s" % (r["toolchain"], r["target"],
              r["primitive"], r["spelling"], r["per_use"], mix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
