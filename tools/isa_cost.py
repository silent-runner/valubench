#!/usr/bin/env python3
"""Instructions per message, from the generated code.

Static and deterministic, so it costs nothing and cannot be perturbed by load.

DO NOT READ THIS AS A THROUGHPUT PREDICTION. It was used as one, and it was
wrong on both ARM parts:

  Graviton3, SVE-256 vs NEON-128   predicted 1.33x-1.68x   measured 0.52x-1.17x
  Graviton4, SVE2-128 vs NEON-128  predicted 0.91x-1.03x   measured 0.32x-0.91x

Two reasons, both measured on Graviton4 with perf. First, SVE code retires
about half the instructions per cycle that NEON does -- 1.56 against 3.18 --
at one micro-op per instruction either way, so it is an issue-rate limit that
no instruction count can see. Second, comparing each ISA at *its own* best
stream count compares the wrong pairs: the stream count that minimises
instructions is not the one that maximises throughput.

Throughput ratio decomposes as (instruction ratio) x (IPC ratio), and this tool
supplies only the first. On Graviton4 that product came within 2% of measured.
So: use this to understand where instructions go, and to compare an ISA against
itself across a change. To predict throughput, measure IPC too.

A kernel body processes one group of (lanes x streams) messages, so
instructions per message is body / (lanes x streams). For a vector-length
agnostic ISA the body does not change with the vector length -- the same
instructions simply cover more lanes -- so the lane count is supplied rather
than read from the object.

Usage: isa_cost.py [--no-movprfx] [objdir [objdump]]

objdir defaults to build-arm64 and objdump to aarch64-linux-gnu-objdump, the
cross build; for a native AArch64 or Mac build pass build and objdump. The
stream counts are whatever the objects hold, and a kernel the symbol table
lists that the disassembly never yields is an error, not a gap in the table.
"""
import argparse, os, re, subprocess, sys

# lanes per vector for 32-bit and 64-bit words, at the width being modelled
WIDTH = {
    "neon": (4, 2, "128-bit, fixed"),
    "sve":  (8, 4, "256-bit, Neoverse V1"),
    "sve2": (4, 2, "128-bit, Neoverse V2"),
}

# A kernel's symbol; Mach-O prefixes every C symbol with an underscore.
KERNEL = re.compile(r"^_?vb_(md5|sha1|sha512)_(neon|sve|sve2)_s(\d+)$")


def fail(msg):
    sys.stderr.write("isa_cost: %s\n" % msg)
    sys.exit(1)


def objdump(tool, obj, *flags):
    try:
        p = subprocess.run([tool] + list(flags) + [obj],
                           capture_output=True, text=True)
    except FileNotFoundError:
        fail("%s not found; pass the toolchain's own objdump" % tool)
    if p.returncode != 0:
        fail("%s %s %s: %s" % (tool, " ".join(flags), obj, p.stderr.strip()))
    return p.stdout


def kernels(tool, obj):
    """(section, address) -> (alg, isa, streams) for every kernel symbol.

    By address as well as name because the disassembly labels a function with
    whichever symbol it picks there, which on Mach-O is ltmp0 for the first one
    in a section; addresses in an object are section offsets, hence the section.
    """
    found = {}
    for ln in objdump(tool, obj, "-t").splitlines():
        f = ln.split()
        if "F" not in f[1:-1]:
            continue
        m = KERNEL.match(f[-1])
        if m:
            sec = f[f.index("F", 1) + 1]
            found[(sec, f[0].lstrip("0"))] = (m.group(1), m.group(2), int(m.group(3)))
    return found


# movprfx exists because most SVE data instructions are destructive: it copies
# a register so the following instruction can overwrite it. Neoverse is
# documented to fuse the pair at rename, so counting it inflates SVE's apparent
# cost against NEON's three-operand encoding. Both figures are reported: the
# truth is somewhere between, and which end depends on the core.
def body_counts(tool, obj, drop_movprfx=False):
    syms = kernels(tool, obj)
    res, sec, cur = {}, "", None
    for ln in objdump(tool, obj, "-d", "--no-show-raw-insn").splitlines():
        m = re.match(r"^Disassembly of section (.+):$", ln)
        if m:
            sec, cur = m.group(1), None
            continue
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", ln)
        if m:
            # GCC emits internal labels inside SVE functions (.SVLPSPL0 and
            # friends, for the lazy-save prologue). They are not function
            # boundaries, and treating them as such truncated every SVE body
            # to its first fourteen instructions.
            if m.group(2).startswith("."):
                continue
            cur = syms.get((sec, m.group(1).lstrip("0")))
            if cur is None:
                k = KERNEL.match(m.group(2))
                cur = (k.group(1), k.group(2), int(k.group(3))) if k else None
            if cur:
                res.setdefault(cur, 0)
            continue
        if cur and re.match(r"^\s+[0-9a-f]+:", ln):
            if drop_movprfx and "movprfx" in ln:
                continue
            res[cur] += 1
    unread = sorted(set(syms.values()) - set(res))
    if unread:
        fail("%s: %s listed in the symbol table but never read from the "
             "disassembly" % (obj, ", ".join("%s/%s-s%d" % k for k in unread)))
    return res


def main():
    ap = argparse.ArgumentParser(
        description="Instructions per message for the NEON, SVE and SVE2 "
                    "kernels, from the generated code.",
        epilog="Instruction counts only: not a throughput prediction. See the "
               "module docstring for why.")
    ap.add_argument("objdir", nargs="?", default="build-arm64",
                    help="where the kernel_<isa>.o objects are (default: build-arm64)")
    ap.add_argument("objdump", nargs="?", default="aarch64-linux-gnu-objdump",
                    help="the objdump for their target (default: aarch64-linux-gnu-objdump)")
    ap.add_argument("--no-movprfx", action="store_true",
                    help="do not count movprfx, as if fused at rename")
    a = ap.parse_args()

    counts = {}
    for isa in WIDTH:
        obj = os.path.join(a.objdir, "kernel_%s.o" % isa)
        if os.path.exists(obj):       # a build need not have every ISA
            counts.update(body_counts(a.objdump, obj, a.no_movprfx))
    if not counts:
        fail("no NEON, SVE or SVE2 kernels under %s" % a.objdir)
    streams = sorted({s for (_, _, s) in counts})

    print("counting movprfx: %s\n" % ("no (assumed fused at rename)" if a.no_movprfx else "yes"))
    print("Instructions per message, from the generated code\n")
    for alg, w in (("md5", 32), ("sha1", 32), ("sha512", 64)):
        print("  %s" % alg)
        print("    %-10s %-22s %s" % ("isa", "width",
              " ".join("%9s" % ("s%d" % st) for st in streams)))
        per = {}
        for isa in ("neon", "sve", "sve2"):
            if not any(k[1] == isa for k in counts):
                continue
            l32, l64, label = WIDTH[isa]
            lanes = l32 if w == 32 else l64
            row = []
            for st in streams:
                n = counts.get((alg, isa, st))
                row.append(n / (lanes * st) if n else None)
            per[isa] = row
            print("    %-10s %-22s %s" % (isa, label,
                  " ".join("%9.1f" % v if v else "        -" for v in row)))
        # Matched stream counts only. Comparing each ISA at its own cheapest
        # stream count pairs configurations that no run would ever choose.
        for other, label in (("sve", "SVE  vs NEON"), ("sve2", "SVE2 vs NEON")):
            if "neon" not in per or other not in per:
                continue
            cells = []
            for i, st in enumerate(streams):
                x, y = per["neon"][i], per[other][i]
                cells.append("s%d %.2fx" % (st, x / y) if x and y else "s%d   -" % st)
            print("    %s, instructions only: %s" % (label, "  ".join(cells)))
        print("    (instructions only -- multiply by the IPC ratio for throughput)")
        print()


main()
