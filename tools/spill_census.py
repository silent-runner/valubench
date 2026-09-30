#!/usr/bin/env python3
"""Count stack traffic per stream in the scalar kernels.

The question: what sets the best stream count on a given core?
Live state per stream orders the algorithms identically on x86-64 and AArch64 --
MD5 four, SHA-1 one to two, SHA-512 one -- but does not predict the peak across
machines. AArch64 has 31 general-purpose registers to x86-64's 16, yet
sha1/scalar peaks at s1 on Neoverse V1 and s2 on Zen 5, the opposite of what
register count implies.

If register pressure is the mechanism, spill traffic should rise where
throughput falls off. This counts it: every load and store whose address is
formed from the stack pointer or frame pointer inside a kernel body is a spill
or a reload, because the message words come from a pointer argument and the
digest goes out through another one -- nothing else legitimately lives on the
stack in these functions.

Usage:
    tools/spill_census.py [--objdump OBJDUMP] [OBJECT ...]

OBJECT defaults to build/kernel_scalar.o and OBJDUMP to objdump. Pass the
toolchain's own objdump for a cross build (aarch64-linux-gnu-objdump for
build-arm64/kernel_scalar.o), and run once per object when they need different
objdumps. The architecture is read from the object, not assumed.

What it counts comes from the object as well: every scalar kernel in the
symbol table, at every stream count the build has. It once looked for s1 to s4
by name, and so could not see s6 and s8, and a kernel it failed to find was
skipped -- on a Mac, where symbols carry a leading underscore and the first
function hides behind the section-start label ltmp0, it read nothing and
printed {} with a success status. A kernel the symbol table lists and the
disassembly never yields is now an error.
"""
import argparse, json, re, subprocess, sys

# x86-64 AT&T: an operand like 0x38(%rsp) or -0x20(%rbp).
# Instructions that can fold a memory operand into the arithmetic. Not
# exhaustive by design: these are the forms that appear in the round bodies.
X86_RMW = ('add', 'sub', 'and', 'or', 'xor', 'cmp', 'test',
           'adc', 'sbb', 'inc', 'dec', 'imul', 'lea', 'rol', 'ror')

X86_MEM = re.compile(r'-?0x[0-9a-f]+\((%rsp|%rbp)\)|\((%rsp|%rbp)\)')
# AArch64: [sp, #96] / [x29, #-16] / [sp]
ARM_MEM = re.compile(r'\[(sp|x29)(,|\])')

# A scalar kernel's symbol; Mach-O prefixes every C symbol with an underscore.
KERNEL = re.compile(r'^_?vb_(md5|sha1|sha512)_scalar_s(\d+)$')
ALGS = ('md5', 'sha1', 'sha512')


def fail(msg):
    sys.stderr.write('spill_census: %s\n' % msg)
    sys.exit(1)


def objdump(tool, obj, *flags):
    try:
        p = subprocess.run([tool] + list(flags) + [obj],
                           capture_output=True, text=True)
    except FileNotFoundError:
        fail('%s not found; pass the toolchain\'s own with --objdump' % tool)
    if p.returncode != 0:
        fail('%s %s %s: %s' % (tool, ' '.join(flags), obj, p.stderr.strip()))
    return p.stdout


def architecture(listing, obj):
    # "file format elf64-x86-64", "elf64-littleaarch64", "mach-o arm64" ...
    m = re.search(r'file format (.+)$', listing, re.M)
    fmt = m.group(1).lower() if m else ''
    if 'x86-64' in fmt or 'x86_64' in fmt:
        return 'x86'
    if 'aarch64' in fmt or 'arm64' in fmt:
        return 'arm'
    fail('%s: no x86-64 or AArch64 rules for file format %r' % (obj, fmt))


def kernels(listing):
    """(section, address) -> (alg, streams) for every scalar kernel symbol.

    Keyed by address as well as name because the disassembly labels a function
    with whichever symbol it picks at that address, which on Mach-O is ltmp0
    for the first one in a section. Addresses in an object are section offsets,
    hence the section.
    """
    found = {}
    for ln in listing.splitlines():
        f = ln.split()
        if 'F' not in f[1:-1]:
            continue
        m = KERNEL.match(f[-1])
        if m:
            sec = f[f.index('F', 1) + 1]
            found[(sec, f[0].lstrip('0'))] = (m.group(1), int(m.group(2)))
    return found


def bodies(listing, syms):
    """Instruction text per kernel, the label resolved through the symbols."""
    out, sec, on = {}, '', None
    for ln in listing.splitlines():
        m = re.match(r'^Disassembly of section (.+):$', ln)
        if m:
            sec, on = m.group(1), None
            continue
        m = re.match(r'^([0-9a-f]+) <(.+)>:$', ln)
        if m:
            on = syms.get((sec, m.group(1).lstrip('0')))
            if on is None:
                k = KERNEL.match(m.group(2))
                on = (k.group(1), int(k.group(2))) if k else None
            continue
        if on and re.match(r'^\s+[0-9a-f]+:', ln):
            out.setdefault(on, []).append(
                ln.split('\t', 1)[-1].strip() if '\t' in ln else ln.strip())
    return out


def analyse(arch, lines):
    total = len(lines)
    ld = st = 0
    frame = 0
    for ln in lines:
        parts = ln.split(None, 1)
        if not parts:
            continue
        op, rest = parts[0], (parts[1] if len(parts) > 1 else '')
        rest_full = ln
        if arch == 'x86':
            # Frame size from the prologue's sub $N,%rsp -- which LLVM's
            # objdump spells subq $N, %rsp.
            m = re.match(r'subq?\s+\$(0x[0-9a-f]+),\s*%rsp', ln)
            if m and not frame:
                frame = int(m.group(1), 16)
            if not X86_MEM.search(rest):
                continue
            # AT&T: destination last. Memory on the right is a store.
            src, _, dst = rest.rpartition(',')
            if op.startswith(('mov', 'movq', 'movl')):
                if X86_MEM.search(dst):
                    st += 1
                elif X86_MEM.search(src):
                    ld += 1
            elif op.startswith(X86_RMW):
                # A folded memory operand: `addl 0x18(%rsp),%eax` reads the
                # stack slot as part of the arithmetic, no separate mov. This
                # is the form x86 reaches for under register pressure, which
                # is exactly the case MD5 creates -- and counting only mov*
                # made those reloads invisible while AArch64, which has no
                # folded form and must emit an ldr, counted every one. The
                # comparison this tool exists to make was biased against
                # AArch64 by however much the folding saved.
                if X86_MEM.search(dst):
                    st += 1          # read-modify-write touches the slot twice
                    ld += 1
                elif X86_MEM.search(src):
                    ld += 1
        else:
            m = re.match(r'sub\s+sp, sp, #(0x[0-9a-f]+|\d+)', ln)
            if m and not frame:
                frame = int(m.group(1), 16) if m.group(1).startswith('0x') else int(m.group(1))
            if not ARM_MEM.search(rest):
                continue
            # Skip the ABI prologue and epilogue. AArch64 saves callee-saved
            # registers with stp/ldp through memory, where x86-64 uses
            # push/pop -- which carry no memory operand and so were never
            # counted on that side. Counting them here would have made every
            # ARM figure look worse by a fixed amount that has nothing to do
            # with register pressure in the round body.
            if re.match(r'(ld|st)p\s+(x(19|2[0-8])|x29), (x(19|2[0-8])|x30),', rest_full):
                continue
            if op.startswith(('ldr', 'ldp', 'ldur')):
                ld += 2 if op.startswith('ldp') else 1
            elif op.startswith(('str', 'stp', 'stur')):
                st += 2 if op.startswith('stp') else 1
    return dict(insns=total, loads=ld, stores=st, stack=ld + st, frame=frame)


def census(tool, obj):
    syms = kernels(objdump(tool, obj, '-t'))
    listing = objdump(tool, obj, '-d', '--no-show-raw-insn')
    arch = architecture(listing, obj)
    if not syms:
        fail('%s: no scalar kernels in the symbol table' % obj)
    got = bodies(listing, syms)
    unread = sorted(set(syms.values()) - set(got))
    if unread:
        fail('%s: %s listed in the symbol table but never read from the '
             'disassembly' % (obj, ', '.join('%s/scalar-s%d' % k for k in unread)))
    return arch, {k: analyse(arch, got[k]) for k in sorted(
        got, key=lambda k: (ALGS.index(k[0]), k[1]))}


def main():
    ap = argparse.ArgumentParser(
        description='Count stack loads and stores in each scalar kernel.',
        epilog='Prints JSON keyed <arch>/<alg>/s<streams>. Exits 1 if an '
               'object holds no scalar kernels, or one it lists cannot be read.')
    ap.add_argument('objects', nargs='*', default=['build/kernel_scalar.o'],
                    metavar='OBJECT', help='default: build/kernel_scalar.o')
    ap.add_argument('--objdump', default='objdump',
                    help="the objdump for these objects' target (default: objdump)")
    a = ap.parse_args()

    res = {}
    for obj in a.objects:
        arch, counts = census(a.objdump, obj)
        for (alg, s), v in counts.items():
            key = '%s/%s/s%d' % (arch, alg, s)
            if key in res:
                fail('%s: a second %s object; run once per architecture' % (obj, arch))
            res[key] = v
    print(json.dumps(res, indent=1))


main()
