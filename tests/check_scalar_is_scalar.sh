#!/bin/sh
#
# check_scalar_is_scalar.sh -- the scalar kernel must contain no vector code.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, The valubench authors. See LICENSE.
#
# The scalar rung is the denominator of every ISA ratio this project reports,
# so it has to be scalar. It silently was not: at -O2 gcc fuses the independent
# streams with SLP vectorisation and emits SSE2 on x86-64 and NEON on AArch64 --
# 88% and 79% of instructions respectively in the two-stream kernel, using two
# of four lanes. Clang does not do it at all, so the same source produced a
# different baseline depending on the compiler, which is worse than either
# behaviour on its own.
#
# The Makefile builds this translation unit with -fno-tree-vectorize and
# -fno-tree-slp-vectorize. This checks the result rather than trusting the flag,
# because a compiler that stops honouring the spelling would put the old
# behaviour back with no other symptom than numbers that quietly improve.
#
# A guard that reads disassembly is only as good as its idea of what the
# disassembler prints, and that is per platform. So it also runs as its own
# control: with --control, the named functions are kernels known to be vector
# code, and each must read as mostly vector. Apple's objdump is why -- it
# printed NEON in a syntax the register patterns did not match, and the guard
# passed on macOS while seeing little but loads and stores.
#
# Usage: check_scalar_is_scalar.sh <object-file> [objdump]
#        check_scalar_is_scalar.sh --control <isa> <object-file> [objdump]

set -eu

WANT=scalar
CONTROL=0
if [ "${1:-}" = "--control" ]; then
    WANT=${2:?usage: $0 --control <isa> <object-file> [objdump]}
    CONTROL=1
    shift 2
fi

OBJ=${1:?usage: $0 <object-file> [objdump]}
OBJDUMP=${2:-objdump}

if ! command -v "$OBJDUMP" >/dev/null 2>&1; then
    echo "  skip  scalar-purity  ($OBJDUMP not found)"
    exit 0
fi
[ -f "$OBJ" ] || { echo "  skip  scalar-purity  ($OBJ not built)"; exit 0; }

# The symbol table first, then the disassembly, in one stream. The table says
# which functions exist, so a function the disassembly never names is caught
# rather than skipped: in a Mach-O object the first function shares its
# address with the section-start label ltmp0, objdump names it by that, and
# md5/scalar-s1 went uninspected on every Mac.
{
    "$OBJDUMP" -t "$OBJ"
    echo "@@ disassembly"
    "$OBJDUMP" -d --no-show-raw-insn "$OBJ"
} | awk -v want="$WANT" -v control="$CONTROL" -v objdump="$OBJDUMP" '
    /^@@ disassembly$/ { dis = 1; next }

    # Function symbols: address, flags, "F", section, [size,] name.
    !dis {
        for (i = 2; i < NF; i++)
            if ($i == "F")
                break
        if (i >= NF || $NF !~ want)
            next
        a = $1; sub(/^0+/, "", a)
        sym[$(i + 1), a] = $NF
        listed[$NF] = 1
        next
    }

    /^Disassembly of section / { sec = $4; sub(/:$/, "", sec); next }

    # A label objdump chose from several at one address is resolved through
    # the symbol table, keyed by section as well, since object addresses are
    # section offsets.
    /^[0-9a-f]+ </ {
        fn = $2; gsub(/[<>:]/, "", fn)
        a = $1; sub(/^0+/, "", a)
        if (fn !~ want && ((sec, a) in sym))
            fn = sym[sec, a]
        next
    }

    # Vector registers: x86 %xmm/%ymm/%zmm; AArch64 q0, and v0 in either
    # syntax -- generic (v0.4s, v0.s[1]) or Apple (add.4s v0, v1, v2 and
    # mov.s v0[1], w8), which is what objdump prints for a Darwin target. A
    # handful of hits is normal -- the prologue zeroes a register and the ABI
    # moves things around -- so the test is a proportion, not a count.
    /^[ \t]+[0-9a-f]+:/ {
        if (fn !~ want) next
        total[fn]++
        if ($0 ~ /%[xyz]mm[0-9]/ || $0 ~ /[ \t,]q[0-9]+/ ||
            $0 ~ /[ \t,{]v[0-9]+([.,} \t]|$)/ || $0 ~ /[ \t,{]v[0-9]+\[/)
            vec[fn]++
    }

    END {
        n = 0; for (f in total) n++
        # A guard that cannot find its input must fail, not pass. This printed
        # "ok scalar-purity (0 kernels, no vector instructions)" on a Graviton4
        # where the build had gone to build-gcc13 rather than build, so the
        # object held no scalar symbols -- and the check that exists to stop
        # the ratio denominator silently becoming vector code silently checked
        # nothing at all.
        if (n == 0) {
            printf "  FAIL  scalar-purity  no %s kernels found in the object\n",
                   want
            print "        Nothing was checked. Either the object is missing or"
            print "        the symbol naming changed; either way this guard was"
            print "        not doing its job."
            exit 1
        }
        missed = 0
        for (f in listed)
            if (!(f in total)) {
                printf "  FAIL  %-24s in the symbol table, never inspected\n", f
                missed = 1
            }
        if (missed) {
            print ""
            printf "  %s labelled these functions with names this check\n", objdump
            print "  could not resolve, so their instructions were not read."
            exit 1
        }

        bad = 0
        for (f in total) {
            pct = total[f] ? 100.0 * vec[f] / total[f] : 0
            if (control ? pct < 50.0 : pct > 5.0) {
                printf "  FAIL  %-24s %5.1f%% vector instructions (%d of %d)\n",
                       f, pct, vec[f], total[f]
                bad = 1
            }
        }
        if (bad && control) {
            print ""
            printf "  These are %s kernels, so this disassembly should read as\n", want
            print "  mostly vector code. The register patterns above do not match"
            printf "  what %s prints, and the scalar-purity result\n", objdump
            print "  cannot be trusted until they do."
            exit 1
        }
        if (bad) {
            print ""
            print "  The scalar kernel was vectorised by the compiler. Every ISA"
            print "  ratio divides by this kernel, so a vector scalar rung makes"
            print "  those ratios meaningless. Check KFLAGS_scalar in the Makefile."
            exit 1
        }
        if (control)
            printf "  ok    scalar-purity  control (%d %s kernels read as vector)\n",
                   n, want
        else
            printf "  ok    scalar-purity  (%d kernels, no vector instructions)\n", n
    }'
