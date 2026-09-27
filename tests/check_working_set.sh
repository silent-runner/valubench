#!/bin/sh
#
# check_working_set.sh -- corpus size accounting, per algorithm.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, The valubench authors. See LICENSE.
#
# The reported working set must equal the corpus actually built:
#
#     batch_messages x blocks_per_message x block_bytes
#
# It did not. The result path hardcoded 64 bytes per block, which is right for
# MD5 and SHA-1 and half the truth for SHA-512, so every SHA-512 row in every
# capture understated its corpus by a factor of two -- in the JSON, and in the
# CSV column that sweep.py derives from it. Working set is one of the three axes
# this benchmark sweeps, and the number decides which side of a cache a
# measurement is read as sitting on, so a factor of two is not cosmetic.
#
# The bug was possible because two places computed the same quantity: a helper
# from the config, correctly, and the result path from the corpus, wrongly. The
# helper had no callers, so nothing ever compared them. It is gone, and this
# checks what remains against the JSON it emits, for every algorithm, because
# the block size is the thing that varies.
#
# Usage: check_working_set.sh <binary>

set -eu

BIN=${1:?usage: $0 <binary>}
[ -x "$BIN" ] || { echo "  skip  working-set  ($BIN not built)"; exit 0; }

# Message lengths either side of each algorithm's padding boundary, where the
# block count changes: 55/56 for the 64-byte block algorithms, 111/112 for
# SHA-512's 128-byte one. A block-size assumption shows up as a wrong block
# count as readily as a wrong byte count.
CASES="md5:55 md5:56 md5:247 sha1:55 sha1:56 sha512:55 sha512:111 sha512:112 sha512:255"

fail=0
for case in $CASES; do
    alg=${case%:*}
    mb=${case#*:}
    # Exit 3 is "too noisy to trust", which short samples on a busy machine
    # earn routinely. This test reads accounting, not throughput, so the
    # measurement being noisy is irrelevant -- only a missing or unparseable
    # document is a failure. Exit 1 (verification failed) is a real failure and
    # would leave no usable JSON either.
    out=$("$BIN" --algorithm "$alg" --message-bytes "$mb" --working-set-kb 1024 \
                 --samples 2 --time-ms 40 --warmup-ms 40 --json 2>/dev/null) &&
        rc=0 || rc=$?
    if [ "$rc" = 4 ]; then
        printf '  FAIL  working-set  %s/%s: exit 4, could not run -- the runner\n' \
               "$alg" "$mb"
        printf '        cannot run a 1 MiB CPU measurement, which is the runner at fault\n'
        fail=1; continue
    fi
    case "$out" in
        *'"working_set_bytes"'*) ;;
        *) printf '  FAIL  working-set  %s/%s: no usable JSON from the run\n' \
                  "$alg" "$mb"
           fail=1; continue ;;
    esac

    printf '%s' "$out" | python3 -c '
import json, sys
p = json.load(sys.stdin)["parameters"]
want = p["batch_messages"] * p["blocks_per_message"] * p["block_bytes"]
got  = p["working_set_bytes"]
alg, mb = sys.argv[1], sys.argv[2]
# The block count itself must follow the algorithm block size: a message plus
# 1 length-field byte and the length must fit, or it spills into another block.
blk = p["block_bytes"]
if blk not in (64, 128):
    print("  FAIL  working-set  %s/%s: unexpected block size %d" % (alg, mb, blk))
    raise SystemExit(1)
if got != want:
    print("  FAIL  working-set  %s/%-4s reports %d, corpus is %d x %d x %d = %d"
          % (alg, mb, got, p["batch_messages"], p["blocks_per_message"], blk, want))
    raise SystemExit(1)
print("  ok    working-set  %-7s %4s B -> %d blk x %d B, %9d bytes total"
      % (alg, mb, p["blocks_per_message"], blk, got))
' "$alg" "$mb" || fail=1
done

[ "$fail" = 0 ] || {
    echo ""
    echo "  The reported working set does not match the corpus that was built."
    echo "  Look for a block size assumed rather than taken from the algorithm."
    exit 1
}
