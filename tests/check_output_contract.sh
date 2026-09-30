#!/bin/sh
#
# check_output_contract.sh -- the machine-readable output must be machine-readable.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, The valubench authors. See LICENSE.
#
# Every other test here checks that the numbers are right. This one checks that
# they can be read at all, which nothing did: CI built the binary, ran the
# kernel checks and never once parsed the JSON the tool exists to emit, nor
# exercised a single documented exit code.
#
# Two defects lived in that gap for months and were found by review rather than
# by testing:
#
#   - the energy `sources` array separated elements on the loop index rather
#     than on what had been emitted, so a machine whose first power source was
#     invalid produced "sources": [, {...}] -- unparseable;
#   - seven options were parsed with atoi(), so `--threads abc` ran on one
#     thread and reported it as though it had been requested.
#
# So: parse every document, check every exit code, and confirm the tools that
# consume the contract can still drive the binary.
#
# Usage: check_output_contract.sh <binary> [srcdir]

set -eu

BIN=${1:?usage: $0 <binary> [srcdir]}
SRC=${2:-.}
[ -x "$BIN" ] || { echo "  skip  output-contract  ($BIN not built)"; exit 0; }

fail=0
pass=0

# ---- every JSON document the tool can emit must parse -----------------------
#
# One run per document shape, plus a couple of parameter combinations, because
# the shapes differ: a device run carries a "device" object a CPU run does not,
# and an iterated run reaches padding paths a single-iteration one does not.
check_json() {
    desc=$1; shift
    out=$("$@" 2>/dev/null || true)
    if printf '%s' "$out" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
        pass=$((pass + 1))
    else
        printf '  FAIL  output-contract  %s: not valid JSON\n' "$desc"
        printf '%s' "$out" | head -3 | sed 's/^/          /'
        fail=1
    fi
}

check_json "result, defaults"      "$BIN" --json --samples 2 --time-ms 30 --warmup-ms 30
check_json "result, sha512"        "$BIN" --json --algorithm sha512 --samples 2 --time-ms 30 --warmup-ms 30
check_json "result, iterated"      "$BIN" --json --algorithm sha1 --message-bytes 64 --iterations 4 \
                                        --samples 2 --time-ms 30 --warmup-ms 30
check_json "result, all threads"   "$BIN" --json --threads 2 --samples 2 --time-ms 30 --warmup-ms 30
check_json "capabilities"          "$BIN" --list --json
check_json "devices"               "$BIN" --list-devices --json

# ---- the capability dump is a contract, so check its shape ------------------
if "$BIN" --list --json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
need = ("schema", "benchmark", "algorithms", "kernels", "limits",
        "exit_codes", "transfer_modes", "host_memory_modes",
        "primitives_modes", "compile_modes", "backend_filters",
        "backends", "devices")
missing = [k for k in need if k not in d]
if missing:
    print("  FAIL  output-contract  capabilities missing: %s" % ", ".join(missing))
    raise SystemExit(1)
if not str(d["schema"]).startswith("valubench/capabilities/"):
    print("  FAIL  output-contract  capabilities schema is %r" % d["schema"])
    raise SystemExit(1)
for k in d["kernels"]:
    for f in ("name", "isa", "algorithm", "lanes", "streams", "where", "available"):
        if f not in k:
            print("  FAIL  output-contract  kernel row missing %s: %r" % (f, k))
            raise SystemExit(1)
' ; then pass=$((pass + 1)); else fail=1; fi

# ---- the precomputed reference must equal the recomputed one ----------------
#
# --expect lets a sweep solve an iteration ladder once instead of per point,
# which only stays honest if the supplied value is the value the binary would
# have computed. So: emit a ladder, then run each rung twice -- once verifying
# against the supplied answer, once against its own -- and require the same
# checksum out of both. A wrong --expect must fail the run, or the flag is a
# way to switch verification off rather than a way to speed it up.
if "$BIN" --reference-ladder 1,2,4,8 --algorithm md5 --message-bytes 64 \
        --working-set-kb 64 2>/dev/null | python3 -c '
import json, subprocess, sys
doc = json.load(sys.stdin)
binary = sys.argv[1]
if doc.get("schema") != "valubench/reference/1":
    print("  FAIL  output-contract  reference schema is %r" % doc.get("schema"))
    raise SystemExit(1)
common = [binary, "--json", "--algorithm", "md5", "--message-bytes", "64",
          "--working-set-kb", "64", "--kernel", "md5/scalar-s1",
          "--samples", "1", "--time-ms", "20", "--warmup-ms", "20"]
for rung in doc["checksums"]:
    it = str(rung["iterations"])
    def run(extra):
        p = subprocess.run(common + ["--iterations", it] + extra,
                           capture_output=True, text=True)
        return p.returncode, p.stdout
    rc_s, out_s = run(["--expect", rung["checksum"]])
    rc_r, out_r = run([])
    if rc_s not in (0, 3) or rc_r not in (0, 3):
        print("  FAIL  output-contract  iterations=%s exited %d/%d"
              % (it, rc_s, rc_r))
        raise SystemExit(1)
    got_s = json.loads(out_s)["verification"]["checksum"]
    got_r = json.loads(out_r)["verification"]["checksum"]
    if not (got_s == got_r == rung["checksum"]):
        print("  FAIL  output-contract  iterations=%s: ladder %s, supplied %s, "
              "recomputed %s" % (it, rung["checksum"], got_s, got_r))
        raise SystemExit(1)
    # A supplied value that is wrong must be caught, not trusted.
    bad = "f" * len(rung["checksum"])
    rc_bad, _ = run(["--expect", bad])
    if rc_bad != 1:
        print("  FAIL  output-contract  iterations=%s: a wrong --expect "
              "exited %d, not 1" % (it, rc_bad))
        raise SystemExit(1)
' "$BIN"; then pass=$((pass + 1)); else fail=1; fi

# ---- the documented exit codes must be the ones actually used ---------------
#
# All four are reachable from the command line. verify_failed=1 needs no broken
# kernel: an --expect that no kernel can match produces it, and is the cheapest
# way to prove the verification path still reports as one.
expect_exit() {
    want=$1; desc=$2; shift 2
    "$@" >/dev/null 2>&1 && got=0 || got=$?
    if [ "$got" = "$want" ]; then
        pass=$((pass + 1))
    else
        printf '  FAIL  output-contract  %s: expected exit %s, got %s\n' "$desc" "$want" "$got"
        fail=1
    fi
}

expect_exit 2 "unknown option"        "$BIN" --no-such-option
expect_exit 2 "missing argument"      "$BIN" --threads
expect_exit 2 "non-numeric argument"  "$BIN" --threads abc
expect_exit 2 "trailing garbage"      "$BIN" --message-bytes 12x
expect_exit 2 "negative"              "$BIN" --threads -4
expect_exit 2 "out of range"          "$BIN" --samples 0
expect_exit 2 "unknown algorithm"     "$BIN" --algorithm nosuchalg
expect_exit 2 "unknown host memory"   "$BIN" --host-memory locked
expect_exit 2 "zero pipeline chunks"  "$BIN" --transfer overlap --pipeline-chunks 0
# A chunk count without a pipelined upload would be silently ignored.
expect_exit 2 "chunks without overlap" "$BIN" --transfer stream --pipeline-chunks 4
expect_exit 2 "unknown primitives"    "$BIN" --primitives fast
expect_exit 2 "geometry, one number"  "$BIN" --device-geometry 4096
expect_exit 2 "geometry, zero"        "$BIN" --device-geometry 0,64
expect_exit 2 "geometry, not numbers" "$BIN" --device-geometry a,b
expect_exit 2 "geometry, negative"    "$BIN" --device-geometry -4096,64
expect_exit 2 "dump dir missing"      "$BIN" --dump-device-code /nonexistent/valubench
expect_exit 2 "dump dir is a file"    "$BIN" --dump-device-code "$BIN"
# A device-only option on a run that can only pick a CPU kernel would do
# nothing at all -- an empty dump directory, a geometry never used.
expect_exit 2 "geometry with --where cpu" \
    "$BIN" --device-geometry 4096,64 --where cpu
expect_exit 2 "neutral primitives on a CPU kernel" \
    "$BIN" --primitives neutral --kernel md5/scalar-s1
expect_exit 2 "unknown backend"       "$BIN" --backend metal
expect_exit 2 "unknown compile mode"  "$BIN" --compile-mode fatbin
expect_exit 2 "backend with --where cpu" "$BIN" --backend cuda --where cpu
expect_exit 2 "compile mode on a CPU kernel" \
    "$BIN" --compile-mode cubin --kernel md5/scalar-s1
# A kernel the backend filter excludes, and a CUDA compile mode for a kernel
# that is not CUDA: exit 2 whether or not the device exists. A contradiction is
# wrong on every machine, so it outranks the 4 a missing device would give.
expect_exit 2 "kernel excluded by --backend" \
    "$BIN" --backend cuda --kernel md5/ocl-s1
expect_exit 2 "compile mode on an OpenCL kernel" \
    "$BIN" --compile-mode cubin --kernel md5/ocl-s1
expect_exit 2 "import from a missing file" "$BIN" --import-ptx /nonexistent.ptx
# Imported PTX is one kernel, so it needs --kernel to say which.
expect_exit 2 "import without --kernel" "$BIN" --import-ptx "$0"
expect_exit 2 "import for an OpenCL kernel" \
    "$BIN" --import-ptx "$0" --kernel md5/ocl-s1

# --device used to bound at VB_MAX_THREADS (1024) while the measurement path
# reserved VB_OCL_MAX_DEVICES (32), and copied between them unchecked. Indices
# out of range were rejected, so overflowing it needed repeats -- and repeats
# were accepted. Thirty-three of them segfaulted on any machine with a GPU.
#
# These cases need no device present: the parser rejects them before the
# OpenCL path is entered, which is the point.
expect_exit 2 "device list too long"  "$BIN" --device "$(python3 -c 'print(",".join(["0"]*33))')"
expect_exit 2 "device index repeated" "$BIN" --device 0,0
expect_exit 2 "device out of range"   "$BIN" --device 99
expect_exit 2 "device negative"       "$BIN" --device -1
expect_exit 2 "device overflows long" "$BIN" --device 99999999999999999999
expect_exit 2 "device not a number"   "$BIN" --device zz
expect_exit 2 "expect, wrong width"   "$BIN" --expect deadbeef
expect_exit 2 "expect, not hex"       "$BIN" --expect "$(printf 'z%.0s' $(seq 32))"
expect_exit 2 "ladder descending"     "$BIN" --reference-ladder 8,4,1
expect_exit 2 "ladder empty"          "$BIN" --reference-ladder ""
expect_exit 2 "ladder not a list"     "$BIN" --reference-ladder "1;2"

# A --kernel that contradicts an explicit --algorithm used to win silently and
# measure the kernel's own algorithm under the other's name.
expect_exit 2 "kernel contradicts algorithm" \
    "$BIN" --algorithm sha1 --kernel md5/scalar-s1

# A verification failure is exit 1 however the candidates were restricted.
# Under --where cpu it used to come back as 2, "no cpu kernel is available",
# and a sweep recorded a wrong answer from the hardware as a usage error.
NOMATCH=00000000000000000000000000000000
QUICK="--threads 2 --samples 2 --time-ms 20 --warmup-ms 0"
expect_exit 1 "verification failure"             "$BIN" --expect $NOMATCH $QUICK
expect_exit 1 "verification failure, --where cpu" \
    "$BIN" --where cpu --expect $NOMATCH $QUICK
expect_exit 1 "verification failure, forced kernel" \
    "$BIN" --kernel md5/scalar-s1 --expect $NOMATCH $QUICK
# Exit 4 is a valid command this machine, now, could not run: the test for it
# is whether the same command would succeed on a different machine. Hiding the
# device runtimes from one run is how a GPU box stands in for a machine without
# them -- an ICD loader pointed at no vendors, and no CUDA device visible.
NODEV="OCL_ICD_VENDORS=/nonexistent CUDA_VISIBLE_DEVICES="
expect_exit 4 "forced device kernel, no device runtime" \
    env $NODEV "$BIN" --kernel md5/ocl-s1 $QUICK
expect_exit 4 "--where device, no device runtime" \
    env $NODEV "$BIN" --where device $QUICK
# A contradiction outranks a missing device: a command wrong on every machine
# is 2 even where it also could not run. With the device present these would
# pass whichever came first, so they run with it hidden.
expect_exit 2 "kernel excluded by --backend, no device runtime" \
    env $NODEV "$BIN" --backend cuda --kernel md5/ocl-s1 $QUICK
expect_exit 2 "compile mode on an OpenCL kernel, no device runtime" \
    env $NODEV "$BIN" --compile-mode cubin --kernel md5/ocl-s1 $QUICK
expect_exit 2 "device kernel with --where cpu, no device runtime" \
    env $NODEV "$BIN" --where cpu --kernel md5/ocl-s1 $QUICK
# A --device inside the static bound but past this machine's last device used
# to be dropped by autotune, which then measured the CPU and exited 0.
ndev=$("$BIN" --list-devices --json 2>/dev/null |
       python3 -c 'import json,sys; print(len(json.load(sys.stdin)["devices"]))' \
       2>/dev/null || echo 0)
if [ "$ndev" -lt 32 ]; then
    expect_exit 4 "--device past the last device" "$BIN" --device "$ndev" $QUICK
fi
# Two checks make an allocation fail on demand by capping the address space
# with ulimit -v. macOS refuses to set that limit at all, and the shell exits 1
# before the binary runs -- which read as the binary exiting 1. They are skipped,
# visibly, only off Linux and only when the limit cannot be set; on Linux a limit
# that will not set is a failure of the check, not a reason to pass it.
as_limit=yes
if [ "$(uname -s)" != Linux ] && ! sh -c 'ulimit -v 400000' 2>/dev/null; then
    as_limit=no
    echo "  skip  output-contract  allocation failure (no address-space limit on $(uname -s))"
fi

# A corpus the address space cannot hold. Not under ASan, which reserves most of
# the address space for itself and would fail for its own reasons.
if [ "$as_limit" = yes ] && ! grep -aq __asan_init "$BIN"; then
    expect_exit 4 "corpus cannot be allocated" \
        sh -c 'ulimit -v 400000 && exec "$0" --kernel md5/scalar-s1 \
               --working-set-kb 1048576 --threads 1 --samples 1 --time-ms 20 \
               --warmup-ms 0' "$BIN"
fi

# A valid run exits 0, or 3 if the machine was too noisy to trust the number.
# Both mean it ran; 1, 2 and 4 mean it did not. Asserting 0 here would make
# this test flaky on precisely the shared, contended runners CI uses -- which is
# the same reason the tool reports noise instead of hiding it.
"$BIN" --samples 5 --time-ms 120 --warmup-ms 120 >/dev/null 2>&1 && rc=0 || rc=$?
if [ "$rc" = 0 ] || [ "$rc" = 3 ]; then
    pass=$((pass + 1))
elif [ "$rc" = 4 ]; then
    printf '  FAIL  output-contract  a default run exited 4, could not run: this\n'
    printf '        runner cannot run a default measurement, which is the runner at\n'
    printf '        fault rather than the contract\n'
    fail=1
else
    printf '  FAIL  output-contract  a valid run exited %s (want 0 or 3)\n' "$rc"
    fail=1
fi

# ---- a rejected argument must not be silently substituted -------------------
#
# The atoi defect: the run proceeded and the JSON recorded the substituted
# value, so the failure was invisible in the artifact it produced.
if "$BIN" --threads abc --json >/dev/null 2>&1; then
    echo "  FAIL  output-contract  --threads abc was accepted"
    fail=1
else
    pass=$((pass + 1))
fi

# ---- the tools that consume the contract must still drive the binary --------
if [ -f "$SRC/tools/sweep.py" ]; then
    tmp=$(mktemp -d)
    if python3 "$SRC/tools/sweep.py" --bin "$BIN" --algorithm md5 \
            --kernel md5/scalar-s1,md5/scalar-s2 --threads 1 \
            --samples 2 --time-ms 30 --warmup-ms 30 -q --csv "$tmp/a.csv" >/dev/null 2>&1 \
       && [ -s "$tmp/a.csv" ] \
       && python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 2, "expected 2 rows, got %d" % len(rows)
for r in rows:
    assert r["checksum"], "no checksum recorded"
    assert r["verified"] == "true", "row not verified"
' "$tmp/a.csv" 2>/dev/null; then
        pass=$((pass + 1))
    else
        echo "  FAIL  output-contract  sweep.py could not drive the binary"
        fail=1
    fi
    rm -rf "$tmp"
fi

# A grid outside what the binary accepts is refused before any point runs, with
# the binary's usage status. Each of these used to fail once per point and exit 1.
if [ -f "$SRC/tools/sweep.py" ]; then
    expect_exit 2 "sweep, --threads 0" \
        python3 "$SRC/tools/sweep.py" --bin "$BIN" --threads 0 -q
    expect_exit 2 "sweep, --time-ms past the binary's limit" \
        python3 "$SRC/tools/sweep.py" --bin "$BIN" --time-ms 3600001 -q
fi

# A point this machine could not run (exit 4) is recorded and the sweep goes on:
# here a corpus beyond an address-space limit, beside one that fits. The sweep
# still exits 1, as for any failed point, so what shows it carried on is its
# summary and the row that did run -- a traceback would also exit 1. Not under
# ASan, which reserves most of the address space for itself.
if [ -f "$SRC/tools/sweep.py" ] && [ "$as_limit" = yes ] &&
   ! grep -aq __asan_init "$BIN"; then
    tmp=$(mktemp -d)
    out=$(sh -c 'ulimit -v 600000 && exec python3 "$0" --bin "$1" \
                 --algorithm md5 --kernel md5/scalar-s1 --threads 1 \
                 --working-set-kb 1024,1048576 --samples 2 --time-ms 20 \
                 --warmup-ms 0 --csv "$2"' \
                 "$SRC/tools/sweep.py" "$BIN" "$tmp/b.csv" 2>&1) && rc=0 || rc=$?
    if [ "$rc" = 1 ] && printf '%s' "$out" | grep -q "could not run:" \
       && printf '%s' "$out" | grep -q "1/2 points in .*1 failed" \
       && python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 1, "expected the one point that ran, got %d" % len(rows)
assert rows[0]["verified"] == "true", "row not verified"
' "$tmp/b.csv" 2>/dev/null; then
        pass=$((pass + 1))
    else
        echo "  FAIL  output-contract  sweep.py did not record a point that could not"
        echo "        run and carry on (exit $rc)"
        fail=1
    fi
    rm -rf "$tmp"
fi

# A reference ladder must be held to the same digest-fits-message rule as
# --iterations. It was not: the guard tested cfg.iterations, which the ladder
# never sets, so the feedback memcpy wrote a 64-byte SHA-512 digest over the
# head of a 55-byte message and past the end of its allocation. A heap
# overflow inside the correctness oracle, reachable from a documented flag.
for spec in "--algorithm sha512 --reference-ladder 1,2" \
            "--algorithm md5 --message-bytes 8 --reference-ladder 1,2"; do
    # shellcheck disable=SC2086
    "$BIN" $spec >/dev/null 2>&1 && rc=0 || rc=$?
    if [ "$rc" = 2 ]; then
        pass=$((pass + 1))
    else
        echo "  FAIL  output-contract  '$spec' exited $rc, want 2 (usage)"
        fail=1
    fi
done

# And a ladder whose message is long enough must still run.
for spec in "--algorithm sha512 --message-bytes 64 --reference-ladder 1,2" \
            "--algorithm md5 --reference-ladder 1,2,4"; do
    # shellcheck disable=SC2086
    if "$BIN" $spec >/dev/null 2>&1; then
        pass=$((pass + 1))
    else
        echo "  FAIL  output-contract  '$spec' was refused but is valid"
        fail=1
    fi
done

# A forced kernel fixes the algorithm, so every check that depends on the
# algorithm must see the kernel's, not the default. They did not: --kernel was
# resolved after --message-bytes and --expect had been validated against MD5,
# so `--kernel sha512/scalar-s1 --iterations 2` passed the digest-fits-message
# guard and the oracle wrote a 64-byte digest into a 55-byte message. glibc
# aborted on the corrupted heap.
expect_exit 2 "forced kernel, digest outgrows message" \
    "$BIN" --kernel sha512/scalar-s1 --iterations 2 --threads 1 \
           --samples 2 --time-ms 20 --warmup-ms 0
# --expect is parsed at the forced kernel's digest width: an MD5-width value
# for a SHA-512 kernel is a usage error, not a run that then fails verification.
expect_exit 2 "forced kernel, expect at wrong width" \
    "$BIN" --kernel sha512/scalar-s1 --expect "$(printf '0%.0s' $(seq 32))"
# And the right width is accepted.
sum=$("$BIN" --algorithm sha512 --working-set-kb 64 --reference-ladder 1 2>/dev/null |
      python3 -c 'import json,sys; print(json.load(sys.stdin)["checksums"][0]["checksum"])' \
      2>/dev/null || true)
"$BIN" --kernel sha512/scalar-s1 --working-set-kb 64 --expect "$sum" --threads 1 \
       --samples 2 --time-ms 20 --warmup-ms 0 >/dev/null 2>&1 && rc=0 || rc=$?
if [ -n "$sum" ] && { [ "$rc" = 0 ] || [ "$rc" = 3 ]; }; then
    pass=$((pass + 1))
else
    echo "  FAIL  output-contract  forced sha512 kernel refused its own checksum (exit $rc)"
    fail=1
fi

# A corpus with more messages than the message length can hold distinct ones
# repeats messages, and repeated digests cancel under XOR. 1536 one-byte
# messages are 256 values six times over, so the fingerprint was all zeros and
# a kernel returning nothing would have verified. Refused, as is its ladder.
expect_exit 2 "corpus repeats messages"        "$BIN" --message-bytes 1 --working-set-kb 96
expect_exit 2 "corpus repeats messages, ladder" \
    "$BIN" --message-bytes 2 --working-set-kb 8192 --reference-ladder 1
# The largest corpus that stays distinct still runs: 65536 two-byte messages.
expect_exit 0 "corpus exactly distinct, ladder" \
    "$BIN" --message-bytes 2 --working-set-kb 4096 --reference-ladder 1

# The default thread count is the CPUs the process may use, not the CPUs the
# machine has. It was the online count, so under `taskset -c 0,1` on a 32-CPU
# box a default run put 32 workers on two CPUs and reported threads_used 32.
# compare.py's exit statuses are its interface, as the binary's are: 0 no
# regression, 1 a regression, 2 input it cannot use, 3 not comparable. Every
# input error used to exit 1, so a mistyped path read as a regression, and a
# JSON file holding [] crashed it. The slower and mismatched results are the
# real one with its median or checksum edited.
if [ -f "$SRC/tools/compare.py" ]; then
    tmp=$(mktemp -d)
    cmp="python3 $SRC/tools/compare.py"
    if "$BIN" --json --kernel md5/scalar-s1 --threads 1 --samples 3 \
           --time-ms 30 --warmup-ms 0 > "$tmp/base.json" 2>/dev/null ||
       [ -s "$tmp/base.json" ]; then
        python3 - "$tmp" <<'PY'
import json, os, sys
d = sys.argv[1]
doc = json.load(open(os.path.join(d, "base.json")))
slow = json.loads(json.dumps(doc)); slow["result"]["median"] *= 0.5
json.dump(slow, open(os.path.join(d, "slow.json"), "w"))
other = json.loads(json.dumps(doc)); other["verification"]["checksum"] = "0" * 32
json.dump(other, open(os.path.join(d, "other.json"), "w"))
PY
        echo '[]' > "$tmp/list.json"
        mkdir "$tmp/dir" && cp "$tmp/base.json" "$tmp/list.json" "$tmp/dir/"
        expect_exit 0 "compare, same results"      $cmp "$tmp/base.json" "$tmp/base.json"
        expect_exit 1 "compare, a regression"      $cmp "$tmp/base.json" "$tmp/slow.json"
        expect_exit 3 "compare, checksum mismatch" $cmp "$tmp/base.json" "$tmp/other.json"
        expect_exit 2 "compare, missing file"      $cmp "$tmp/base.json" "$tmp/nonesuch.json"
        expect_exit 2 "compare, a JSON list"       $cmp "$tmp/list.json" "$tmp/base.json"
        expect_exit 2 "compare, negative threshold" \
            $cmp --threshold -1 "$tmp/base.json" "$tmp/base.json"
        # A stray non-result in a results directory is skipped, not fatal.
        expect_exit 0 "compare, a directory with a stray file" \
            $cmp "$tmp/dir" "$tmp/base.json"
    else
        echo "  FAIL  output-contract  no result to drive compare.py with"
        fail=1
    fi
    rm -rf "$tmp"
fi

if command -v taskset >/dev/null 2>&1 && [ "$(nproc --all 2>/dev/null || echo 1)" -ge 3 ]; then
    used=$(taskset -c 0,1 "$BIN" --json --where cpu --samples 2 --time-ms 20 \
               --warmup-ms 0 2>/dev/null |
           python3 -c 'import json,sys; print(json.load(sys.stdin)["environment"]["threads_used"])' \
           2>/dev/null || true)
    if [ "$used" = 2 ]; then
        pass=$((pass + 1))
    else
        echo "  FAIL  output-contract  default threads under a 2-CPU mask: got '$used', want 2"
        fail=1
    fi
fi

[ "$fail" = 0 ] || exit 1
printf '  ok    output-contract  (%d checks: JSON parses, exit codes, tooling)\n' "$pass"
