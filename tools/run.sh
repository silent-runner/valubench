#!/usr/bin/env bash
#
# run.sh -- one-command capture for a time-boxed session on rented hardware.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, The valubench authors. See LICENSE.
#
# Written for the case where the clock is running on a machine you are paying
# for and you want data rather than a shell history. It captures the
# environment, gates on correctness, runs the measurement matrix, and leaves a
# single tarball to copy off the box -- including if the run fails or is
# interrupted.
#
#   ./tools/run.sh                  everything this machine can do
#   ./tools/run.sh -q               quick pass, for a first smoke test
#   ./tools/run.sh --only cpu       CPU phases only
#   ./tools/run.sh --only device    device phases only, if there is a device
#   ./tools/run.sh --skip-check     skip 'make check' (the gate -- not advised)
#
# CPU phases run first and device phases second, and the device phases skip
# themselves when the machine has no OpenCL device -- so the same command is
# correct on a laptop, a CPU instance and a GPU instance. On a metered GPU box
# where the accelerator is the reason you are paying, `--only device` puts the
# expensive question first; run again without it for the rest.
#
# WHAT THE PHASES MEASURE, in the order they run. Earlier phases answer the
# questions that need the least hardware luck, so an interrupted session still
# leaves something worth keeping.
#
#   C1  The ISA ladder: MD5 on every instruction set this machine has, at every
#       stream count, one thread. The headline, and the phase most worth having
#       on an unfamiliar part. Rungs come from the binary's own capability dump,
#       so a machine with an ISA nobody anticipated still gets a full ladder.
#
#   C2  Licence-based downclocking, sampled *during* the run. Wide vector work
#       can pull the clock down, and the throughput ratio alone cannot tell you
#       whether it did. Needs a cpufreq interface, which most VMs do not expose,
#       so this phase skips itself more often than it runs.
#
#   C3  Energy per hash. Wider work draws more power, so a throughput ratio and
#       an efficiency ratio can point opposite ways, and only one of them is a
#       purchasing argument. Needs readable powercap counters.
#
#   C4  A fixed-function hash unit against the integer vector path, where the
#       CPU has one. What that unit is worth is not a constant -- it depends
#       entirely on the vector path it sits beside, and has been measured both
#       faster and slower than the SIMD kernels on different cores.
#
#   C5  All three algorithms, one thread and all cores. The scaling shape, and
#       the verification checksums -- which are machine-independent, so a new
#       part matching the recorded values is the stronger result of the two.
#
#   C6  Stream interleaving for every (algorithm, ISA) pair. Interleaving
#       independent chains is the largest single effect in the benchmark, and
#       the best stream count is a property of the algorithm and the core rather
#       than a constant. This phase has repeatedly found things nothing else did.
#
#   C7  Message length and working set on the best MD5 kernel. Compressions per
#       second should stay flat across both; a machine where it does not is the
#       finding.
#
#   D1  The PCIe crossover, and break-even against the whole CPU. The question
#       the device path exists to answer: how much work an upload has to buy
#       before the accelerator is worth its bus. Solved by fit, not bracketed.
#
#   D2  Every algorithm at every device stream count, resident. Whether "one
#       stream is right on a device" is a property of devices or of the one
#       device it was first measured on.
#
#   D3  Working set on the device, the memory axis.
#
#   D5  The compute plateau, resident. D1 puts the link in the timed region and
#       finds where compute overtakes it; this leaves the corpus on the device
#       and finds where compute overtakes *memory*. Read compressions/sec, not
#       hash rate: the first flattens at the ceiling, the second falls by
#       construction. Two working sets, so a cache-resident corpus cannot pass
#       itself off as a compute ceiling; two message sizes, because the knee
#       is compressions per byte read and blocks_per_message is half of that
#       ratio. All three algorithms.
#
#   D4  Multi-device slicing, where there is more than one device. The checksum
#       must match the single-device value exactly; a difference means the work
#       split is wrong, not that the machine is fast.
#
# Output is one directory plus a tarball. Copy it off before tearing the machine
# down -- and preferably as phases land, not at the end.

set -u

OUT=""
QUICK=0
SKIP_CHECK=0
ONLY=""
XFER_WS=262144          # KiB of corpus for the device streaming sweep

usage() {
    cat <<EOF
Usage: $0 [options]

  -o DIR            output directory (default: results-<host>-<timestamp>)
  -q, --quick       smaller grids and fewer samples, for a first pass
      --only WHICH  'cpu' or 'device' -- run only that half
      --skip-check  skip 'make check' (the correctness gate -- not advised)
      --ws KB       corpus for the device streaming sweep (default $XFER_WS KiB)
  -h, --help        this text

Run it on a quiet machine. All-core numbers are the least trustworthy thing
here; the single-thread ISA ladder in C1 is the headline and is stable.

The streaming sweep wants a corpus large enough that the upload takes
milliseconds, or launch overhead swamps the transfer it is trying to measure.
Keep --ws well under the device's max allocation, which --list-devices reports.
EOF
}

# `shift 2` on a lone trailing option shifts nothing -- the script is set -u,
# not set -e, so the failure is silent and $1 stays put forever. Check before
# shifting rather than after.
need_arg() {
    [ "$2" -ge 2 ] || { echo "run.sh: $1 needs an argument" >&2; exit 2; }
}

while [ $# -gt 0 ]; do
    case "$1" in
        -o) need_arg "$1" $#; OUT="${2:-}"; shift 2 ;;
        -q|--quick) QUICK=1; shift ;;
        --only) need_arg "$1" $#; ONLY="${2:-}"; shift 2 ;;
        --skip-check) SKIP_CHECK=1; shift ;;
        --ws) need_arg "$1" $#; XFER_WS="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$0: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

case "$ONLY" in
    ""|cpu|device) ;;
    *) echo "$0: --only takes 'cpu' or 'device', not '$ONLY'" >&2; exit 2 ;;
esac
DO_CPU=1; DO_DEV=1
[ "$ONLY" = device ] && DO_CPU=0
[ "$ONLY" = cpu ]    && DO_DEV=0

cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
BIN=$REPO/build/valubench
SWEEP=$REPO/tools/sweep.py

[ -n "$OUT" ] || OUT="$REPO/results-$(hostname -s 2>/dev/null || echo host)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT" || exit 1
LOG="$OUT/run.log"

START=$(date +%s)
elapsed() { printf '%dm%02ds' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 )); }

# The log is written first and the terminal second, deliberately, and the
# showing happens in a subshell so it can die without taking the script with it.
#
# The reason is SIGPIPE. Pipe this script into anything that exits early -- a
# `| head`, or an ssh session that drops -- and the next write to stdout kills
# the shell outright; redirecting stderr does not help, because it is a signal
# rather than an error, and a `tee` would take it too and stop writing the file.
# On a metered box the transcript is the thing you paid for, so it must not
# depend on anyone still watching. Run under tmux or nohup anyway.
emit() { printf '%s\n' "$1" >> "$LOG"; ( printf '%s\n' "$1" ) 2>/dev/null || true; }
say()  { emit "$(printf '\n=== [%s] %s' "$(elapsed)" "$*")"; }
note() { emit "$(printf '    %s' "$*")"; }
# For command output: same guarantee, for anything that would otherwise tee.
show() { local t; t=$(cat); emit "$t"; }

# Always leave a tarball, however the run ends.
package() {
    tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null
    printf '\n=== [%s] done\n' "$(elapsed)"
    printf '    %s\n' "$OUT.tar.gz"
    printf '    copy it off with:  scp %s:%s .\n' \
           "$(hostname -s 2>/dev/null || echo HOST)" "$OUT.tar.gz"
}
# Ctrl-C must stop the run, not fall through into the next phase. A single
# trap that only returns leaves that to the shell's discretion, and can run the
# packaging twice -- once on the signal and again on the eventual exit.
#
# So: signals set the conventional status and exit; EXIT does the packaging
# once and preserves whatever status brought us here.
vb_packaged=0
package_once() {
    [ "$vb_packaged" = 1 ] && return 0
    vb_packaged=1
    package
}
trap 'vb_rc=$?; package_once; exit $vb_rc' EXIT
trap 'say "interrupted"; exit 130' INT
trap 'say "terminated"; exit 143' TERM

if [ "$QUICK" = 1 ]; then
    SAMPLES=3; TIME_MS=100; WARMUP=150; FREQ_SECS=6
    ITERS="1:6:+1,16,64"
    WS_AXIS="1024,16384,262144"
    XFER_WS=$(( XFER_WS / 4 ))
else
    SAMPLES=9; TIME_MS=200; WARMUP=400; FREQ_SECS=20
    ITERS="1:8:+1,16:1024:*2"
    WS_AXIS="256:1048576:*4"
fi

SW="python3 $SWEEP --samples $SAMPLES --time-ms $TIME_MS --warmup-ms $WARMUP
    --keep-going -q"

# nproc first, because on Linux it counts the affinity mask a pool can use.
# macOS has no nproc, and falling through to 1 would run C5's "all cores" pass
# on one thread and label the result as the whole machine.
NPROC=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)

# The Makefile needs GNU make 4.3 or newer. macOS's own make is 3.81 and
# Homebrew installs a current one as gmake; where gmake exists it is GNU make,
# and on the Linux distributions that ship it, the same binary as make.
MK=make
command -v gmake >/dev/null 2>&1 && MK=gmake

# ---------------------------------------------------------------- environment

say "environment"
{
    echo "# date";  date -u
    echo; echo "# uname"; uname -a
    echo; echo "# machine identity"
    # DMI first, because it needs no network and no provider API and works on
    # every host: on EC2 Nitro sys_vendor is "Amazon EC2" and product_name is
    # the instance type itself, which is what the metadata service would have
    # told us. On a bare-metal or non-cloud host it names the actual hardware.
    for f in sys_vendor product_name board_vendor; do
        [ -r "/sys/class/dmi/id/$f" ] && \
            printf '  %-14s %s\n' "$f" "$(cat "/sys/class/dmi/id/$f" 2>/dev/null)"
    done
    # Then the provider metadata services, for the detail DMI does not carry.
    # Timeouts are not optional: these are link-local addresses that are
    # blackholed rather than refused off-provider, and an untimed curl hangs
    # the whole capture before it has measured anything.
    CURL="curl -s --connect-timeout 1 --max-time 2"
    aws_tok=$($CURL -X PUT "http://169.254.169.254/latest/api/token" \
              -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)
    aws=$($CURL -H "X-aws-ec2-metadata-token: $aws_tok" \
          http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null)
    gcp=$($CURL -H "Metadata-Flavor: Google" \
          http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null)
    azure=$($CURL -H "Metadata:true" \
            "http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text" 2>/dev/null)
    for pair in "aws:$aws" "gcp:$gcp" "azure:$azure"; do
        v=${pair#*:}
        [ -n "$v" ] && printf '  %-14s %s\n' "${pair%%:*}" "$v"
    done
    echo; echo; echo "# cpu"
    if [ "$(uname -s)" = Darwin ]; then
        # No lscpu and no /proc. The core tiers matter more here than on any
        # other platform: Apple silicon mixes core types, and with no affinity
        # API nothing decides which type a worker lands on.
        sysctl machdep.cpu.brand_string hw.physicalcpu hw.logicalcpu hw.nperflevels
        i=0
        while [ "$i" -lt "$(sysctl -n hw.nperflevels 2>/dev/null || echo 0)" ]; do
            sysctl "hw.perflevel$i"; i=$((i + 1))
        done
    else
        lscpu 2>/dev/null || grep -m1 'model name' /proc/cpuinfo
    fi
    echo; echo "# flags of interest"
    # The two families spell their capability lists differently and share no
    # names, so asking for the x86 set on an AArch64 part prints six MISSINGs
    # and says nothing about the part you are actually on.
    case "$(uname -m)" in
        aarch64|arm64) FLAGS="asimd sve sve2 sha1 sha2 sha512" ;;
        *)             FLAGS="sse2 avx2 avx512f avx512dq avx512bw sha_ni" ;;
    esac
    if [ "$(uname -s)" = Darwin ]; then
        # Darwin reports these as hw.optional sysctls. Printed under the Linux
        # names, so "sve MISSING" means the same thing whichever kernel wrote
        # this file.
        for pair in asimd:neon sve:arm.FEAT_SVE sve2:arm.FEAT_SVE2 \
                    sha1:arm.FEAT_SHA1 sha2:arm.FEAT_SHA256 sha512:arm.FEAT_SHA512; do
            [ "$(sysctl -n "hw.optional.${pair#*:}" 2>/dev/null)" = 1 ] \
                && echo "  ${pair%%:*}" || echo "  ${pair%%:*} MISSING"
        done
    else
        for f in $FLAGS; do
            grep -qm1 " $f" /proc/cpuinfo && echo "  $f" || echo "  $f MISSING"
        done
    fi
    echo; echo "# cpufreq"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null || echo "  (no cpufreq)"
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c
    echo; echo "# powercap (energy)"
    ls /sys/class/powercap/ 2>/dev/null || echo "  (none)"
    for e in /sys/class/powercap/*/; do
        [ -r "$e/name" ] && printf '  %s = %s readable=%s\n' \
            "$(basename "$e")" "$(cat "$e/name" 2>/dev/null)" \
            "$( [ -r "$e/energy_uj" ] && echo yes || echo NO)"
    done
    echo; echo "# background upgrader"
    printf '  unattended-upgrades: %s\n' \
        "$(systemctl is-active unattended-upgrades 2>/dev/null || echo unknown)"
    echo; echo "# memory"; free -m 2>/dev/null | head -2
    echo; echo "# load"; uptime
    echo; echo "# compiler"; ${CC:-cc} --version 2>/dev/null | head -1
    echo; echo "# accelerators"
    nvidia-smi 2>&1 || echo "(no nvidia-smi)"
    echo; echo "# pcie link"
    nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current \
        --format=csv 2>&1 || echo "(unavailable)"
    echo; echo "# nvidia topology"
    nvidia-smi topo -m 2>&1 || echo "(unavailable)"
    echo; echo "# clinfo"
    (clinfo 2>&1 | head -40) || echo "(no clinfo)"
} > "$OUT/environment.txt" 2>&1

note "$(grep -m1 'Model name' "$OUT/environment.txt" | cut -c1-72)"
note "$NPROC logical CPUs"
grep -q '^  avx512f$' "$OUT/environment.txt" && HAVE512=1 || HAVE512=0
grep -q '^  sha_ni$'  "$OUT/environment.txt" && HAVESHA=1 || HAVESHA=0
note "avx512f: $([ $HAVE512 = 1 ] && echo yes || echo NO -- C1..C3 will be thin)"
note "sha_ni:  $([ $HAVESHA = 1 ] && echo yes || echo no -- C4 skipped)"

# A rented instance that reboots mid-run looks like a network fault from the
# other end, so record the one thing that reboots rented instances.
if systemctl is-active --quiet unattended-upgrades 2>/dev/null; then
    note "unattended-upgrades is ACTIVE and can reboot this machine mid-run:"
    note "  sudo systemctl disable --now unattended-upgrades"
fi

# Energy needs the driver loaded and the counters readable. -n is not optional:
# plain sudo prompts for a password where one is required, and with stderr
# discarded that is a silent hang before the capture has measured anything.
# Passwordless sudo is the norm on a rented instance and not the norm anywhere
# else.
#
# The modprobe is for AWS bare metal, whose kernels ship without intel_rapl_msr
# so that /sys/class/powercap does not exist at all -- C3 then skips reporting
# "no readable counter", which is true and unhelpful, since a package supplies
# the driver. Loading it here is too late if the module is absent (that needs
# linux-modules-extra, which the session setup script installs), but it costs
# nothing and covers the case where it is merely unloaded.
[ -d /sys/class/powercap ] || sudo -n modprobe intel_rapl_msr 2>/dev/null || true
sudo -n chmod a+r /sys/class/powercap/*/energy_uj 2>/dev/null || true
HAVE_RAPL=0
for e in /sys/class/powercap/intel-rapl:*/energy_uj \
         /sys/class/powercap/amd-rapl:*/energy_uj; do
    [ -r "$e" ] && HAVE_RAPL=1
done
if [ "$HAVE_RAPL" = 0 ] && ls /sys/class/powercap/ >/dev/null 2>&1; then
    note "no readable {intel,amd}-rapl:* energy counter."
    note "  src/power.c accepts those two prefixes. If environment.txt shows"
    note "  powercap entries under some other name, C3 will report nothing and"
    note "  the fix is one more prefix in that filter."
fi

# ------------------------------------------------------------------- build

say "build"
"$MK" -C "$REPO" -s clean >/dev/null 2>&1
if ! "$MK" -C "$REPO" -j"$NPROC" > "$OUT/build.log" 2>&1; then
    note "BUILD FAILED -- see build.log"; tail -20 "$OUT/build.log" | show; exit 1
fi
"$MK" -C "$REPO" config > "$OUT/make-config.txt" 2>&1
"$BIN" --list        > "$OUT/kernels.txt"      2>&1
"$BIN" --list --json > "$OUT/capabilities.json" 2>&1
"$BIN" --list-devices > "$OUT/devices.txt"     2>&1
note "kernels registered: $(grep -c 'yes$' "$OUT/kernels.txt") available"

# `grep -c` prints 0 and exits 1 when it matches nothing, so `|| echo 0`
# appended a second zero and NDEV became "0\n0" -- every `[ "$NDEV" -gt 0 ]`
# after it then errored instead of answering, on exactly the device-less
# machines the count exists to detect.
NDEV=$(grep -c '^\[[0-9]' "$OUT/devices.txt" 2>/dev/null || true)
NDEV=${NDEV:-0}
note "OpenCL devices: $NDEV"
if [ "$NDEV" -eq 0 ]; then
    note "no OpenCL device -- device phases will be skipped. Reason:"
    sed 's/^/      /' "$OUT/devices.txt" | show
fi
[ "$NDEV" -gt 0 ] || DO_DEV=0

# ------------------------------------------------------------ correctness gate

if [ "$SKIP_CHECK" = 0 ]; then
    # On an unfamiliar part this is a result in its own right, not a formality:
    # it is the evidence that the kernels this machine selected compute the same
    # digests as every other machine.
    say "correctness gate"
    if [ "$NDEV" -gt 0 ]; then
        note "this builds every OpenCL kernel variant, and NVIDIA JITs through"
        note "PTX, so it can take several minutes."
    fi
    if "$MK" -C "$REPO" check > "$OUT/check.log" 2>&1; then
        note "$(grep -E '^[0-9]+ checks' "$OUT/check.log" | tail -1)"
    else
        note "CHECK FAILED -- every number below is suspect. Continuing anyway;"
        note "the failure is the result in that case. See check.log"
        grep -E "FAIL|failures" "$OUT/check.log" | head -20 | show
        # Stop closed was the reviewer's suggestion and would break what this
        # harness is for: on a rented box the capture is the evidence you
        # diagnose from before terminating, and a failed gate is exactly when
        # you need it. But a bundle can be copied away from the warning that
        # came with it, so the bundle carries its own.
        : > "$OUT/INVALID"
        {
            echo "The correctness gate failed on this machine."
            echo
            echo "Every measurement in this capture was taken after 'make check'"
            echo "reported failures, and none of it should be quoted, compared or"
            echo "published. It is kept because a failed gate is a result and the"
            echo "capture is how you diagnose it."
            echo
            echo "See check.log for what failed."
        } >> "$OUT/INVALID"
    fi
else
    say "correctness gate SKIPPED by request"
fi

if [ "$DO_CPU" = 1 ]; then
# The best kernel per ISA rung, widest first, from the ladder C1 measures.
# Phases that want "the widest kernel and the one below it" ask this rather than
# naming kernels, because a named kernel is wrong on any machine that does not
# have it -- silently on x86 without AVX-512, and fatally on AArch64.
rungs() {
    python3 - "$OUT/c1-isa-ladder.csv" "${1:-9}" 2>/dev/null <<'PY'
import csv, sys
ORDER = ["scalar", "sse2", "avx2", "avx512", "neon", "sve", "sve2"]
try:
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["hashes_per_sec"]]
except OSError:
    rows = []
best = {}
for r in rows:
    isa = r["kernel"].split("/")[1].split("-")[0]
    v = float(r["hashes_per_sec"])
    if v > best.get(isa, (0, ""))[0]:
        best[isa] = (v, r["kernel"])
wide = sorted(best, key=lambda i: ORDER.index(i) if i in ORDER else -1,
              reverse=True)
print(" ".join(best[i][1] for i in wide[:int(sys.argv[2])]))
PY
}

# ------------------------------------------------- C1: the ISA ladder (headline)

say "C1  ISA ladder, single thread  (the headline)"
note "md5 across every ISA this machine has, at every stream count."

# The rungs come from the binary rather than from a list written here. A
# hardcoded x86 ladder degrades to a single scalar row on an AArch64 part --
# silently, because --keep-going treats the absent kernels as skips -- and the
# ladder is the one phase that most wants to run on an unfamiliar machine.
ISAS=$(python3 - "$OUT/capabilities.json" <<'PY'
import json, sys
ORDER = ["scalar", "sse2", "avx2", "avx512", "neon", "sve", "sve2"]
seen = []
for k in json.load(open(sys.argv[1]))["kernels"]:
    alg, _, rest = k["name"].partition("/")
    # where == "device" is the OpenCL path. It belongs in the GPU phases of
    # gpu_run.sh, not in a single-thread ISA ladder: it would sit at the top of
    # the table comparing a 24-EU iGPU against a scalar loop, and P7 would then
    # pick it as "the best MD5 kernel" and measure the device's axes instead of
    # the core's.
    if alg != "md5" or not k.get("available") or k.get("where") != "cpu":
        continue
    isa = rest.split("-")[0]
    if isa not in seen:
        seen.append(isa)
seen.sort(key=lambda i: (ORDER.index(i) if i in ORDER else len(ORDER), i))
print(" ".join(seen))
PY
)
note "rungs on this machine: ${ISAS:-none}"

KS=""
for isa in $ISAS; do
    for s in 1 2 3 4; do KS="$KS${KS:+,}md5/$isa-s$s"; done
done
$SW --algorithm md5 --kernel "$KS" --threads 1 --message-bytes 55 \
    --csv "$OUT/c1-isa-ladder.csv" >> "$LOG" 2>&1

if [ -s "$OUT/c1-isa-ladder.csv" ]; then
    python3 - "$OUT/c1-isa-ladder.csv" <<'PY' | show
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["hashes_per_sec"]]
best, order = {}, []
for r in rows:
    isa = r["kernel"].split("/")[1].split("-")[0]
    v = float(r["hashes_per_sec"]) / 1e6
    if isa not in order:
        order.append(isa)
    if v > best.get(isa, (0, ""))[0]:
        best[isa] = (v, r["kernel"], float(r["cov_percent"]))
print("      %-10s %10s  %-16s %8s" % ("isa", "MH/s", "best kernel", "CoV%"))
for isa in order:
    v, k, c = best[isa]
    print("      %-10s %10.2f  %-16s %7.2f%%" % (isa, v, k, c))

# Each rung against the one below it, which is the comparison the ladder exists
# to make. No prior figures are embedded here: they go stale, and one of them
# was silently wrong for a month.
PAIRS = [("sse2", "scalar"), ("avx2", "sse2"), ("avx512", "avx2"),
         ("neon", "scalar"), ("sve", "neon"), ("sve2", "sve")]
lines = ["%-8s / %-8s = %.2fx" % (hi, lo, best[hi][0] / best[lo][0])
         for hi, lo in PAIRS if hi in best and lo in best]
if not lines and len(order) > 1:
    hi, lo = order[-1], order[0]
    lines = ["%s / %s = %.2fx" % (hi.upper(), lo.upper(),
                                  best[hi][0] / best[lo][0])]
if lines:
    print("      %s" % ("-" * 52))
    for line in lines:
        print("      %s" % line)
PY
fi

# --------------------------------------------- C2: frequency under 512-bit load

say "C2  licence downclocking  (clock sampled during the run, not before it)"
if [ ! -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]; then
    note "no scaling_cur_freq -- cannot observe the clock. Most VMs do not"
    note "expose one; this phase wants bare metal."
else
    sample_freq() {                      # $1 = label, $2 = kernel
        local out="$OUT/p2-freq-$1.txt"
        ( while :; do
              cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
              sleep 0.25
          done ) > "$out" 2>/dev/null &
        local sampler=$!
        "$BIN" --kernel "$2" --threads 1 --samples "$SAMPLES" \
               --time-ms $(( FREQ_SECS * 1000 / SAMPLES )) --warmup-ms 200 \
               > "$OUT/p2-run-$1.txt" 2>&1
        kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
        # Median and a low percentile, not the mean. A licence downclock shows up
        # as a *tail* -- most samples at full clock, some below -- and the mean
        # is exactly the statistic a tail corrupts. On Sapphire Rapids the mean
        # read 3719 MHz against AVX2's 3800 and looked like a 2% downclock; the
        # median was 3800, identical, with 9% of samples in a tail. The count
        # below a threshold is what actually distinguishes the two.
        sort -n "$out" | awk -v label="$1" '
             { v[n++] = $1 }
             END {
                 if (!n) exit
                 med = v[int(n/2)]; p5 = v[int(n/20)]
                 lo = 0
                 for (i = 0; i < n; i++) if (v[i] < med * 0.97) lo++
                 printf "      %-10s median %.0f MHz   p5 %.0f   min %.0f   " \
                        "%d of %d below 97%% of median\n",
                        label, med/1000, p5/1000, v[0]/1000, lo, n
             }' | show
    }
    note "one core loaded, sampling cpu0 every 250 ms for ~${FREQ_SECS}s per kernel."
    note "scalar is the non-vector baseline; the question is whether the"
    note "widest kernel sits below it and below the next rung down."
    # The widest two rungs and the scalar one: the comparison is between a
    # vector kernel and something that cannot trigger a licence transition.
    for k in $(rungs 2); do
        sample_freq "$(echo "$k" | sed 's|.*/||;s|-s[0-9]*$||')" "$k"
    done
    SCALARK=$(python3 - "$OUT/c1-isa-ladder.csv" 2>/dev/null <<'PY'
import csv, sys
try:
    rows = [r for r in csv.DictReader(open(sys.argv[1]))
            if r["hashes_per_sec"] and "/scalar-" in r["kernel"]]
except OSError:
    rows = []
if rows:
    print(max(rows, key=lambda r: float(r["hashes_per_sec"]))["kernel"])
PY
)
    case " $(rungs 2) " in
        *" $SCALARK "*) ;;
        *) [ -n "$SCALARK" ] && sample_freq scalar "$SCALARK" ;;
    esac
    note "equal medians mean no steady-state licence penalty. A tail -- samples"
    note "below the median where the narrower kernel has none -- is a transient one."
fi

# ------------------------------------------------------------- C3: energy

say "C3  energy per hash, widest vector kernel against the next one down"
if [ "$HAVE_RAPL" = 0 ]; then
    note "no readable energy counter -- skipped. See the note in the environment"
    note "section above; on AMD this may be a naming mismatch rather than absence."
else
    # Two working sets per kernel, not one. C7 showed a 512-bit kernel losing 44%
    # once the corpus leaves the last-level cache, which means it spends that
    # time waiting on DRAM -- and a kernel waiting on memory is still burning
    # power. Whether stalling is cheaper or dearer per hash than computing is a
    # question no VM in this project could answer for want of RAPL.
    for k in $(rungs 2); do
        for ws in 1024 262144; do
            tag=$(echo "$k-$ws" | tr '/' '-')
            "$BIN" --kernel "$k" --threads 1 --working-set-kb "$ws" --json \
                   --samples "$SAMPLES" --time-ms "$TIME_MS" \
                   --warmup-ms "$WARMUP" > "$OUT/c3-energy-$tag.json" 2>&1
        done
    done
    python3 - "$OUT"/c3-energy-*.json <<'PY' | show
import json, sys
rows = []
print("      %-16s %8s %10s %12s %10s"
      % ("kernel", "WS MiB", "MH/s", "kH/J", "watts"))
for p in sys.argv[1:]:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    e = d.get("energy", {})
    if not e.get("available"):
        print("      %-16s %10.2f   (no energy: %.50s)"
              % (d["kernel"]["name"], d["result"]["median"]/1e6,
                 e.get("reason", "unknown")))
        continue
    ws = d["parameters"]["working_set_bytes"] / (1024.0 * 1024.0)
    rows.append((d["kernel"]["name"], ws, d["result"]["median"]/1e6,
                 e.get("hashes_per_joule", 0), e.get("cpu_package_watts", 0)))
    print("      %-16s %8.0f %10.2f %12.0f %10.1f"
          % (rows[-1][0], ws, rows[-1][2], rows[-1][3]/1e3, rows[-1][4]))

# Throughput and efficiency can disagree, and only one of them settles a
# purchasing question, so state both rather than leaving it to the eye.
print("      %s" % ("-" * 62))
by_ws = {}
for name, ws, mhs, hpj, w in rows:
    by_ws.setdefault(round(ws), []).append((name, mhs, hpj, w))
for ws in sorted(by_ws):
    v = sorted(by_ws[ws], key=lambda r: -r[1])
    if len(v) >= 2 and v[1][1] and v[1][2]:
        print("      at %4d MiB: %s / %s = %.2fx throughput, %.2fx efficiency"
              % (ws, v[0][0].split("/")[1], v[1][0].split("/")[1],
                 v[0][1] / v[1][1], v[0][2] / v[1][2]))
# The new question: is a kernel that stalls on DRAM cheaper or dearer per hash?
for name in sorted({r[0] for r in rows}):
    pair = sorted([r for r in rows if r[0] == name], key=lambda r: r[1])
    if len(pair) == 2 and pair[0][3] and pair[1][3]:
        small, big = pair
        print("      %-16s cache-resident to DRAM: %.2fx throughput, "
              "%.2fx efficiency, %.1f -> %.1f W"
              % (name.split("/")[1], big[2] / small[2] if small[2] else 0,
                 big[3] / small[3], small[4], big[4]))
PY
fi

# ------------------------------------------------------- C4: SHA-NI on this core

if [ "$HAVESHA" = 1 ]; then
    say "C4  SHA-NI against integer SIMD"
    note "what the unit is worth depends on the vector path beside it: measured"
    note "at 2.13x on Gracemont and 0.46x on Zen 5, the same instructions."
    KS="sha1/shani-s1,sha1/shani-s2,sha1/shani-s3,sha1/shani-s4,sha1/avx2-s3,sha1/avx2-s4"
    [ "$HAVE512" = 1 ] && KS="$KS,sha1/avx512-s2,sha1/avx512-s3"
    $SW --algorithm sha1 --kernel "$KS" --threads 1 \
        --csv "$OUT/c4-shani.csv" >> "$LOG" 2>&1
    [ -s "$OUT/c4-shani.csv" ] && python3 - "$OUT/c4-shani.csv" <<'PY' | show
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["hashes_per_sec"]]
ni   = max((float(r["hashes_per_sec"]) for r in rows if "shani" in r["kernel"]), default=0)
simd = max((float(r["hashes_per_sec"]) for r in rows if "shani" not in r["kernel"]), default=0)
for r in sorted(rows, key=lambda r: -float(r["hashes_per_sec"])):
    print("      %-16s %10.2f MH/s" % (r["kernel"], float(r["hashes_per_sec"])/1e6))
if ni and simd:
    print("      %s" % ("-" * 40))
    print("      SHA-NI / best SIMD = %.2fx" % (ni / simd))
    print("      above 1.0 the fixed-function unit wins, below it the vector")
    print("      path does. Both have been measured; neither generalises.")
PY
fi

# ------------------------------------- C5: algorithm baseline and the fingerprint

say "C5  all three algorithms: single thread, then all cores"
$SW --algorithm md5,sha1,sha512 --where cpu --threads 1 \
    --csv "$OUT/c5-1thread.csv" >> "$LOG" 2>&1
$SW --algorithm md5,sha1,sha512 --where cpu --threads "$NPROC" \
    --csv "$OUT/c5-allcores.csv" >> "$LOG" 2>&1
for f in c5-1thread c5-allcores; do
    [ -s "$OUT/$f.csv" ] && python3 - "$f" "$OUT/$f.csv" <<'PY' | show
import csv, sys
label, path = sys.argv[1], sys.argv[2]
print('      %s' % label)
with open(path, newline='') as fh:
    for r in csv.DictReader(fh):
        if r['hashes_per_sec']:
            print('        %-8s %-16s %10.2f MH/s  CoV %5s%%' % (
                r['algorithm'], r['kernel'],
                float(r['hashes_per_sec']) / 1e6, r['cov_percent']))
PY
done

note "checksum cross-check against the recorded machine-independent values:"
python3 - "$OUT/c5-1thread.csv" <<'PY' | show
import csv, sys
# The XOR fingerprint is invariant across machines, lanes, streams and threads.
# A mismatch here is a correctness event, not a performance one.
KNOWN = {"md5-full-55x1": "955e84cbbc05470019604a2bd9ff2821"}
for r in csv.DictReader(open(sys.argv[1])):
    w, c = r.get("workload"), r.get("checksum")
    if w in KNOWN:
        ok = "MATCH" if c == KNOWN[w] else "*** MISMATCH ***"
        print("        %-18s %s  %s" % (w, ok, c[:32]))
PY

# ----------------------------------------------- C6: does interleaving still pay

say "C6  stream interleaving, every algorithm on every ISA"
note "the largest single effect in the benchmark, and the best count is a"
note "property of the (algorithm, core) pair rather than a constant -- one part"
note "wanted s4/s2/s1 across md5/sha1/sha512. Sweep all of it."

# md5 alone used to stand for the whole benchmark here. It does not: on AArch64
# md5 climbs to s4 while sha512 is fastest at one stream and loses 21% by four.
# That was found by hand after the capture had finished, which is the argument
# for it being in the capture.
KS=""
for a in md5 sha1 sha512; do
    for isa in $ISAS; do
        for s in 1 2 3 4; do KS="$KS${KS:+,}$a/$isa-s$s"; done
    done
done
$SW --algorithm md5,sha1,sha512 --kernel "$KS" --threads 1 \
    --csv "$OUT/c6-streams.csv" >> "$LOG" 2>&1
[ -s "$OUT/c6-streams.csv" ] && python3 - "$OUT/c6-streams.csv" <<'PY' | show
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["hashes_per_sec"]]
by = {}
for r in rows:
    alg, rest = r["kernel"].split("/")
    isa, _, sn = rest.rpartition("-s")
    by.setdefault((alg, isa), {})[int(sn)] = float(r["hashes_per_sec"]) / 1e6
print("      %-8s %-8s %8s %8s %8s %8s   %s"
      % ("alg", "isa", "s1", "s2", "s3", "s4", "best"))
for (alg, isa), v in by.items():
    cells = "".join("%8.2f " % v[s] if s in v else "%8s " % "-"
                    for s in (1, 2, 3, 4))
    bs = max(v, key=lambda s: v[s])
    gain = v[bs] / v[1] if 1 in v and v[1] else 0
    flag = "" if bs == 4 else "   <-- not s4"
    print("      %-8s %-8s %s  s%d, %.2fx over s1%s"
          % (alg, isa, cells, bs, gain, flag))
PY

# ------------------------------------------------ C7: the two cheap axes

say "C7  message length and working set: the roofline axes"
note "MC/s should be flat against message length. Against working set it is"
note "flat only while the corpus fits -- a fast enough kernel asks for more"
note "bandwidth than the machine has, and the loss shows where the roof is."

# The two fastest rungs, not just the fastest. On a part where the widest ISA
# outruns DRAM and the next one down does not, one kernel cannot show that:
# the sweep records a collapse with no way to tell the machine from the kernel.
BESTK=$(python3 - "$OUT/c1-isa-ladder.csv" 2>/dev/null <<'PY'
import csv, sys
try:
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["hashes_per_sec"]]
except OSError:
    rows = []
best = {}
for r in rows:
    isa = r["kernel"].split("/")[1].split("-")[0]
    v = float(r["hashes_per_sec"])
    if v > best.get(isa, (0, ""))[0]:
        best[isa] = (v, r["kernel"])
top = sorted(best.values(), reverse=True)[:2]
print(",".join(k for _, k in top))
PY
)
if [ -n "$BESTK" ]; then
    note "using $BESTK"
    $SW --algorithm md5 --kernel "${BESTK%%,*}" --threads 1 \
        --message-bytes 8,24,55,56,119,247,503,1015,4087 \
        --csv "$OUT/c7-msgsize.csv" >> "$LOG" 2>&1
    # Out to 128 MiB, and with points either side of a typical last-level
    # cache, so a knee can be located rather than merely noticed.
    $SW --algorithm md5 --kernel "$BESTK" --threads 1 \
        --working-set-kb 64,1024,4096,16384,32768,49152,65536,131072 \
        --csv "$OUT/c7-workingset.csv" >> "$LOG" 2>&1
    python3 - "$OUT/c7-msgsize.csv" "$OUT/c7-workingset.csv" <<'PY' | show
import csv, sys
def load(p):
    try:
        return [r for r in csv.DictReader(open(p)) if r["hashes_per_sec"]]
    except OSError:
        return []
rows = load(sys.argv[1])
if rows:
    print("      %8s %6s %10s %10s" % ("bytes", "blocks", "MH/s", "MC/s"))
    mc = []
    for r in rows:
        c = float(r["compressions_per_sec"]) / 1e6
        mc.append(c)
        print("      %8s %6s %10.2f %10.2f"
              % (r["message_bytes"], r["blocks_per_message"],
                 float(r["hashes_per_sec"]) / 1e6, c))
    if mc:
        print("      MC/s spread across the range: %.1f%%"
              % ((max(mc) / min(mc) - 1) * 100))
rows = load(sys.argv[2])
if rows:
    # One column per kernel, so a knee that only the wider one hits is visible
    # as a knee rather than as noise.
    ks, by = [], {}
    for r in rows:
        k, w = r["kernel"], int(r["working_set_kb"])
        if k not in ks:
            ks.append(k)
        by.setdefault(w, {})[k] = (float(r["hashes_per_sec"]) / 1e6,
                                   float(r["message_bytes_per_sec"]) / 1e9)
    def ws(kb):
        return "%d MiB" % (kb // 1024) if kb >= 1024 else "%d KiB" % kb
    print("      %10s %s" % ("working set",
                             "".join("%16s" % k.split("/")[1] for k in ks)))
    for w in sorted(by):
        cells = "".join("%10.1f MH/s" % by[w][k][0] if k in by[w] else "%16s" % "-"
                        for k in ks)
        print("      %10s %s" % (ws(w), cells))
    for k in ks:
        vals = [by[w][k][0] for w in sorted(by) if k in by[w]]
        gbs  = [by[w][k][1] for w in sorted(by) if k in by[w]]
        if not vals:
            continue
        loss = (1 - min(vals) / max(vals)) * 100
        verdict = ("memory-bound past the knee" if loss > 15 else
                   "compute-bound throughout")
        print("      %-16s %5.0f%% from resident to DRAM, %.1f -> %.1f GB/s -- %s"
              % (k.split("/")[1], loss, max(gbs), min(gbs), verdict))
PY
else
    note "no C1 CSV to pick a kernel from -- skipped."
fi
fi   # DO_CPU

if [ "$DO_DEV" = 1 ]; then
# ------------------------------------------------- D1: the PCIe crossover

if [ "$NDEV" -gt 0 ]; then
    say "D1  PCIe crossover  (the question the device path exists to answer)"
    note "corpus re-uploaded before every launch, iterations swept."
    note "compute/transfer crosses 1.0 at N* -- below it the link binds,"
    note "above it the device does."
    #
    # 64-byte messages, deliberately, and the same length for all three.
    #
    # It is the length at which every algorithm stores exactly 128 bytes per
    # message once padded -- MD5 and SHA-1 spill into a second 64-byte block,
    # SHA-512 still fits one 128-byte block -- so the transfer side of the ratio
    # is identical across the three and only the compute side varies. That is
    # the controlled comparison the crossover needs.
    #
    # It is also the shortest length SHA-512 can iterate at, since the digest is
    # fed back over the head of the message. At the 55-byte default every
    # SHA-512 point with iterations > 1 is skipped and the sweep collapses to a
    # single row.
    #
    XMSG=64
    for alg in md5 sha1 sha512; do
        #
        # The CPU side of break-even is swept at *identical* parameters -- same
        # message length, same working set, same iteration ladder -- because
        # the two curves have to be crossable. They were not: the CPU baseline
        # in P3 runs at the default message length, a different working set and
        # a single iteration, so nothing in a captured session could answer
        # "when does offload beat the machine I already own".
        #
        # The best CPU kernel is found once rather than autotuned per point:
        # probing thirteen kernels at every rung of the ladder would cost more
        # than the ladder does, and the winner does not move with N. --where cpu
        # is what makes that probe a CPU answer on a box whose device kernel
        # would otherwise win it.
        #
        CPUK=$("$BIN" --algorithm "$alg" --where cpu --json \
                   --message-bytes "$XMSG" --working-set-kb 8192 \
                   --samples 3 --time-ms 40 --warmup-ms 100 2>/dev/null \
               | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["kernel"]["name"])
except Exception:
    pass' 2>/dev/null)
        [ -n "$CPUK" ] && note "$alg CPU baseline kernel: $CPUK"

        $SW --algorithm "$alg" --kernel "$alg/ocl-s1${CPUK:+,$CPUK}" \
            --transfer stream \
            --message-bytes "$XMSG" \
            --working-set-kb "$XFER_WS" --iterations "$ITERS" \
            --csv "$OUT/d1-crossover-$alg.csv" >> "$LOG" 2>&1
        if [ -s "$OUT/d1-crossover-$alg.csv" ]; then
            note "$alg:"
            python3 - "$OUT/d1-crossover-$alg.csv" "$REPO" <<'PY' | show
import csv, math, sys
sys.path.insert(0, sys.argv[2] + "/tools")
from sweep import balance_point, break_even

all_rows = list(csv.DictReader(open(sys.argv[1])))
# The CSV now carries both sides. N* is a device question; the CPU rows are
# there for break-even below.
rows = [r for r in all_rows if r.get("runs_on") == "device"]
cpu_rows = [r for r in all_rows if r.get("runs_on") == "cpu"]
print("      %6s %10s %10s %8s %s" % ("iters", "MH/s", "GB/s", "c/x", "bound by"))
for r in rows:
    try:
        ratio = float(r["compute_transfer_ratio"] or 0)
    except ValueError:
        ratio = 0.0
    print("      %6s %10.2f %10s %8.2f %s" % (
        r["iterations"], float(r["hashes_per_sec"]) / 1e6,
        r["transfer_gbytes_per_sec"] or "-", ratio, r["bound_by"]))

# Solve rather than bracket. Transfer is constant in the iteration count and
# kernel time is linear in it, so the balance point is an intercept, not a
# search. A geometric sweep alone would only ever answer to within its step.
fit = balance_point(rows)
if fit:
    print("      %s" % ("-" * 52))
    print("      N* = %.2f iterations   (compute-bound from %d up)"
          % (fit["n_star"], max(1, math.ceil(fit["n_star"]))))
    print("      %.3f ms/iteration + %.3f ms launch  vs  %.3f ms transfer"
          % (fit["per_iter_ns"]/1e6, fit["launch_overhead_ns"]/1e6,
             fit["transfer_ns"]/1e6))
    flag = ""
    if fit["r2"] < 0.98:
        flag = "   *** nonlinear -- do not quote N* ***"
    if fit["transfer_spread"] > 0.15:
        flag += "   *** transfer varied %.0f%% ***" % (fit["transfer_spread"]*100)
    print("      fit r2 = %.4f over %d points, transfer flat to %.1f%%%s"
          % (fit["r2"], fit["points"], fit["transfer_spread"]*100, flag))
else:
    print("      (not enough points to solve for N*)")

# Break-even: the purchasing question, which N* does not answer. N* says
# whether the bus is in the way; this says whether the device beats the host
# at all, transfers included.
if cpu_rows:
    def best(rs):
        by_n = {}
        for r in rs:
            try:
                n, hps = int(r["iterations"]), float(r["hashes_per_sec"])
            except (ValueError, KeyError):
                continue
            if hps > by_n.get(n, 0.0):
                by_n[n] = hps
        return by_n

    d, c = best(rows), best(cpu_rows)
    shared = sorted(set(d) & set(c))
    be = break_even([(n, d[n]) for n in shared],
                    [(n, c[n]) for n in shared]) if len(shared) > 1 else None
    if be:
        print("      %s" % ("-" * 52))
        cpuk = cpu_rows[0]["kernel"]
        thr = cpu_rows[0]["threads"]
        if be["verdict"] == "never":
            print("      break-even: never -- the device computes an iteration"
                  " slower than %s CPU threads on %s" % (thr, cpuk))
        elif be["verdict"] == "always":
            print("      break-even: offload wins at every iteration count"
                  " (vs %s on %s threads)" % (cpuk, thr))
        else:
            print("      break-even N = %.2f iterations   (offload pays from"
                  " %d up, vs %s on %s threads)"
                  % (be["n_break_even"], max(1, math.ceil(be["n_break_even"])),
                     cpuk, thr))
        print("      %.3f us/iteration device vs %.3f us CPU  (%.2fx compute"
              " advantage), %.3f us fixed device cost per hash"
              % (be["device_per_iter_s"]*1e6, be["cpu_per_iter_s"]*1e6,
                 be["compute_advantage"], be["device_fixed_s"]*1e6))
        r2d, r2c = be["device_fit"]["r2"], be["cpu_fit"]["r2"]
        print("      fit r2 = %.4f device, %.4f CPU over %d points"
              % (r2d, r2c, len(shared)))
        for w in be["warnings"]:
            print("      *** %s ***" % w)
else:
    print("      (no CPU rows -- break-even not computed)")
PY
        fi
    done
fi

# ------------------------------------- D2: resident baseline + stream counts

if [ "$NDEV" -gt 0 ]; then
    say "D2  resident baseline, all algorithms x all device stream counts"
    note "tests whether 'one stream is right on a device' survives a change of"
    note "vendor and OpenCL compiler, or was an Intel iGPU artifact."
    KS=""
    for alg in md5 sha1 sha512; do
        for s in 1 2 3 4; do KS="$KS${KS:+,}$alg/ocl-s$s"; done
    done
    $SW --algorithm md5,sha1,sha512 --kernel "$KS" --transfer resident \
        --working-set-kb 65536 --csv "$OUT/d2-device-streams.csv" >> "$LOG" 2>&1
    [ -s "$OUT/d2-device-streams.csv" ] && \
        python3 - "$OUT/d2-device-streams.csv" <<'PY' | show
import csv, sys
with open(sys.argv[1], newline='') as fh:
    for r in csv.DictReader(fh):
        print('      %-16s %10.2f MH/s  CoV %5s%%' % (
            r['kernel'], float(r['hashes_per_sec']) / 1e6, r['cov_percent']))
PY
fi

# ------------------------------------------------- D3: memory / working set

if [ "$NDEV" -gt 0 ]; then
    say "D3  working-set sweep on the device (memory axis)"
    $SW --algorithm md5 --kernel md5/ocl-s1 --transfer resident \
        --working-set-kb "$WS_AXIS" --csv "$OUT/d3-workingset.csv" >> "$LOG" 2>&1
    note "wrote d3-workingset.csv"
fi

# ------------------------------------------- D5: the compute plateau, resident

if [ "$NDEV" -gt 0 ]; then
    say "D5  iteration ladder, resident (where the device stops being bandwidth-bound)"
    note "D1 sweeps iterations with the transfer inside the timed region, which"
    note "answers where compute overtakes the link. This asks a different"
    note "question: with the corpus already resident, how much compute per byte"
    note "does the device need before it stops being fed by memory?"
    note "Hash rate falls monotonically here by construction -- one hash is N"
    note "chained hashes -- so the figure to read is compressions/sec, which"
    note "climbs while memory binds and flattens once compute does. The knee is"
    note "the answer: 'X compressions/sec sustained, once iterations exceed Y'."
    #
    # Two working sets, because a plateau at one size cannot tell a compute
    # ceiling from a corpus that fits in cache. 16 MiB fits an A100's 40 MiB L2
    # and 256 MiB does not, so if the knee moves it was cache; if it does not,
    # it is the device.
    #
    # The ladder stops at 128 on purpose. Once compute binds, the curve is flat
    # and further rungs add nothing -- while cost per point grows with
    # iterations, because a single pass over the corpus cannot be subdivided.
    # D1 learned that the expensive way: it estimated 78 s for 30 points and
    # spent 3704, with one point killed at the timeout.
    D5_LADDER='1:8:+1,16,32,64,128'
    D5_WS=16384,262144
    #
    # Two legs, because the knee is not a property of the iteration count
    # alone. It is compressions per byte read, and blocks_per_message is the
    # other term: 55 bytes is one MD5 block and 64 bytes is two, since padding
    # needs nine bytes and spills. On an A100 the 55-byte md5 ladder rose 55%
    # from one iteration to its plateau while the 64-byte one rose 5% --
    # already past the knee at the first point. One message size would have
    # shown one of those and implied it was the whole story.
    #
    # Leg 1 finds the knee, at one block per message. SHA-512's digest is 64
    # bytes and --iterations needs message >= digest, so it cannot appear here.
    $SW --algorithm md5,sha1 --kernel md5/ocl-s1,sha1/ocl-s1 --transfer resident \
        --message-bytes 55 --iterations "$D5_LADDER" --working-set-kb "$D5_WS" \
        --csv "$OUT/d5a-knee-1block.csv" >> "$LOG" 2>&1
    note "wrote d5a-knee-1block.csv  (the knee, where it is visible)"
    #
    # Leg 2 puts all three algorithms on the same footing at two blocks.
    $SW --algorithm md5,sha1,sha512 \
        --kernel md5/ocl-s1,sha1/ocl-s1,sha512/ocl-s1 --transfer resident \
        --message-bytes 64 --iterations "$D5_LADDER" --working-set-kb "$D5_WS" \
        --csv "$OUT/d5b-plateau.csv" >> "$LOG" 2>&1
    note "wrote d5b-plateau.csv  (per-algorithm sustained rate)"
fi

# ------------------------------------------------------- D4: multi-device

if [ "$NDEV" -gt 1 ]; then
    say "D4  multi-device  ($NDEV devices)"
    note "the checksum must match the single-device value exactly: equality"
    note "proves the slices tile the corpus with no gaps and no overlaps."
    #
    # The corpus is held constant PER DEVICE, so the total grows with the device
    # count. --working-set-kb is a total everywhere else, and splitting a fixed
    # total starves each device as more are added: on 2xH100 that put 32 MiB on
    # each, where a GPU needs ~256 MiB to saturate, and the resulting 1.70x read
    # like a scaling limit when it was an underfed slice.
    #
    # It also makes the more interesting measurement. Fixed-per-device means
    # total PCIe traffic scales with the device count, so if the host bus is the
    # constraint, aggregate throughput stops scaling and says so. A fixed total
    # cannot distinguish that from starvation.
    D4_WS=65536
    D4_WS_ALL=$(( D4_WS * NDEV ))
    note "corpus held at ${D4_WS} KiB per device: ${D4_WS} for one, ${D4_WS_ALL} for $NDEV"
    "$BIN" --json --kernel md5/ocl-s1 --device 0 --working-set-kb "$D4_WS" \
        --samples "$SAMPLES" --time-ms "$TIME_MS" > "$OUT/d4-one-device.json" 2>>"$LOG"
    "$BIN" --json --kernel md5/ocl-s1 --device all --working-set-kb "$D4_WS_ALL" \
        --samples "$SAMPLES" --time-ms "$TIME_MS" > "$OUT/d4-all-devices.json" 2>>"$LOG"
    #
    # A third run, for the checksum claim alone. The two above deliberately use
    # different corpus sizes -- that is what holds the per-device corpus
    # constant and makes the throughput comparison mean something -- but the
    # XOR fold is invariant across devices for a *fixed* corpus, not across
    # corpus sizes. Comparing those two therefore printed "slicing is wrong" on
    # every multi-GPU host, for slicing that was correct, and sent an operator
    # hunting a phantom on a rented machine. Device 0 at the full corpus is the
    # right baseline for equality.
    "$BIN" --json --kernel md5/ocl-s1 --device 0 --working-set-kb "$D4_WS_ALL" \
        --samples 2 --time-ms 40 > "$OUT/d4-one-device-fullcorpus.json" 2>>"$LOG"
    python3 - "$OUT/d4-one-device.json" "$OUT/d4-all-devices.json" \
             "$OUT/d4-one-device-fullcorpus.json" <<'PY' | show
import json, sys
try:
    a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
except Exception as e:
    print("      could not compare: %s" % e); raise SystemExit
cb = b["verification"]["checksum"]
try:
    ca = json.load(open(sys.argv[3]))["verification"]["checksum"]
except Exception:
    ca = None
print("      1 device   %8.2f MH/s" % (a["result"]["median"]/1e6))
print("      %d devices  %8.2f MH/s  %s" % (b["device"]["device_count"] if "device_count" in b.get("device",{}) else 0,
                                            b["result"]["median"]/1e6, cb[:16]))
if ca is None:
    print("      checksum  baseline run failed; equality not checked")
else:
    print("      checksums %s  (both at the full corpus)"
          % ("MATCH -- the slices tile the corpus" if ca == cb
             else "*** DIFFER -- slicing is wrong ***"))
n = b.get("device", {}).get("device_count", 0) or 1
sc = b["result"]["median"] / a["result"]["median"]
print("      scaling   %.2fx over %d devices (%.0f%% of linear), corpus held"
      " constant per device" % (sc, n, 100.0 * sc / n))
if n > 1 and sc < 0.9 * n:
    print("      the shortfall is not starvation -- each device has the same")
    print("      corpus it had alone -- so look at the host link or the launch")
    print("      path. Total transfer scaled %dx with the device count." % n)
PY
fi
fi   # DO_DEV

# ---------------------------------------------------------------- provenance

say "packaging"
{
    echo "valubench session"
    echo "commit:   $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
    # -d @SECONDS is GNU date; BSD date spells it -r SECONDS. Without the
    # second form a Mac fell through to the bare `date -u`, and every capture
    # recorded that it started at the moment it ended.
    echo "started:  $(date -u -d "@$START" 2>/dev/null || date -u -r "$START" 2>/dev/null || date -u)"
    echo "ended:    $(date -u)"
    echo "quick:    $QUICK"
    echo "only:     ${ONLY:-both}"
    echo "cpus:     $NPROC"
    echo "devices:  $NDEV"
    echo "xfer ws:  $XFER_WS KiB"
} > "$OUT/session.txt"
ls -la "$OUT" | show

say "complete"
