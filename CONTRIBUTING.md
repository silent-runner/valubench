# Contributing

## Building and testing

```bash
sudo apt install build-essential     # the whole requirement to build and run
sudo apt install python3             # make check only; stdlib, no pip packages
make                                 # or: make CC=clang
make check                           # known-answer vectors + every kernel
make config                          # what this toolchain can build
```

`make check` runs twelve checks. The two that everything rests on are the
published test vectors (RFC 1321, FIPS 180-4) against the scalar references,
and every registered kernel against those references across message sizes,
block boundaries, lane counts and iteration counts. The rest guard the harness
around them: pinning, thread-start failure, the JSON and exit-code contract,
working-set sizing, scalar purity and the power model among them. All must pass
with zero failures before a change is worth reviewing; the `check:` line in the
Makefile is the authoritative list.

## Rules that are not negotiable

These exist because breaking one produces a *plausible wrong number* rather than
an error, which is the failure mode this project is built to avoid.

- **No `-march=native`, ever.** It makes results depend on the machine that
  compiled the binary. Each ISA is a separate translation unit with its own `-m`
  flags, selected at run time by CPUID.
- **No `-flto`.** Link-time optimization could prove the message corpus constant
  and fold message words into the round constants — precisely the shortcut this
  workload exists to exclude.
- **Streams are expanded, never looped.** A loop that failed to unroll would
  collapse the interleaved dependency chains into one and under-report the
  hardware several-fold, silently. The same applies to step lists.
- **Round functions are written per ISA, not once generically.** AVX-512
  collapses each to a single `vpternlogd`; a shared "generic" form leaves that
  on the table, and an optimization that helps one ISA can hurt another.
- **Every kernel is validated against the scalar reference**, which is an
  independent implementation written from the specification. Agreement between
  them is evidence only because they share no code.
- **Measured numbers are tracked outside this repository**, and prose should
  characterise results qualitatively — "more than double the best SIMD path"
  survives a re-measurement; "2.13x" does not. A figure that genuinely carries
  an argument must name its machine, compiler and date, because none of those
  are recoverable later.

## Adding a kernel or an algorithm

[src/kernels/cpu/README.md](src/kernels/cpu/README.md) is the reference. Adding
an ISA is three steps; adding an algorithm is about ten and they are enumerated
there. The kernel matrix in `src/kernels/cpu/matrix.h` is the single source of
truth — registry rows and forward declarations are expanded from it, so
`src/registry.c` is never edited by hand.

**A vector-length-agnostic ISA is not three steps.** SVE and SVE2 take their
width from the hardware, and their types are *sizeless*: `svuint32_t a[4];` is a
hard compile error, as is `sizeof`, a struct member, or anything else that needs
a width at compile time. The shared kernel templates declare per-stream state as
exactly those arrays, so an ISA like this cannot use them as written.

The way through is the hook system: each template wraps its state declarations
and accessors in `#ifndef` blocks whose defaults are the original code, and
`sve_hooks.h` and friends override them with individually named scalars and
memory-backed schedule windows. Two consequences worth knowing before you start:

- **The defaults must reproduce the original code exactly.** The x86 objects are
  byte-identical with and without the hooks, and that is checked per symbol. If
  you touch a template, verify it that way.
- **The lane count is a run-time value.** It comes from `svcntw()`/`svcntd()`,
  so a kernel registers `lanes = 0` at build time and the registry fills it in.
  Anything that assumes a width — a buffer size, an offset, a loop bound — is a
  bug that only appears on hardware you probably do not have.

## Verifying a change did what you think

Two techniques this project leans on, worth knowing:

- **Checksum invariance.** The XOR fingerprint is invariant to lanes, streams,
  threads and devices, so `md5-full-55x1` must produce `955e84cb…` before and
  after any change that is not supposed to alter results.
- **Disassembly comparison.** For refactors that should not change generated
  code — renames, macro restructuring — compare `objdump -d` before and after
  with symbol names normalised. This has caught a silently scalar-only build
  that compiled cleanly and ran.

## Testing a kernel for an architecture you do not have

The ARM kernels were written, compiled and validated on an x86 machine. The
recipe generalises to any target with a cross toolchain and a qemu-user build:

```bash
sudo apt install -y gcc-aarch64-linux-gnu qemu-user-static
export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu     # where the ARM loader lives
make CC=aarch64-linux-gnu-gcc HOSTCC=cc BUILD=build-arm64 check
```

Three details make it work. `HOSTCC` keeps the build tools native, since
`embed_cl` has to run on the machine doing the building. `BUILD` keeps the
cross objects out of the native ones. And the architecture comes from
`$(CC) -dumpmachine` rather than `uname -m`, so the build configures for the
target rather than the host.

Emulated **throughput is meaningless** — do not record it. Emulated
**correctness is not**: the checksum is invariant across lanes, streams,
threads, devices *and architectures*, so a kernel that produces the same value
under qemu as on x86 is computing the right thing. That is the property worth
testing this way, and CI does it on every push.

**For a run-time-width ISA, vary the width.** qemu will pretend to any of them:

```bash
qemu-aarch64-static -cpu max,sve-max-vq=1 ./build-arm64/valubench ...   # 128-bit
qemu-aarch64-static -cpu max,sve-max-vq=3 ./build-arm64/valubench ...   # 384-bit
```

384 bits earns its place precisely because no hardware has it — 12 lanes is
where a power-of-two assumption shows up, and 12 x 3 streams is the case that
must decline to run rather than produce a wrong answer. Note that Linux caps a
process at 512 bits by default and qemu emulates the cap, so `sve-max-vq=8`
silently gives you 512 and tests nothing new; reaching 1024 or 2048 needs
`prctl(PR_SVE_SET_VL)`.

**And test the CPU that does not have the ISA at all.**

```bash
qemu-aarch64-static -cpu neoverse-n1 ./build-arm64/valubench --list
```

This is the configuration where a vector-length-agnostic build fails before it
computes anything, so no checksum test can catch it. The dispatcher resolved
lane counts by calling `svcntw()` on every SVE row without checking
`available()` first — `CNTW` is an SVE instruction, and the binary died with
SIGILL at startup on Graviton2, Ampere Altra, every Raspberry Pi and every
Cortex-A5x/A7x. The whole suite passed, because qemu's default CPU has SVE and
nothing ran without it.

Two traps in the obvious fix, both real:

- A kernel that cannot run keeps `lanes = 0`, and **AArch64 `UDIV` by zero
  yields 0 rather than trapping**, so `768 % 0` evaluates to 768 and a group-size
  check reports a misconfigured kernel instead of crashing. The same expression
  on x86 is SIGFPE.
- `HWCAP` is not always self-consistent. `-cpu max,sve=off` clears `HWCAP_SVE`
  and leaves `HWCAP2_SVE2` set, which is impossible on real silicon since
  FEAT_SVE2 implies FEAT_SVE. Believe the weaker claim.

## Measuring on rented hardware

`tools/run.sh` captures a whole session in one command — CPU phases, then
device phases if the machine has an OpenCL device. Several failures are worth
pre-empting before either is worth running, all learned the expensive way.

**Stop the machine patching itself.**

```bash
sudo systemctl disable --now unattended-upgrades
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer
```

A fresh Ubuntu image runs a background upgrader that can install a kernel and
reboot the instance out from under a run. Two instances did exactly that within
minutes of first login. An instance is rented for hours and then destroyed, so
it gains nothing from unattended patching and can lose a session to it.
`run.sh` records whether the service is live, because a machine that reboots
mid-run looks like a network fault from the other end.

**Do not poll the machine with bare TCP probes, and reuse one SSH connection.**

```
Host <instance>
    ControlMaster   auto
    ControlPath     /tmp/cm-%r@%h:%p     # must stay under 108 bytes
    ControlPersist  30m
```

Recent OpenSSH enables `PerSourcePenalties` by default: an address that
repeatedly opens connections and abandons them without authenticating is dropped
silently for an escalating period, up to about ten minutes. A loop of `nc -z` or
`/dev/tcp` reachability checks is exactly that pattern, and the symptom is
indistinguishable from a broken network — timeouts rather than refusals,
recovering on its own. The `ControlPath` limit is not advisory either: a Unix
socket path of 108 bytes or more fails at connection time, and temp directories
routinely exceed it.

**Energy needs the RAPL driver, and AWS kernels ship without it.** On bare
metal, `/sys/class/powercap` may not exist at all — the failure is silent, and
the energy phase simply reports no counter:

```bash
[ -d /sys/class/powercap ] || sudo apt install -y "linux-modules-extra-$(uname -r)"
sudo modprobe intel_rapl_msr
sudo chmod a+r /sys/class/powercap/*/energy_uj
```

Do it **before** starting a capture: the run decides once, during its
environment phase, whether energy is available, so loading the driver mid-run
does not help.

**The PMU needs `perf_event_paranoid` lowered, and Ubuntu ships it at 4.**

```bash
sudo apt install -y linux-tools-common linux-tools-generic "linux-tools-$(uname -r)"
sudo sysctl -w kernel.perf_event_paranoid=1
perf stat -e cycles true          # confirm before the capture, not after
```

Counters work on virtualised Graviton, so this does not need bare metal. A
phase that silently skips itself is worse than one that fails, so check the
probe rather than the exit status of the install.

**Emit the sweep schema, or the capture will not be readable.** `tools/sweep.py`
writes the CSV that `tools/ingest.py` reads, with provenance on every
row. A hand-rolled CSV is skipped at ingest, and a skipped file looks exactly
like a capture nobody took. See [docs/results.md](docs/results.md).

**Record the compiler beside every number, in the file that holds the number.**
Not in a sibling log. On one Graviton4 kernel gcc 13.3 and gcc 15.2 differ by
10% at one thread and agree to within 0.3% at sixteen, which reads as a
scaling defect until you know which compiler produced which row. This is the
mistake that made the project's previous results file unusable, and it has been
made twice since.

**Run detached, and copy results off as they land** rather than in one transfer
at the end. The script writes its log to disk before the terminal and tars the
output directory on exit, including on failure, so a session survives losing the
connection — but a capture you cannot retrieve is worth nothing.

## Style

C11, four-space indent, no tabs. `//` for single-line comments, `/* */` for
multi-line blocks. Comments explain *why*; the code already says what.

## Licensing

BSD 3-Clause. Every source file carries an SPDX identifier and a copyright line:

```c
/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 */
```

New files need both. Generated headers do not — they carry the notice of the
generator that emits them. By contributing you agree your work ships under that
licence.

## AI-assisted contributions

**Allowed and welcome.** Much of this project was written with AI assistance and
the repository does not distinguish between contributions on that basis.

What is required is the same for everyone: the change passes `make check`, it
does not violate the rules above, and you understand it well enough to defend
the design in review. Generated code that nobody can explain is the problem, not
generated code as such.

If you use an assistant, [AGENTS.md](AGENTS.md) carries the project conventions
in a form it can load directly.
