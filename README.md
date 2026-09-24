# valubench

[![build and check](https://github.com/silent-runner/valubench/actions/workflows/ci.yml/badge.svg)](https://github.com/silent-runner/valubench/actions/workflows/ci.yml)
[![license: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

An integer SIMD microbenchmark. It measures how fast hardware executes the
general-purpose integer vector path, using hashing as the vehicle, and verifies that
the hardware computed the right answer while doing it.

BSD 3-Clause. Builds with a C11 compiler and make. No configure
step, no network access at build or run time, and the binary links only libc —
OpenCL and NVML are `dlopen`'d, so the same build runs with or without a GPU.

```
sudo apt install build-essential      # that is the whole requirement for CPU
make                                  # or: make CC=clang
./build/valubench
```

GPU and energy support need a few more packages — see
[docs/dependencies.md](docs/dependencies.md), which has copy-paste blocks for fresh
Ubuntu/Debian and RHEL/Fedora/Amazon Linux cloud instances.

## What it is for

**Deciding things about hardware you are considering, or hardware you already
have, when the answer depends on integer vector work.** Floating-point
benchmarks are abundant and say nothing about the integer path; this measures
that path and where it stops being the constraint.

Four questions it answers with numbers rather than reasoning:

- **What is a wider vector unit worth here?** Not in theory — measured, rung by
  rung, from scalar to AVX-512 or NEON. Doubling the register width is worth
  1.93x on one core and 1.06x on another, because the second cracks every
  256-bit operation into halves. Same source, same instructions.
- **Is a dedicated accelerator worth using?** A CPU's fixed-function SHA unit
  has measured anywhere from 2.13x *faster* than the vector path beside it to
  0.21x as fast — a 10x spread. On every part with a capture it is the
  *slower* of the two, and how much slower depends entirely on the vector path
  beside it. Which one you have decides whether using it is a win or a 79%
  loss.
- **When does offloading to a GPU pay?** Two separate numbers, because they
  disagree: N\* is where compute overtakes the PCIe transfer, and break-even is
  where the accelerator beats the whole host CPU. On an A10 those were 139 and
  42 iterations, so there is a wide band where the bus binds and offloading is
  still right.
- **Where does the memory system take over?** A fast enough kernel outruns DRAM
  on the same data a narrower one does not — AVX-512 losing 44% past the
  last-level cache while AVX2 loses 3%, and burning 62% *more* power to do it.

**And it answers them about the machine in front of you**, which is the point.
Every figure above came from running this on a specific part; none of them
generalise, and several reverse between machines. That is the case for measuring.

## Why it exists

Because the numbers people quote about integer SIMD are usually inherited rather
than measured, and they do not survive contact with a second machine. Every
conclusion in this project's own history that was drawn from one part has since
been overturned by the next one.

So the design is built around not fooling yourself:

- **Correctness is a gate, not a footnote.** Every run verifies its digests
  against an independent scalar reference and reduces them to a fingerprint that
  is invariant across lanes, streams, threads, devices and instruction sets. The
  same value comes back from Gracemont, Cascade Lake, Zen 5, Milan, Ice Lake,
  Sapphire Rapids, Neoverse V1 and V2, an Intel iGPU, an A10, an A100 and two
  H100s. A run that cannot verify produces no number.
- **Dispersion is reported, and a noisy result says so** — in the output and in
  the exit code.
- **The environment is captured** with every result: CPU, ISA path actually
  taken, governor, clock, compiler, thread count, device and driver. A number
  without its machine is not comparable to anything, and on one part and one
  instruction set four compilers spanned 4.15x — more than any architectural
  difference this project has measured.
- **One binary, runtime dispatch, no `-march=native`.** The build cannot depend
  on the machine that produced it.

## Algorithms

Three, selected with `--algorithm`:

| | digest | block | word | endian | why it is here |
|---|---|---|---|---|---|
| `md5` *(default)* | 4 x 32 | 512 bit | 32 bit | little | no hardware accelerator anywhere |
| `sha1` | 5 x 32 | 512 bit | 32 bit | big | expands its schedule; a different instruction mix |
| `sha512` | 8 x 64 | 1024 bit | 64 bit | big | 64-bit words and a 128-bit length field |

They span the axes that break naive generalisation on purpose. SHA-512 is the
one that forced the harness to be genuinely algorithm-agnostic: adding only
SHA-1 would have left a 32-bit word and a 64-byte block hardcoded, because
SHA-1 shares MD5's geometry.

One invariant survived and the corpus layout rests on it: **every block is
sixteen words**, whatever the word size.

SHA-1 costs more than MD5 because it runs 80 rounds and *expands* its sixteen
message words to eighty rather than permuting them. SHA-512 costs more again:
80 rounds of 64-bit work with four sigma functions, at half the lanes per
register.

All three have OpenCL kernels, which makes one comparison possible that a
single-algorithm benchmark would hide — **the GPU's advantage is not uniform**.
Both sides slow down on SHA-512 — AVX2's lanes halve at 64 bits too — but the
GPU gives up substantially more of its relative footing, because consumer GPU
ALUs are 32-bit and every 64-bit add and rotate is emulated. "Is the accelerator
worth it" has no single answer even on fixed hardware:
the measured ratios.

## Why MD5 is the default

Not for its cryptographic properties — it has none left. MD5 is used because
**no hardware has MD5 instructions.**

Benchmarking SHA-1 or SHA-256 on a modern CPU measures the SHA-NI fixed-function
unit, not the integer SIMD ALUs — on this machine that unit is worth more than
double the best AVX2 path (measured), and it is the reason
`--algorithm sha1` ships both kernels side by side rather than one. MD5 has no such accelerator on any architecture,
so it is forced onto the general integer vector path everywhere, which is exactly
what this benchmark is trying to measure. It also happens to lean on the specific
integer capabilities that separate ISA generations — 32-bit rotate and 3-input
boolean logic — which makes it unusually good at exposing them.

> Sample output blocks in this file show the *shape* of what the tool prints.
> Their numbers are illustrative and not maintained. Run it on your own machine;
> that is the only number that describes your machine.

## What the number means

Workload id `md5-full-LxN`: messages of L bytes (`--message-bytes`), N chained
MD5s each (`--iterations`). A message of L bytes occupies `ceil((L+9)/64)` blocks
once padded, and each block is one compression, so

```
compressions/sec = hashes/sec x iterations x blocks_per_message
```

The default is `md5-full-55x1`: 55 bytes is the largest message fitting in a
single block once padded, so one hash is exactly one compression.

All 64 steps run with real message words. There is deliberately **no** constant
folding of `K + w[i]`, no step reversal against a target digest, and no early
exit.

> **Not comparable to figures that take those shortcuts.** An implementation
> using any of the three performs strictly less work per hash it reports, so its
> number is larger and answers a different question. Both are valid measurements;
> they are not the same measurement. See [docs/research.md](docs/research.md) §2.2.
## Verifying the answer, not just timing it

Every kernel XORs together each digest it computes. XOR is commutative and
associative, so that checksum is **invariant to how the work was distributed** —
across lanes, streams, threads, or devices. One expected value therefore
validates every implementation, and the same value appears at any thread count
on any machine.

That gives three things for the price of one:

- **A correctness gate.** Each kernel is checked against an independently
  written scalar reference before it is timed, and re-checked on every timed
  iteration. A mismatch suppresses the result rather than reporting a fast wrong
  answer.
- **A cross-machine fingerprint.** `md5-full-55x1` produces `955e84cb…` on
  Gracemont, Cascade Lake and Zen 5 alike. If two machines disagree, one of them
  is broken.
- **Honest error bars.** Results are medians with a coefficient of variation,
  and the tool exits non-zero when a run is too noisy to trust.

The scalar references are deliberately separate implementations, written from
RFC 1321 and FIPS 180-4 rather than shared with the kernels, so that agreement
between them is evidence rather than a tautology.

## Documentation

| | |
|---|---|
| [docs/guide.md](docs/guide.md) | Full usage: the three axes, sweeping, comparing runs, GPUs, energy, the PCIe crossover |
| [docs/design.md](docs/design.md) | What this measures, why MD5, and what measurement changed about the plan |
| [docs/research.md](docs/research.md) | The decision log — every design choice with its reasoning, including the ones that turned out wrong |
| [docs/results.md](docs/results.md) | Working with captured results: the database, and what is authoritative |
| [docs/schema.md](docs/schema.md) | The JSON and CSV output contracts, and the compatibility rule |
| [docs/dependencies.md](docs/dependencies.md) | Per-distribution packages, and what each is for |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Building, testing, and adding a kernel or an algorithm |

## Status

**Validated on nine CPU microarchitectures and four GPUs**, all producing the
same verification fingerprints: Gracemont, Cascade Lake, Zen 5 (server and
desktop), Milan, Ice Lake-SP, Sapphire Rapids, Neoverse V1 and Neoverse V2, an
Intel iGPU, an NVIDIA A10, an A100 and two H100s.
Scalar, SSE2, AVX2, AVX-512, SHA-NI, NEON, SVE and SVE2 CPU kernels; OpenCL
device kernels for all three algorithms; resident and streaming transfer;
multi-device; autotune; statistics; energy where counters allow; JSON and human
output. GCC 13.3 through 16.2 and Clang 18.1 through 22.1 all build clean under
the full warning set and produce identical checksums, and CI additionally
cross-compiles for AArch64 and runs the NEON kernels under emulation.

Known gaps, in the order they matter:

- **SVE2 is measured only at 128 bits.** The kernels exist and are validated on
  Neoverse V1 and V2, but no part yet offers SVE2 above 128 bits, so vector
  width and instruction-set generation stay conflated on that side.
- **AMD GPUs are untested.** NVIDIA and Intel are validated; ROCm and Mesa
  Rusticl have never run this.
- **Every transfer figure is pageable memory.** An A10 sustained 10.9 GB/s over
  PCIe 4.0 x16, roughly half what pinned staging would achieve, which moves the
  crossover by about that factor.
- **Overlapped transfer and compute.** Streaming uploads then launches, in
  order. The reported ratio already answers the pipelined question, so this
  concerns achieved throughput rather than correctness of the ratio.
- **Nothing pins the toolchain, and the toolchain is the largest effect in the
  project.** Four compilers span 4.15x on one part and one instruction set,
  against at most 1.20x from any instruction-set choice measured anywhere. Even
  on a settled path the compiler changes *which kernel wins*, so autotune's
  pick is toolchain-dependent. Results are comparable within a compiler and not
  across one.

## Layout

| Path | What |
|---|---|
| [src/reference/](src/reference/) | Scalar references from RFC 1321 and FIPS 180-4; the correctness oracles every kernel is checked against |
| [src/kernels/cpu/](src/kernels/cpu/) | CPU kernels, one translation unit per ISA — see its [README](src/kernels/cpu/README.md) for how to add one |
| [src/kernels/cpu/md5_kernel_impl.h](src/kernels/cpu/md5_kernel_impl.h) | The multi-way kernel, written once |
| [src/kernels/gpu/](src/kernels/gpu/) | Device kernels: complete, self-contained OpenCL |
| [src/opencl/](src/opencl/) | The host-side OpenCL driver — loader, context, upload, launch. No hash functions |
| [src/bench.c](src/bench.c) | Validation, autotune, timing, statistics |
| [tools/sweep.py](tools/sweep.py) | Walks a parameter grid, writes CSV, solves for the PCIe balance point |
| [tools/compare.py](tools/compare.py) | Diffs two result sets, gated on the verification checksum |
| [tools/run.sh](tools/run.sh) | One-command capture for a time-boxed session on rented hardware, CPU and device |
| [docs/research.md](docs/research.md) | Background research and every design decision, with rationale |
| [docs/design.md](docs/design.md) | What this measures and why, and what measurement changed about the plan |
| [docs/dependencies.md](docs/dependencies.md) | Packages per distro, and the files they must provide |

```
make check           # known-answer vectors + every kernel against its reference
make config          # show what this toolchain can build
make CC=clang        # build with Clang/LLVM instead of GCC
```

`-flto` is deliberately never enabled: it could let the compiler prove
the message corpus is constant and fold message words into the round constants,
the shortcut this workload excludes.

## Licensing note

The MD5 core is written from the RFC 1321 specification, and the round constants
are generated from `T[i] = floor(2^32 * abs(sin(i)))` rather than transcribed. No
code is copied from any existing MD5 implementation. That includes the RFC 1321
reference implementation, which carries an RSA notice, and permissive-licensed
implementations, which require their notice be retained in derivatives. The
project was written under a public-domain dedication, where neither obligation
could be accepted; it is BSD 3-Clause now, which could accept them, but the
code is already free of both. See
[docs/research.md](docs/research.md) §2.9.
