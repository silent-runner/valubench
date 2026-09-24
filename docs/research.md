# RESEARCH

Background research for `valubench`, conducted before any code was written. Two
questions drove it:

1. What makes open-source benchmarks like the Phoronix Test Suite widely useful?
2. How are fast MD5 implementations built, and which of their techniques belong
   in a benchmark?

A third topic emerged as unavoidable while researching the project's second
goal -- finding the compute-bound / bandwidth-bound crossover, stated precisely
in [design.md](design.md) -- so it is documented here too.

---

## Part 1 — What makes a benchmark widely useful

The Phoronix Test Suite (PTS) is not the fastest or most rigorous benchmark
harness in existence. It is the most *used* one on Linux. The reasons are almost
entirely about packaging, not measurement.

### 1.1 What PTS gets right, condensed

The Phoronix Test Suite is not the most rigorous benchmark harness; it is the
most *used* one on Linux, and the reasons are about packaging rather than
measurement. The parts worth stealing, each of which shaped a decision here:

- **Units and direction of goodness are declared, not implied.** A consumer that
  cannot tell whether higher is better cannot rank results. Both are fields in
  our JSON.
- **Machine-readable output is primary.** PTS scrapes stdout with a fragile
  regex template, and that is the single largest source of broken profiles.
  Our JSON exists so no wrapper ever has to parse our prose.
- **Variance is handled, not wished away.** PTS re-runs until the standard
  deviation falls under a threshold. We report min/median/mean/stddev/CoV and
  the raw samples, and say so when a result is too noisy to carry the 10%
  significance bar.
- **The environment is captured automatically.** A number without a machine
  description is not comparable to anything, which is what makes a results
  database possible at all.
- **The workload is versioned.** When the work changes, the identifier changes,
  and old numbers stop being comparable. This matters more here than for PTS,
  because our whole premise is that 10% is significant.
- **One command, no configuration, no network.** PTS installs dependencies per
  distro, which assumes internet access; we cannot. What transfers is the feel:
  one command, everything else auto-discovered, manual overrides for people who
  know what they want.
- **Fix the shape of the work, let the amount float.** STREAM and BabelStream
  fix five tiny kernels and scale the array to the machine, which is why the
  definition survived a decade of ports. mixbench does the opposite and sweeps
  intensity to produce a roofline curve. We want both: a stable figure of merit,
  and mixbench's sweep shape for the crossover question.

### 1.7 Summary: the properties that make a benchmark widely used

| Property | Why it matters |
|---|---|
| Single command, zero config | Determines whether people run it twice |
| Auto-scaling workload | Survives hardware generations |
| Explicit units + direction | Makes automated comparison possible |
| Machine-readable output | Lets others build on it without scraping |
| Environment captured automatically | Makes results comparable and debuggable |
| Versioned workload definition | Prevents silently comparing different things |
| Variance reported and controlled | Distinguishes a real 10% from noise |
| Correctness verified in-run | A fast wrong answer is not a result |
| Minimal dependencies | Determines whether it runs on the machine you care about |

The last two are where most benchmarks are weakest, and where this project's
goals are unusually demanding. They are the differentiators.

---

## Part 2 — How fast MD5 implementations are built

Survey of the techniques production MD5 implementations use, which of them
belong in a benchmark, and which would quietly invalidate one. No third-party
source is reproduced or relied on here; see §2.9 for why that matters.

### 2.1 The step, and why MD5 suits this benchmark

One step is

```
a = b + rotl32(a + f(b,c,d) + w[i] + T[k], s)
```

and a compression is 64 of them, fully unrolled, in four rounds of sixteen, with
the message-word schedule fixed per step rather than computed. Fast
implementations group the additions so a three-input add can be emitted where the
hardware has one.

Note what is *absent*: no loop, no table lookups, no branches, no memory access
beyond the message words. The entire compression function is straight-line
integer ALU work — exactly the property this benchmark exists to measure.

### 2.2 Three optimizations that matter, and one trap

**(a) Three-input boolean instructions for the round functions.**

`F(x,y,z) = (x&y)|(~x&z)` is three ops written naively. On AMD GPUs it is one
`BFI_INT`; on NVIDIA it collapses into a single `LOP3.LUT`, which computes an
arbitrary 3-input boolean in one instruction; on AVX-512 it is one `vpternlogd`
(§4.1). That is roughly a 2-instruction saving in each of 32 steps.

This is a genuine hardware capability difference and it is worth *exposing*
rather than hiding: a device with 3-input logic ops should score better here,
and that is a true fact about its integer SIMD throughput. So it belongs in the
benchmark.

**(b) Sharing the XOR in round 3.**

Round 3 uses `H(x,y,z) = x^y^z`. Consecutive steps share an operand pair, so
`x^y` can be computed once and reused by the following step. Saves one XOR per
two steps — 8 XORs per compression, free.

Free *on some ISAs*. Where a 3-input boolean instruction exists, `x^y^z` is
already one instruction and there is nothing to share, so applying the trick
adds work. This is the clearest reason round functions must be written per ISA
rather than once generically (§4.1).

**(c) Precomputing `K + w[i]` for message words that do not vary.**

If most of the message is fixed across the batch — which is the normal case for
a search workload, where only a few bytes change per candidate — then `K + w[i]`
is loop-invariant and can be hoisted, so most steps lose an add. In an optimized
search kernel, 60 of 64 steps can drop one addition.

**This one we deliberately exclude.** It makes each reported hash cheaper without
making the hardware faster, so it inflates the score while measuring less. Our
message words are loaded from a corpus in memory precisely so that no compiler
can apply it (§2.8, and the `-flto` prohibition in the Makefile).

**(d) The trap: step reversal and early exit.**

An implementation that only needs to know *whether a digest matches a target* can
run the final steps backwards from the known target instead of forwards, and
abandon the rest on a mismatch. It is not computing MD5 at that point; it is
computing "does this equal that", which is strictly less work.

**This is the trap for a benchmark.** Combine (c) and (d) with a fixed short
message and known padding and you get a figure that is much larger than the
machine's actual MD5 rate. A tool whose actual question is "does this digest
match" may legitimately report that number, because matching is the thing it
measures. A benchmark claiming to measure integer SIMD throughput must either
replicate the shortcuts exactly and say plainly that is what it timed, or refuse
them and not claim comparability. We refuse them.

### 2.3 The vectorization model

The standard structure for a multi-buffer hash implementation: **write the
algorithm once against a lane width that is a compile-time parameter**, and let
each lane hash an independent message. There is no cross-lane communication
anywhere in MD5, which makes it embarrassingly vectorizable — N-way parallel
hashing scales essentially linearly with SIMD width until some other resource
runs out.

This maps directly onto CPU SIMD: SSE2 = 4 lanes, AVX2 = 8, AVX-512 = 16,
NEON = 4, SVE = variable. Same source, recompiled per width. GPU implementations
do the same thing with a vector type parameterised over 1/2/4/8/16 lanes.

**One architectural note worth calling out:** MD5 needs a 32-bit rotate every
step. AVX-512 has `vprold` (one instruction). AVX2 does not — it costs
shift + shift + or, three instructions, 64 times per block. So AVX-512 should show
a *super-linear* gain over AVX2 on this workload, beyond the 2× from width alone.
That is a real and interesting result for an integer-SIMD benchmark, and we should
make sure our implementation does not accidentally hide it.

### 2.4 Serial dependency and why one hash stream is not enough

The 64 steps form a single serial dependency chain: each step's output feeds the
next. Within one message, there is no instruction-level parallelism to exploit.
The chain is roughly 4 dependent ops deep per step (add, add3, rotate, add).

Consequence: a single N-lane SIMD hash stream will be **latency-bound**, not
throughput-bound. To saturate the ALUs you need several independent hash streams
in flight per thread so the out-of-order engine (CPU) or the warp scheduler (GPU)
has something to issue. GPU implementations get this for free from having many
work items resident; on a CPU it requires explicit multi-stream interleaving
inside the kernel.

**This is the single most important tuning knob in the whole benchmark** and it
needs to be a measured, auto-tuned parameter rather than a guess.

### 2.4b What sets the best stream count — read from the generated code

2.4 says stream count is the most important tuning knob. It does not say what
decides where the knob stops paying, and measurement produced an ordering that
looked backwards.

**The anomaly.** Live state per stream orders the algorithms identically on both
architectures — MD5 four streams, SHA-1 one to two, SHA-512 one — but does not
predict the peak *across* machines. AArch64 has 31 general-purpose registers to
x86-64's 16, and yet `sha1/scalar` peaks at **s1** on Neoverse V1 and **s2** on
Zen 5. Twice the register file, and it streams *less*.

**Method.** Disassemble every scalar kernel at s1..s4 for both targets (gcc
13.3, same source, `-fno-tree-vectorize`) and count loads and stores whose
address is formed from the stack pointer or frame pointer. Nothing else
legitimately lives on the stack in these functions: messages arrive through a
pointer argument and digests leave through another. The AArch64 ABI prologue is
excluded — it saves callee-saved registers with `stp`/`ldp` through memory where
x86-64 uses `push`/`pop`, which carry no memory operand, so counting it would
have penalised ARM by a fixed amount unrelated to the round body.

Round-body stack traffic, AArch64 relative to x86-64:

| | s1 | s2 | s3 | s4 | measured peak, x86 / ARM |
|---|---:|---:|---:|---:|---|
| md5 | −20% | −16% | −24% | −51% | s4 / s4 |
| sha1 | **−57%** | −12% | −11% | **+5%** | s2 / s1 |
| sha512 | **−49%** | −45% | +12% | **+29%** | s1 / s1 |

**The register file raises the floor, not the ceiling.** AArch64's advantage is
concentrated entirely at one stream — 57% less stack traffic on SHA-1, 49% on
SHA-512 — and it is spent by the second stream, inverted by the fourth. So ARM
peaks at a lower stream count not because streaming is worse there, but because
**s1 is unusually good**. On x86-64 one stream already spills, so the extra
instruction-level parallelism of a second is worth more than the spilling it
adds. On AArch64 one stream nearly fits, so the second is a real regression.

**Why SHA-1 specifically.** Per-stream live state is about 4–8 values for MD5
(it indexes the message words in place), 21 for SHA-1 (5 state plus a 16-word
rolling schedule window) and 24 for SHA-512. Usable registers are roughly 14 on
x86-64 and 28 on AArch64. SHA-1 is the only algorithm whose per-stream state
falls **between** the two register files, and it is the only algorithm whose best
stream count differs between the two architectures. MD5 fits in both and streams
to s4 everywhere; SHA-512's second stream needs 48 live values and overflows
both, so both peak at s1.

**And spill counts do not explain scaling.** A follow-up on sixteen Graviton3
cores (`c7g4xl-scaling-20260823`) found `sha1/neon-s2` scaling 14.64x where
`sha1/neon-s1` scales 15.97x and `sha1/neon-s4` 15.01x, against per-stream
round-body stack traffic of 265, 90 and 594. If spilling drove multi-core
scaling, s4 would be worst; it is better than s2. Whatever sets the *peak* at one
thread is not what sets the *scaling* across many, and this method speaks only to
the first. That capture's own mechanism is still unexplained, with the candidates
and the experiments that would separate them recorded alongside it.

**What this does not establish.** These are static counts from one compiler
version. A spill count is not a spill cost — store-to-load forwarding makes many
of these nearly free, and the out-of-order window, which 8a names as the other
candidate variable, is invisible to this method. What the counts do settle is
the negative: **register count does not predict the peak stream count**, and the
mechanism that does track it is where the spilling starts rather than how much
of it there is.

**The AVX-512 ladder is bracketed, and it ends in a cliff.** The ladder stops
at eight streams, and on a desktop Zen 5 core `md5/avx512` under gcc was still
gaining there, so whether the peak lay beyond it was open. Measured 2026-09-24
on a Ryzen 9 9950X, one thread, a 256 KiB corpus, five separate processes per
point, with s12 instantiated in a scratch build for the purpose: under gcc
16.2.1, s8 stays best at about 580 MH/s and s12 falls to about 266 — below s4.
Under clang 22.1.8, s6 is best and s8 has already collapsed to about 274; s12
is no better. The peak is s8 or s6 depending on the compiler, never above, so
no AVX-512 single-core figure from this part was a lower bound for want of a
longer ladder, and s12 was not added.

The collapse is a cliff rather than a slope, which is what the section above
predicts. MD5 keeps four state vectors live per stream, so s8 needs all 32
architectural vector registers for its state alone and s12 needs 48. Counting
stack-addressed instructions in each kernel — the whole function this time, not
the round body in isolation — clang's s8 carries about 1.6x gcc's, and clang's
s8 is the one that collapsed. The two compilers disagree about whether 32 live
vectors in 32 registers can be scheduled; neither can do 48.

### 2.5 Autotune

Serious GPU hash implementations do not ask the user for work sizes. They search
at startup, tuning roughly three things:

- total work items in flight (the outer loop)
- inner-loop iterations per work item
- work-group size

with a coarse "how aggressively may I consume the device" profile on top, so an
interactive machine and a headless one behave differently.

**Lesson:** we need the same. Optimal work-group size and batch size vary
enormously across an Intel iGPU, a consumer NVIDIA card, and a datacenter AMD
part. A fixed guess will be wrong by more than the 10% we care about. Autotune
must be part of the benchmark, its chosen parameters must be reported in the JSON,
and it must be overridable so a result can be reproduced exactly.

### 2.6 Correctness as a gate

The established practice in GPU hashing tools is to run each kernel against a
known input/digest pair at startup and **abort** on mismatch, on the grounds that
a wrong answer usually means a broken driver or runtime rather than a broken
program. Tools that offer a flag to skip the check generally refuse bug reports
filed with it disabled.

That is precisely the requirement that "a successful benchmark run
must also indicate that the hardware is computing the correct value", and the
shape is right:

1. Verify **before** timing (catches broken drivers/toolchains up front).
2. Verify **the same kernel** that will be timed, not a reference implementation
   of it. A self-test that exercises different code proves nothing.
3. Make failure **fatal by default**. A benchmark that prints a number next to a
   correctness warning will have the number quoted and the warning dropped.

For our purposes we should go further, because a GPU that is overclocked or
thermally marginal can pass a startup self-test and then produce garbage under
sustained load. Verification should be *continuous and cheap* — reduce all
computed digests into a running checksum and compare against a precomputed
expected value at the end of every timed iteration. Done with a combining
reduction, this costs a tiny fraction of the hashing work and makes the entire
timed region self-verifying rather than just the warm-up.

### 2.7 Runtime backend loading — the answer to "minimal dependencies"

Portable GPU tools generally link against *none* of the vendor runtimes at build
time. They declare their own function-pointer typedefs and `dlopen` whatever is
present at startup — `libOpenCL.so` / `libOpenCL.so.1`, and equivalently
`libcuda.so.1`, `libamdhip64.so.N`, and the runtime compilers — resolving each
entry point into a table, with individual symbols marked required or optional
(not every OpenCL ICD provides every function).

**This is the technique that satisfies our dependency requirement.** With
vendored Khronos headers (`CL/cl.h` is a header-only, permissively licensed
interface) and `dlopen` at runtime:

- Build requires only a C compiler and libdl. No SDK, no ROCm, no CUDA toolkit.
- The same binary runs on an NVIDIA box, an AMD box, an Intel box, or a box with
  no GPU at all — degrading to CPU-only cleanly instead of failing to load.
- No network access needed at build or run time.

The cost is writing out the function-pointer table by hand, which is mechanical.

### 2.8 What to adopt and what to avoid

**Adopt:**
- Three-input boolean instructions for the round functions where the ISA has them
- The round-3 XOR-sharing trick, applied per-ISA only where it actually helps (§4.1)
- Parameterized vector width, algorithm written once
- Autotune before measuring; report the chosen parameters
- Correctness gating, made continuous rather than startup-only
- `dlopen` backend loading with vendored headers

**Avoid:**
- Step reversal and early exit against a target digest — computes less than MD5
- Precomputing `K + w[i]` from message words fixed across the batch — same problem
- Any dependence on a short fixed-length message with known padding, *unless*
  deliberately adopted as a separately-labelled workload variant

The first two are the ones that would make our numbers look better while
measuring less, so they are excluded structurally rather than by discipline: the
message corpus lives in memory and `-flto` is prohibited, so the compiler cannot
apply them either.

### 2.9 Licensing constraint: the licence constrained the implementation

The project was written under a **public domain** dedication (decision 7), which
constrained where the MD5 core could come from and not just what the LICENSE
file said. It relicensed to **BSD 3-Clause** in 2026 and no code changed, because
everything below was already satisfied — which is the argument for choosing the
strict licence first.

The constraint was satisfiable because of the distinction between an
*implementation* and an *algorithm*:

- **MD5 is nobody's to license.** It is specified in RFC 1321, and an algorithm
  is not the subject of copyright. Anyone may implement it.
- **The constants are derived, not authored.** The 64 `T[i]` values are defined as
  `floor(2^32 × |sin(i)|)` for i = 1..64. We generate them from that formula rather
  than transcribing a table, which makes their provenance unambiguous.
- **The optimizations are ideas.** "Map the round function onto a 3-input boolean
  instruction", "share the XOR between consecutive round-3 steps", "hoist
  `K + w[i]` out of the loop" are techniques, not code. Describing them in prose
  and implementing them independently is clean.

**Practical rule: no source from any existing MD5 implementation is copied into
this repository — not macros, not the step schedule, not the constant table.**
The MD5 core is written from RFC 1321's specification prose, constants generated
from the sine formula, and the optimizations applied from the descriptions in
this document.

---

## Part 3 — Finding the compute/bandwidth crossover

Goal 2 — "determine the tradeoff point between pure compute bound and
PCIe or memory bound" — is a roofline measurement, and there is prior art whose
shape we should copy.

### 3.1 mixbench's method

mixbench sweeps **operational intensity** (compute ops per byte of memory
traffic) across a range by varying how many arithmetic ops it performs per memory
access, and emits CSV rows of `(experiment ID, flops/byte, time, GFLOPS, GB/s)`.
Plotting throughput against intensity produces the roofline directly: the ridge
point where the memory-bound slope meets the compute-bound ceiling *is* the
crossover. It covers CUDA, OpenCL, HIP, SYCL and OpenMP, and includes an integer
multiply-add experiment alongside FP32/FP64/FP16.

### 3.2 How this maps onto a hash benchmark

MD5 gives us a naturally variable operational intensity, because **message length
is the knob**:

- A 55-byte message = 1 compression = 64 steps for 64 bytes moved. Low intensity.
- A 1 MiB message = 16384 compressions for 1 MiB moved. High intensity.
- Iterated hashing (hash the digest N times, PBKDF2-style) = arbitrarily high
  intensity with *zero* additional memory traffic.

So a single sweep parameter — bytes of message per hash — walks the workload from
bandwidth-bound to pure compute-bound without changing the algorithm. That is
cleaner than mixbench's synthetic approach, because every point on the curve is
still doing real, verifiable MD5.

There are **three distinct roofs** to find, and they should be separately
reported:

1. **Compute ceiling** — hashes/s with data resident in registers/cache.
2. **Memory-bandwidth roof** — messages streamed from device DRAM (GPU) or system
   DRAM (CPU). Sweep message size across the cache hierarchy to find each knee.
3. **PCIe roof** — messages streamed from host memory across the bus. This is the
   distinctly GPU-relevant one and the one that determines whether offloading is
   worth it at all. Should be measured both with pageable and pinned host memory,
   since the difference is often 2×, and with/without transfer/compute overlap.

The PCIe crossover has a clean closed form worth reporting explicitly: offload
pays off when `bytes_per_hash / PCIe_bandwidth < time_per_hash_on_CPU`. Our
benchmark can measure both sides of that inequality and print the answer.

### 3.3 BabelStream's portability lesson

BabelStream's durability comes from defining the *measurement* independently of
the *implementation*: five trivial kernels, one figure of merit (GB/s), and then
N separate implementations, each written idiomatically for its programming model
rather than through a lowest-common-denominator abstraction layer.

That argues against building a heavy internal abstraction over OpenCL/CUDA/HIP.
Better: one clearly specified workload and timing protocol, plus per-backend
implementations that share the algorithm source but not necessarily the host code.

---

## Part 4 — Portability landscape under our constraints

The goals require AMD + NVIDIA + Intel, C and Python only, Linux only, minimal
dependencies, no internet.

| Option | AMD | NVIDIA | Intel | Verdict |
|---|---|---|---|---|
| **OpenCL 1.2+** | Yes (ROCm/Mesa Rusticl) | Yes (3.0 as of recent drivers) | Yes (NEO) | **Only single API covering all three.** Header-only build dep, `dlopen`able |
| CUDA | No | Yes | No | Vendor-locked; needed only if we want peak NVIDIA numbers |
| HIP | Yes | Yes (via CUDA) | No | Excludes Intel |
| Level Zero | No | No | Yes | Intel-only |
| SYCL | Yes | Yes | Yes | Requires a SYCL compiler — violates minimal-deps, and it is C++ |
| Vulkan compute | Yes | Yes | Yes | Portable but enormous host-side boilerplate; SPIR-V toolchain needed |
| OpenMP target offload | Partial | Partial | Partial | Compiler-dependent, fragile across vendors |

**OpenCL is the clear baseline.** It is the only one that hits all three vendors,
needs no SDK to build against, and can be loaded at runtime. Its weaknesses are
known — NVIDIA's OpenCL has historically trailed CUDA on peak throughput, and
online kernel compilation adds startup cost and introduces a compiler-version
variable into results (which we must capture in metadata).

A reasonable structure is OpenCL as the portable path that always works, with an
*optional* CUDA/HIP backend compiled in only if the toolkit happens to be present,
for the case where someone wants absolute peak NVIDIA/AMD numbers. Whether that
second path is worth the maintenance cost is a project decision, not a technical
necessity — see Open Questions.

### 4.1 CPU SIMD instruction sets

Requirement (confirmed 2026-08-16): **specialised integer SIMD instruction sets
must be used where available.** This is not merely an optimization — exposing the
difference between these ISAs *is* the point of the benchmark.

| ISA | Lanes (u32) | 32-bit rotate | 3-input boolean | Notes |
|---|---|---|---|---|
| SSE2 | 4 | shift+shift+or (3 ops) | no (≈3 ops) | Universal x86-64 baseline |
| AVX2 | 8 | shift+shift+or (3 ops) | no (≈3 ops) | Dev-box ceiling (N100) |
| AVX-512F | 16 | **`vprold` (1 op)** | **`vpternlogd` (1 op)** | Both wins, see below |
| AVX10.x | 8 or 16 | `vprold` | `vpternlogd` | AVX-512 semantics, 256-bit-capable parts |
| NEON | 4 | `shl`+`sri` (2 ops) | no | ARMv8 baseline |
| SVE / SVE2 | **variable** | `revb`/`lsl`+`lsr`+`orr` | no | Vector-length agnostic — see §4.2 |

**The AVX-512 result is larger than the width doubling suggests.** Two separate
single-instruction wins compound on this specific workload:

1. **`vprold`** does a 32-bit rotate in one instruction where SSE2/AVX2 need three.
   MD5 does 64 rotates per compression.
2. **`vpternlogd`** computes *any* 3-input boolean function in one instruction —
   the x86 equivalent of NVIDIA's `LOP3.LUT` and AMD's `BFI_INT`. All four MD5
   round functions collapse to a single instruction each. Immediates derived and
   verified:

   | Function | Definition | `vpternlogd` imm8 |
   |---|---|---|
   | `F(x,y,z)` | `(x&y) \| (~x&z)` | `0xCA` |
   | `G(x,y,z)` | `(x&z) \| (y&~z)` | `0xE4` |
   | `H(x,y,z)` | `x^y^z` | `0x96` |
   | `I(x,y,z)` | `y ^ (x \| ~z)` | `0x39` |

   (`0x96` is the well-known three-way-XOR immediate and `0xCA`/`0xE4` are the
   standard mux truth tables, which cross-checks the derivation.)

Rough per-step op count, ignoring the adds that are common to both paths:
AVX2 needs ~3 (boolean) + ~3 (rotate) = 6 ops; AVX-512 needs 1 + 1 = 2. Combined
with 2× the lanes, **AVX-512 should land above 2× AVX2 on MD5.** This is exactly
the kind of hardware truth this benchmark exists to surface, and it is a strong argument
for MD5: the algorithm happens to lean on precisely the integer capabilities that
separate ISA generations.

**MEASURED, 2026-08-17** on a Xeon Silver 4210 (Cascade Lake), gcc 15.2 —
the figures, consistent across thread counts.

**The direction was right and the magnitude estimate was not.** An earlier
version of this section guessed "plausibly 3-4×"; the answer is a little over
2×. The
instruction-count prediction held exactly — per kernel body, AVX2 emits 981
instructions for 8 messages (387 of them boolean/shift work) against AVX-512's
735 for 16 messages (128 non-add ops: 64 `vpternlogd`, 64 `vprold`). That is
**2.67× fewer instructions per message**.

The shortfall is issue width, not instruction count. On Skylake-SP and Cascade
Lake, 512-bit operations execute on fewer ports than 256-bit ones — ports 0 and 1
fuse to serve one 512-bit unit — so doubling the register width does not double
the issue rate. Getting the measured 2.20×
from a 2.67× instruction reduction implies roughly 0.82× the instructions retired
per cycle, which is what that port arrangement predicts.

Two caveats on this figure. It was taken in a VM with no `cpufreq` interface and
a nominal-TSC `/proc/cpuinfo`, so **AVX-512 licence-based downclocking could not
be observed**; on bare metal, where heavy 512-bit work drops the core clock, the
ratio may be lower. And a Xeon Silver is the low tier of the Cascade Lake range;
Gold and Platinum parts have more 512-bit execution resource, so the ratio should
be better there. Both are reasons to expect this number to move on other
hardware, not reasons to distrust it here.

**MEASURED, 2026-08-22, on AMD Zen 5 — the explanation holds.** An EPYC 9R45
gives 2.72x, above the 2.67x instruction
reduction rather than below it, which is what the port-fusion account predicts
for a part that does not fuse. The same run settles the width question in
isolation: AVX2 over SSE2 is 1.93x there against ~1.06x on Gracemont, so a
doubling of register width is worth nearly 2x when the datapath is actually that
wide. Width and generation were confounded in the Cascade Lake number and are
separable across the three machines.

The downclocking caveat survives -- that instance is also a VM -- but its
consequence is now bounded: the AVX-512 advantage is 2.71x at one thread and
2.71x at sixteen, so whatever the part
does to its clock under 512-bit load, it is not taking the ratio back.

Note also that the `MD5_H1`/`MD5_H2` XOR-sharing trick from §2.2(b) becomes
*pointless* on AVX-512 — `vpternlogd` already does `x^y^z` in one instruction, so
there is nothing to share. The trick must therefore be per-ISA, not global. A
naive port that applies it everywhere would slow the AVX-512 path down.

### 4.1b The SHA extensions: what a fixed-function unit is actually worth

Every ISA in the table above is a general integer vector ALU: it differs from the
others in width and in how few instructions a round function costs. **SHA-NI is a
different kind of thing entirely**, and implementing it turns design.md finding 1
from an assertion into a number.

`SHA1RNDS4` performs four real SHA-1 rounds; `SHA1MSG1`/`SHA1MSG2` perform the
message expansion. There is nothing left for a kernel template to do, and the
128-bit register holds *one* message's state rather than a vector of messages —
so this path is `lanes = 1` and every other kernel here is not.

**MEASURED, 2026-08-17** on an Intel N100 (Gracemont E-core), gcc 13.3, single
thread at 2.9 GHz — the four SHA-1 paths, from one lane to
the fixed-function unit.

Two findings, the second more interesting than the first.

**1. The unit is worth about 2x here, and that number does not generalise.**
Gracemont has a 128-bit vector datapath, so its 256-bit AVX2 operations are
cracked into two µops and the SIMD side of this comparison is roughly half as
strong as it would be on a P-core. A core with full-width AVX-512 and no SHA unit
could plausibly beat SHA-NI outright. The ratio is a property of the pair, not of
the SHA extension.

**MEASURED, 2026-08-22 — and the speculation was too cautious.** It does not
take a core lacking a SHA unit: an AMD EPYC 9R45 *has* SHA-NI, and its own
AVX-512 path beats it by 2.2x, putting the ratio at **0.46x**
. Across the two machines the same
comparison spans 4.6x, from 2.13x to 0.46x, with the SHA extension unchanged
throughout. "What is a fixed-function unit worth" is therefore not a question
about the unit at all; it is a question about what sits beside it. This is the
strongest evidence in the project for measuring rather than assuming, because a
rule of thumb that accelerators win would cost more than half the available
throughput on this part.

**2. Stream interleaving is worthless on SHA-NI — uniquely.** Interleaving
independent dependency chains is the single most load-bearing optimisation in
this benchmark -- 2.30x on a genuinely scalar path on Zen 5, 1.70x on the N100
-- and how many streams pay is ordered by how much live state each carries. MD5
climbs to four streams on every instruction set measured. SHA-1 and SHA-512 keep
a rolling sixteen-word message schedule as well as their state, peak at two
streams and one respectively, and then fall off sharply.

That ordering is the same on x86-64 and AArch64, which is what makes live state
the plausible mechanism. The *peak* is not so tidy: AArch64 has 31
general-purpose registers to x86-64's 16, and MD5 does tolerate one more stream
there, but SHA-1's scalar peak moves the other way. Register count alone is
therefore not the whole story, and the depth of the out-of-order window is the
untested candidate. On SHA-NI the ordering inverts and is monotonic.

A single chain already saturates the unit. Working backwards from the measured
rate: ~63 cycles per 64-byte block, over 20 `SHA1RNDS4` instructions, so ~3.1
cycles per four-round group — which means `SHA1RNDS4`'s reciprocal throughput is
essentially equal to its latency on this core. It is not pipelined for
independent work. Additional streams cannot use issue slots that do not exist and
merely cost registers (nine xmm per stream against sixteen architectural), so
they get slowly worse.

This is a genuinely different performance model from a wide vector unit: the SHA
extension wins on *latency-bound, one-message-at-a-time* work, precisely the case
multi-buffer SIMD cannot help. It is also a reminder of why the harness picks the
stream count by measurement rather than by rule — a rule derived from every other
kernel would have picked the worst variant here.

### 4.2 The SVE problem

SVE is **vector-length agnostic**: hardware implementations range from 128 to
2048 bits and the width is *not known at compile time*. This directly conflicts
with the model in §2.3, where lane count is a compile-time constant that types
the vector and drives full unrolling.

Two viable approaches:

- **VLA-native** — write the SVE path using `svuint32_t` and predication, letting
  the hardware pick the width. Idiomatic and future-proof, but a different code
  shape from the fixed-width paths, so the "one algorithm source" goal partially
  breaks down for this backend.
- **Fixed-width via `-msve-vector-bits=N`** — compile SVE code to a known width.
  Keeps the shared source structure, but produces a binary that is wrong on
  hardware of a different width, which conflicts with single-binary portability.

Recommendation: VLA-native, accepting SVE as the one backend that does not share
the fixed-width source structure. This mirrors BabelStream's principle from §3.3 —
implement idiomatically per model rather than forcing a lowest-common-denominator
abstraction.

Worth flagging as a scope observation: **SVE/NEON imply ARM, which the original
goals did not list** (they name AMD, NVIDIA and Intel). ARM CPU support is either an
intentional widening of scope or SVE should be treated as a lower-priority
future path. Recorded as open question 6.

### 4.3 Dispatch strategy

Plain C with compiler intrinsics, **one binary containing every ISA path, selected
at runtime** via `CPUID` on x86 and `getauxval(AT_HWCAP/AT_HWCAP2)` on ARM. Each
path lives in its own translation unit compiled with its own `-m` flags
(`-mavx2`, `-mavx512f`, …) so that no ISA leaks into the dispatcher.

This matters for reproducibility: `-march=native` would make the binary's
behaviour depend on the build host, which is exactly the kind of hidden variable
that makes benchmark results incomparable. One binary, runtime dispatch, and the
selected path reported in the JSON metadata.

Consequence for validation: the dev box (N100, AVX2 ceiling) **cannot execute the
AVX-512 path**. That path can be compiled, disassembled and inspected here, and
verified for correctness only on AVX-512 hardware. Given the 10%-significance bar
for significance, shipping an unrun AVX-512 path is a real gap, not a rounding error.

---

## Part 5 — Measurement methodology

Collected from benchmarking practice literature; these are the things that
determine whether a 10% delta is real.

**Environment control (measure and report, warn if unfavourable):**
- CPU governor — `performance` vs `powersave` is not a small effect; one measured
  case went 99.6 µs → 54.5 µs. Read `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.
- Turbo/boost — variable clocks make results depend on ambient temperature and
  prior load. Report the sustained frequency actually observed during the run.
- SMT — disabling SMT in one measured case improved coefficient of variation 3×
  (to 0.26%). We cannot disable it for the user, but we can report its state.
- Thermal state — a GPU benchmarked from cold and one benchmarked after 20 minutes
  of load are different machines. Report temperature/clock at start and end and
  flag throttling.
- CPU pinning — avoids migration and cold-cache effects. We control this; use it.

**Timing:**
- `clock_gettime(CLOCK_MONOTONIC_RAW)` on the host — immune to NTP slew, unlike
  `CLOCK_REALTIME`, and unaffected by adjtime unlike plain `CLOCK_MONOTONIC`.
- On GPU, prefer the runtime's own event timing (`cl_event` profiling) for kernel
  time, but *also* report wall-clock end-to-end including transfers — the gap
  between the two is exactly the PCIe story from Part 3.
- Never use `rdtsc` for cross-machine numbers without invariant-TSC verification.

**Statistics:**
- Warm up until steady state, discard warm-up, then take N ≥ 30 samples where
  time permits.
- Report **min, median, mean, stddev, CoV** and the raw samples. Minimum is the
  best estimator of the machine's capability (noise is one-directional — it can
  only slow you down); median is the better estimator of what a user will
  experience. They answer different questions, so publish both rather than
  picking.
- Precision beats accuracy for a benchmark. A slightly slower but tightly
  reproducible measurement is more useful than a faster noisy one.

---

## Decisions

Resolved in discussion on 2026-08-16, before implementation began.
1. **Workload definition: clean full MD5.** Every hash performs the full 64-step
   compression with correct padding and a fully computed digest. No target-digest
   step reversal, no early exit, no fixed-message constant folding. Consequence,
   accepted deliberately: **our numbers will read lower than any figure produced
   with those shortcuts, and are not comparable to one.** They measure generic MD5
   throughput — i.e. the integer SIMD ALUs — which is what the goals ask for.
   This should be stated plainly in user-facing output so nobody mistakes the
   two.

   Note the optimizations from §2.2 that remain *in* scope, because they preserve
   the full computation: bitselect for F/G, and the H-round `t = x^y` sharing
   trick. Only the shortcuts that reduce the work below a true MD5 are excluded.

2. **Backends: OpenCL + CPU now, CUDA/HIP left open.** OpenCL is the portable GPU
   path across AMD/NVIDIA/Intel, `dlopen`ed at runtime per §2.7. Host-side code
   should be structured so an optional vendor backend can be added later without
   restructuring, but no CUDA/HIP implementation is in the initial scope.

3. **MD5 first, structured for more.** Single algorithm to start, but the kernel
   and harness interfaces are designed so a second algorithm with a different
   instruction mix (rotate-heavy vs add-heavy, different ILP profile) can be added
   without reworking the harness. Rationale: one algorithm characterizes one
   instruction mix, not "integer SIMD" in general.

4. **Python's role: orchestration only.** Python drives sweeps, orchestrates runs
   and renders reports from the JSON. It is a convenience layer over the C
   kernels and never enters the timed region. Stdlib only — no numpy, per the
   no-internet/self-contained constraint. The C binary must remain fully usable
   standalone, with Python strictly optional.

5. **Specialised integer SIMD ISAs are required where available** — AVX/AVX2/
   AVX-512 on x86, NEON/SVE on ARM — via runtime dispatch from a single binary.
   See Part 4.1–4.3 for the ISA landscape, the `vpternlogd`/`vprold` findings, and
   the SVE vector-length-agnostic problem.

6. **ARM is in scope, as a follow-on phase.** NEON and SVE paths are planned but
   not part of the initial delivery. The dispatch architecture of §4.3 must
   accommodate them from the start — `getauxval(AT_HWCAP/AT_HWCAP2)` detection and
   per-ISA translation units — so that adding them later is additive rather than a
   restructuring. The SVE vector-length-agnostic question (§4.2) is deferred with
   the phase, but the recommendation stands: VLA-native.

7. **License: BSD 3-Clause** (was public domain). The original decision
   constrained how the MD5 core could be written — see §2.9. No source may be
   copied from any existing MD5 implementation: the RFC 1321 reference carries an
   RSA notice, and permissive-licensed ones require their notice be retained in
   derivatives. The implementation is written from the specification with
   generated constants.

   **Relicensed 2026-08-25 to BSD 3-Clause**, copyright "The valubench authors".
   No code changed, because the stricter constraint was already satisfied
   everywhere — which is the whole argument for picking the strict licence
   before writing anything. Sources carry an SPDX tag rather than a copyright
   line, so the holder is named in one file. Generated headers carry no notice
   at all: they are gitignored, never redistributed as source, and the `.cl`
   files they come from are tagged.

   Noted for the record, from when the choice was public domain: CC0 1.0 has
   stronger standing in jurisdictions that do not recognise dedication to the
   public domain, whereas the Unlicense is simpler and more common for code.
   Neither grants patent rights, and neither does BSD 3-Clause — immaterial
   here, since MD5 dates to 1992 and is patent-free. BSD 3-Clause was chosen
   over MIT for its non-endorsement clause.

8. **Multi-threading via a persistent, batch-partitioned worker pool.** The
   verified batch is split across threads; each thread holds the reference
   checksum for its own slice and verifies it every rep, so nothing
   synchronises in the hot path. XOR's associativity means the partials
   recombine to the single-threaded value, so **the reported checksum is
   identical at any thread count** and remains a cross-machine fingerprint.

   One measured lesson: the driving thread must do real work, not just wait at
   a barrier. With N workers plus an idle driver on N cores, one core carries
   two runnable threads, which core varies per sample, and the coefficient of
   variation exceeded 25%. Making the driver worker 0 fixed it.

9. **Compute intensity is tunable via chained iterations.** `--iterations N`
   runs N chained MD5s per hash, feeding each digest back as the first 16
   message bytes *of the same message* rather than hashing the 16-byte digest
   alone. That distinction matters: hashing the bare digest would leave 12 of
   16 message words constant, so later iterations would be cheaper than the
   first and the knob would be non-linear. Feeding back into the full message
   keeps four varying words and an identical instruction mix every iteration.

   Verified on the N100: compressions/sec stays flat (50.2 -> 53.3 MC/s) from 1
   to 256 iterations, so the knob is linear in compute as intended. The small
   rise is per-group setup being amortised over more work. Together with
   message size (phase 2) this gives the two axes needed for the
   socket-throughput surface described in part 3.

10. **Message length is a parameter, and messages come from a corpus in memory.**
   Generating message content on the fly would have kept the benchmark
   permanently compute-bound and unable to find the memory crossover at all, so
   a corpus of pre-padded, lane-interleaved messages is built once outside the
   timed region and streamed by the kernels. Padding is applied at build time
   (kernels never see it) and the layout puts the N lane-copies of each word
   contiguously, so kernels use plain vector loads -- transposing inside the
   timed region would measure shuffle throughput instead of hash throughput.

   Batch size is now derived from a target working set in bytes rather than
   being a fixed message count, which is what makes the memory axis
   controllable independently of message length.

   **Measured negative result worth recording:** MD5 barely becomes memory
   bound. On the N100 with 1015-byte messages, moving the working set from
   L2-resident to 128 MiB costs only about a tenth of throughput
   (measured), and DRAM is no worse than L3. One
   compression is several hundred integer ops per 64 bytes, so operational
   intensity is high enough that this workload sits far right of the roofline
   ridge point on essentially any machine. Locating a genuine bandwidth-bound
   regime for goal 2 will need much larger messages, or a deliberately
   bandwidth-hungry companion kernel, rather than just a bigger corpus.

11. **The iteration scheme is not PBKDF2, and must not be read as if it were.**
   People reach for the PBKDF2 analogy immediately, so the difference is
   recorded here.

   PBKDF2 chains `U_n = HMAC(P, U_{n-1})` and XORs every `U` into the output.
   Each iteration is an **HMAC**, which is two MD5 invocations; with the ipad
   and opad states precomputed — the standard implementation — that is **2
   compressions per iteration**. valubench chains `digest -> first 16 bytes of
   the same L-byte message`, takes only the final digest, and costs
   `ceil((L+9)/64)` compressions per iteration (1 at the default L=55).

   | | PBKDF2 | valubench |
   |---|---|---|
   | Per iteration | HMAC = 2 compressions | 1 MD5 = ceil((L+9)/64) compressions |
   | Chain input | the bare 16-byte U | digest in the first 16 bytes of an L-byte message |
   | Accumulation | XOR of every U | final digest only |
   | Keyed | yes, password as HMAC key | no |

   What genuinely matches is the **serial dependency chain**: each iteration
   needs the previous digest, so a chain cannot be parallelised internally. That
   is the property making PBKDF2 a KDF and making `--iterations` a clean compute
   knob here.

   **The trap worth spelling out.** Both designs end up with mostly-constant
   message words. At L=55, valubench iteration 2+ has words 0..3 carrying the
   digest and words 4..15 fixed — 12 of 16 constant, structurally the same as
   PBKDF2's inner hash. The difference is that valubench forces those words to
   be loaded from the corpus in memory, so `K + w[i]` cannot be folded. A real
   PBKDF2 implementation *does* fold them, and that is a large part of why
   optimized PBKDF2 kernels are fast per compression.

   Consequence: **a valubench number does not predict PBKDF2 throughput.** A
   specialised PBKDF2 kernel will beat our per-compression rate because it is
   allowed a shortcut we deliberately refuse — the same reason we are not
   comparable to any implementation that takes them (§2.2c, §2.2d).

   **Why not switch to true PBKDF2 iteration:** it would reintroduce exactly the
   constant-folding exposure that stops a number measuring the ALUs, and 2
   compressions per iteration is not a better intensity knob than 1 — both are
   linear, and the simpler denominator is easier to reason about in the GPU
   crossover arithmetic (design.md finding 2).

   Where a KDF workload *would* earn its place is as a separate algorithm: "how
   fast is this hardware at KDF work" is a real and different question, and the
   algorithm abstraction that SHA-1 and SHA-512 forced into existence is what
   would carry it. Two caveats stand in the way of it being worth much:
   PBKDF2-HMAC-MD5 is a benchmark construct nobody deploys, and the realistic
   choices (SHA-256, SHA-512) run into the hardware-accelerator problem of
   design.md finding 1.

12. **The GPU work decomposition mirrors the CPU one exactly**, so the XOR
   checksum matches bit for bit and the existing scalar reference validates
   device kernels with no new machinery. This is the verification design of
   section 2.6 paying off: a whole new backend inherited its correctness test
   for free.

   Two GPU-specific choices worth recording. Digests are reduced to one digest
   per work-group *on the device* before readback -- writing a partial per
   work-item would put megabytes of transfer inside the timed region and corrupt
   the PCIe measurement the GPU work exists to make. And the work-group size is
   deliberately decoupled from the corpus lane width: lanes drive coalescing and
   the group arithmetic the checksum depends on, while work-group size is a pure
   occupancy knob the host picks at launch, so it can be autotuned without
   changing what is computed.

13. **The algorithm abstraction, and why SHA-512 had to be in the first batch.**
   `include/algorithm.h` carries digest width, word size, block size, length
   field width, endianness and a reference function; the corpus builder,
   reference oracle, checksum and reporting take it from there. Kernels stay
   algorithm-specific, because MD5's 64 fixed-schedule steps and SHA-512's 80
   expanded ones share nothing worth sharing, but nothing around them does.

   Adding SHA-1 alone would have produced a false sense of generality: it has
   MD5's 64-byte block, 32-bit word and 8-byte length field, so only endianness
   and digest width would have moved. SHA-512 breaks all four at once, which is
   why it went in at the same time rather than later.

   One invariant made the lane-interleaved corpus survive unchanged: **every
   supported block is exactly sixteen words**. 16 x 32 bits is MD5's and SHA-1's
   512-bit block; 16 x 64 bits is SHA-512's 1024-bit block. The layout indexes
   by word, so it never needed to know how wide a word was.

   The XOR checksum generalised for free, as predicted -- widened to
   `uint64_t[8]` with a per-algorithm word count, it still validates every
   kernel against the reference and still folds identically across lanes,
   streams, threads and devices.

   Constant tables are transcribed from RFC 1321 and FIPS 180-4 rather than
   derived. An earlier build step generated MD5's from the sine formula; that
   bought a build dependency and nothing else, since the published tables are
   facts and the known-answer vectors are what actually establish correctness.
   Removing it also removed the project's last use of libm.

## Sources

**Phoronix Test Suite**
- [Phoronix Test Suite](https://www.phoronix-test-suite.com/)
- [PTS main documentation](https://github.com/phoronix-test-suite/phoronix-test-suite/blob/master/documentation/phoronix-test-suite.md)
- [PTS test profile creation guide](https://github.com/phoronix-test-suite/phoronix-test-suite/blob/master/documentation/test-profile-creation.md)
- [PTS test-profiles repository](https://github.com/phoronix-test-suite/test-profiles)
- [PTS test-suites repository](https://github.com/phoronix-test-suite/test-suites)
- [OpenBenchmarking.org pts test profiles](https://openbenchmarking.org/tests/pts)
- [Phoronix Test Suite (Wikipedia)](https://en.wikipedia.org/wiki/Phoronix_Test_Suite)

**Roofline / portability prior art**
- [mixbench](https://github.com/ekondis/mixbench)
- [mixbench README](https://github.com/ekondis/mixbench/blob/master/README.md)
- [mixbench (DeepWiki)](https://deepwiki.com/ekondis/mixbench)
- [Mixbench on OpenBenchmarking.org](https://openbenchmarking.org/test/pts/mixbench)
- [BabelStream v4.0 release notes](https://uob-hpc.github.io/2021/12/22/babelstream-v4_0.html)
- [BabelStream (ExCALIBUR-tests docs)](https://ukri-excalibur.github.io/excalibur-tests/apps/babelstream/)
- [Evaluating attainable memory bandwidth of parallel programming models via BabelStream](https://www.researchgate.net/publication/328546469_Evaluating_attainable_memory_bandwidth_of_parallel_programming_models_via_BabelStream)
- [Performance Portability across Diverse Computer Architectures (P3HPC/SC19)](https://sc19.supercomputing.org/proceedings/workshops/workshop_files/ws_p3hpc101s2-file1.pdf)

**Benchmarking methodology**
- [Tune the system for benchmarks — pyperf](https://pyperf.readthedocs.io/en/latest/system.html)
- [Tuning a Server for Benchmarking](https://david.alvarezrosa.com/posts/tuning-a-server-for-benchmarking/)
- [Reliably benchmarking small changes](https://ankush.dev/p/reliable-benchmarking)
- [Measuring Software Performance: Why Your Benchmarks Are Probably Lying](https://kakkoyun.me/posts/fosdem-2026-measuring-software-performance/)
- [Intel's Variable Clock Speeds and Benchmarking](https://www.mjr19.org.uk/IT/clocks.html)
