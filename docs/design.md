GOALS
This project is desiged to create a microbenchmark that highlights the integer SIMD performance of hardware.  It is designed to be used for the following purposes:
- Understand the SIMD performance of integer workloads on CPUs and GPUs.
- Determine the tradeoff point between pure compute bound and PCIe or memory bound.

REQUIREMENTS
- Should be extremely high performance.  Improvements of 10% or more are signifiant.
- Must be accurate and reproducable.
- Must also produce correct values.  A successful benchmark run must also indicate that the hardware is computing the correct value.
- Dependencies should be minimized and should not require Internet access to use.  The benchmark is expected to run on nearly new hardware in a bare Linux environment and should be self-contained.
- Allowable languges are C and Python
- Benchmark must be portable across AMD, NVidia and Intel devices.
- Must run on Linux distributions.  Apple and Windows are not relevant for this application
- Must produce output for either machine analysis (JSON) or human readable.

---

# What we learned building it

Everything above is the original brief and stands unchanged. This section records
what turned out to be true once the work started — the things that would have
changed the plan had we known them on day one. Full reasoning lives in
research.md; this is the executive version. Figures quoted here name the
machine they came from.

## 1. MD5 is the right vehicle, for a reason that is not obvious

Not for its cryptographic properties — it has none left. **No hardware has MD5
instructions**, and that is the whole point.

Modern x86 has SHA-NI and ARM has SHA extensions, so benchmarking SHA-1 or
SHA-256 would measure a *fixed-function accelerator*, not the integer SIMD ALUs
the brief asks about. The development box (Intel N100) advertises `sha_ni`, so
this is not a hypothetical. MD5 has no such unit anywhere, which forces it onto
the general integer vector path on every architecture.

It is also unusually good at exposing hardware differences, because it leans on
precisely the capabilities that separate ISA generations: 32-bit rotate and
3-input boolean logic (see finding 5).

The constraint this places on future work: **any additional algorithm must not
have a hardware accelerator** — or, if it does, both paths must be implemented
and reported separately, or the benchmark silently stops measuring what it
claims to measure.

The second option is the more interesting one, and it is now done. SHA-1 is
implemented both ways — multi-buffer SIMD and via SHA-NI — so the ratio between
them is a measured answer to "what is the SHA unit worth on this chip".

**The constraint is not static, and SHA-512 has since fallen foul of it.** Intel
added SHA-512 instructions in Arrow Lake-S and Lunar Lake in late 2024, so on a
recent Intel client CPU `--algorithm sha512` is now in the position SHA-1 was in
before its SHA-NI path existed: there is a fixed-function unit the benchmark
does not use, and the number it reports is no longer that machine's best. An
algorithm that satisfies this finding when it is added can stop satisfying it
later, without anything in the project changing. The roadmap carries the work
under "New instruction sets".

Measured on the N100, the fixed-function path runs at **more than twice the best
integer-SIMD path** — and the SIMD path there is
using eight lanes and three interleaved streams to get its number, against one
message at a time on the SHA unit.

**On Zen 5 the same comparison inverts: the SHA unit is worth less than half the
integer-SIMD path**. Nothing about SHA-NI
changed between the two machines; the vector path beside it did. So the question
this finding exists to answer — what is a fixed-function unit worth — turns out
to have no answer that survives a change of core, and the honest form of it is a
ratio quoted with the part it was measured on. The benchmark reporting both
paths is what makes that visible rather than a footnote.

Two caveats that matter more than the ratio itself:

- **The ratio is flattered by this particular CPU.** Gracemont is an E-core with
  a 128-bit vector datapath, so its 256-bit AVX2 ops are cracked in two and the
  SIMD side of the comparison is about half as strong as it would be on a P-core.
  Expect a materially smaller ratio elsewhere — plausibly below 1 on a core with
  full-width AVX-512 and no SHA unit.
- **Stream interleaving does nothing here, and that is itself the finding.**
  Interleaving is the single largest win in the benchmark everywhere else,
  though how many streams pay depends on the algorithm: MD5 wants four, SHA-1
  two, SHA-512 one or two, ordered by how much live state a stream carries.
  That ordering holds on x86-64 and AArch64 alike; the peak itself moves between
  cores in ways register count alone does not predict. On SHA-NI more streams
  are monotonically *worse*
  (the ladder). A single dependency chain already saturates
  the unit, so `SHA1RNDS4`'s reciprocal throughput must be close to its latency.
  Extra streams buy nothing and cost registers.

So the accelerator is real but not transformative, and it is *architecturally*
different from a wide vector unit: it wins on latency-bound single-stream work
rather than by doing many messages at once. Which is exactly the distinction the
brief cares about, and exactly what benchmarking MD5 avoids conflating.

## 2. Goal 2, stated precisely

Clarified 2026-08-16. The question goal 2 exists to answer is:

> **How many iterations of the kernel are needed to keep the overall computation
> GPU-limited rather than PCIe-limited?**

with the motivation being: *do not use or pay for an accelerator that does not
actually provide much benefit.*

That motivation needs two answers, not one, and they can disagree:

| | Question | What it tells you |
|---|---|---|
| **Saturation** | At what iteration count N\* does the GPU become compute-limited rather than bus-limited? | Whether the accelerator can be kept busy at all — and so whether a *bigger* one would help |
| **Break-even** | At what N does offloading beat the CPU already in the box? | Whether to buy the accelerator at all |

A device can be GPU-limited and still slower end-to-end than the host CPU (weak
device, fat transfers), or bus-limited and still faster. Break-even is the
purchasing decision; saturation explains why the answer came out as it did. Both
get reported.

**The measurement.** Sweep `--iterations` at fixed message size and plot
compressions/sec. It starts low (device starved by the bus), rises, then
plateaus at the device's compute roof. N\* is the smallest iteration count
reaching ~95% of that plateau. N\* is a curve over message size, not a single
number, because bytes-per-hash is the denominator of the intensity.

This works because `--iterations` raises device compute with **zero additional
PCIe traffic** — the digest feeds back on-device — so it is a clean operational
intensity knob. That property was designed in for the CPU compute axis and turns
out to be exactly what the GPU question needs.

Decisions that change what N\* means, and are therefore measured explicitly:
transfer/compute **overlap** (headline figures assume double-buffered overlap,
since that is what a competent implementation achieves; the non-overlapped
baseline is reported alongside), **digest readback** (checksum-only and
full-digest variants reported separately — full readback adds ~29% to bus
traffic at 55-byte messages and ~0.4% at 4096), and **pinned vs pageable host
memory**, which is often a 2x bandwidth difference and shifts N\* accordingly.

## 2a. The CPU memory-side crossover is real, but only a fast ISA reaches it

Goal 2 as originally written — "pure compute bound and PCIe **or memory**
bound" — also implies a memory-side crossover. For a long time this section said
there was not one on a CPU. That was wrong, and instructively so.

One MD5 compression is several hundred integer operations per 64 bytes consumed,
which puts the workload far to the right of the roofline ridge point — *at the
rate the machine can hash*. Three machines saw no knee: the development N100
loses about a tenth of its throughput moving from L2-resident to 128 MiB, and
Graviton3 loses 1.6% over the same range. Both conclusions were correct about
those parts and wrong as generalisations.

A Zen 5 core running AVX-512 hashes fast enough to ask for **17.6 GB/s** of
message traffic, and single-thread DRAM supplies about 10. It loses **44%** when
the corpus leaves the L3 slice. On the same machine, in the same sweep, AVX2
loses 3% and scalar 2% — so one ISA is memory-bound and another compute-bound on
identical data.

The lesson is that operational intensity is a property of the workload *and the
implementation*, not the workload alone. Widening the datapath moves you along
the roofline toward the ridge, which is the same thing as saying a faster kernel
is easier to starve.

What this means for planning:

- **Goal 1 (integer SIMD throughput) and a CPU memory-side crossover want
  opposite things**, and the Zen 5 result sharpens this rather than softening
  it. Goal 1 wants high operational intensity — that is what isolates the ALUs.
  A memory crossover needs intensity low enough to reach the ridge. It took a
  512-bit datapath to drag this workload down to the ridge at all, and at that
  point the number measures the memory system rather than the ALUs. One hash
  function still cannot serve both goals well.
- If a CPU compute-vs-memory crossover is wanted anyway, it needs a
  deliberately bandwidth-hungry companion kernel — closer to BabelStream or
  mixbench — not a bigger corpus.
- **None of this affects the PCIe question in 2 above,** which is a different
  and much more favourable measurement: PCIe is roughly two orders of magnitude
  slower than device memory, so the host-to-device knee is real, findable, and
  is the part of goal 2 that carries the decision.

**Update 2026-08-18: that knee is now measurable, and it was not before.** Until
now the corpus was uploaded once at device setup and every launch ran against
resident data, so there was no host-to-device traffic inside the timed region
and nothing for goal 2 to find — every device number the benchmark produced
silently assumed the data was already on the accelerator. `--transfer stream`
re-uploads before every launch and reports kernel time and transfer time
separately; the ratio between them is the answer, and N\* is where it crosses
1.0. See finding 2c.

Had we known this at the start, goal 2 would have been scoped as a GPU/PCIe
question from the outset rather than a general compute-vs-memory one.

## 2b. The accelerator's advantage is algorithm-dependent, and 64-bit narrows it

All three algorithms now have OpenCL kernels, which makes a comparison possible
that MD5 alone would have hidden: the same iGPU against one N100 core, each side
at its own best kernel, gives a **different accelerator advantage for every
algorithm** (the table).

Both sides get slower on SHA-512 — the CPU's AVX2 lanes halve at 64 bits too —
but the GPU loses about half again as much of its relative footing. The cause is
structural: consumer GPU ALUs are 32-bit, so every 64-bit add, rotate and shift
is emulated by the compiler.

**This is goal 2's question in a form the roofline does not capture.** "Is the
accelerator worth paying for" has no single answer even on fixed hardware with a
fixed working set — the ratio moves by more than half between MD5 and SHA-512,
before PCIe is considered at all. A benchmark that reported one hash function
would have given a number that quietly does not transfer to the next workload.

A second, narrower result worth carrying into any future device kernel:
**stream interleaving is a CPU technique that does not port.** On the CPU it is
the single biggest win. On the GPU it is at best neutral and often catastrophic
(the numbers). The device already has thousands of
work-items in flight, so a second stream hides no latency that was not already
hidden; it only consumes registers. The right device answer for a
register-hungry algorithm is one stream.

## 2c. Goal 2 is now measurable, and the ratio is the deliverable

`--transfer stream` puts the host-to-device upload inside the timed region and
reports the two halves separately:

```
  kernel busy 20.7% of wall time
  transfer    39.3% of wall, 7.62 GB/s host->device (64.0 MiB per pass)
  bound by    TRANSFER  (compute/transfer = 0.53)
```

Sweeping `--iterations` walks that ratio through 1.0, and the crossing is N\* —
the answer to "how many iterations of the kernel do I need to keep the overall
computation GPU-limited rather than PCIe-limited". On the development iGPU, MD5
crosses within the first few iterations while the measured link rate stays flat
throughout (the sweep), which is the check that the link
is being measured consistently rather than varying with the workload.

**N\* is solved, not searched for.** Transfer time is constant in the iteration
count (the same bytes cross the link at every point) and kernel time is linear
in it (every iteration is identical work by construction), so

```
    kernel_ns(N) = a + b·N     transfer_ns = T     N* = (T - a) / b
```

`sweep.py` fits both across a streaming sweep and reports the intercept, with
r^2 as the check that the linear model held -- it has been 1.0000 on every sweep
run so far. A geometric sweep alone would only bracket N\* to within its own
step, which is a factor-of-two answer rather than a balance point. Separating
the fixed launch cost `a` is what makes it exact: the cruder `N / ratio`
estimate charges that cost to compute and drifts where the fit does not
(both).

**N\* is a property of the (kernel, working set) pair, not of the kernel.**
Doubling the corpus moved it by a third, because compute scaled linearly while
transfer scaled faster -- the achieved link rate itself fell
(measured). Quote the working set alongside the number or
the claim does not travel.

Three further things about this measurement are worth stating, because they are
what make the number trustworthy:

- **The ratio answers the question for a pipelined implementation too**, even
  though this one uploads and computes in sequence. Overlapping transfer with
  compute can hide the smaller of the two but never the larger, so whichever
  side exceeds 1.0 is the binding constraint either way. That is why the two
  times are reported separately rather than folded into one throughput figure.
- **`repeats` must be forced to 1 in streaming mode**, and is. It amplifies
  compute without amplifying transfer, so any other value would inflate exactly
  the side of the ratio being measured. This is a correctness constraint, not a
  tuning choice.
- **Compare algorithms at 64-byte messages.** It is the one length where all
  three store exactly 128 bytes per padded message — MD5 and SHA-1 spill into a
  second 64-byte block, SHA-512 still fits one 128-byte block — so the transfer
  side is identical across the three and only compute varies. It is also the
  shortest length SHA-512 can iterate at, since the digest is fed back over the
  head of the message.

The iGPU figure is a demonstration of the mechanism, not a useful answer: there
is no PCIe on an integrated part, so the link rate above is a copy within system
RAM. The number that carries a purchasing decision needs a discrete GPU on a
real link, where the device is also 10-50x faster and N\* correspondingly higher.

## 3. The dependency chain, not the SIMD width, is the thing to get right

MD5's 64 steps form a single serial chain, and SIMD does not break it — every
lane of a vector advances in lockstep as one chain. A straightforward
implementation therefore measures dependency *latency*, not throughput.

Interleaving independent chains ("streams") was worth **3.1×** on the scalar path
alone, with no change in instruction set. A benchmark shipped without it would
have under-reported hardware by a factor of three while looking entirely
plausible.

The general lesson: for this class of workload the first-order question is "how
many independent chains are in flight", and only the second-order question is
"how wide is the vector". Any future kernel — GPU included — has to answer the
first one before its number means anything.

## 4. "Portable across AMD, NVidia and Intel" resolves to one API, plus a trick

OpenCL is the only single API covering all three vendors that does not require a
vendor SDK at build time. The trick that satisfies the minimal-dependency
requirement simultaneously is `dlopen`: vendor the Khronos headers, load
`libOpenCL.so.1` at runtime, and resolve entry points into a function-pointer
table. Build dependencies then reduce to a C compiler and libdl, one binary runs
on any vendor or no GPU at all, and nothing needs the network.

This is worth knowing up front because it shapes the whole GPU design, and
because the obvious alternatives (SYCL, Vulkan compute, OpenMP offload) each fail
one of the stated requirements.

## 5. Instruction set generation matters more than vector width

AVX-512 is not "AVX2 but twice as wide" for this workload. Two single-instruction
capabilities compound: `vpternlogd` collapses each MD5 round function from ~3 ops
to 1, and `vprold` does a 32-bit rotate in 1 op where AVX2 needs 3. Per step that
is roughly 6 ops down to 2, on top of 2× the lanes — so AVX-512 should be well
*above* 2× AVX2, not at it.

The corollary bit us in the opposite direction too: an optimization that helps
one ISA can hurt another. Sharing the `x^y` term between consecutive round-3
steps is a real saving on AVX2 and pointless on AVX-512, where `vpternlogd`
already does `x^y^z` in one instruction. **Round functions must be written per
ISA, not once generically.**

Also relevant to planning: AVX2 barely beat SSE2 on the development machine
(~6%), which is correct behaviour — Gracemont executes 256-bit ops on 128-bit
units. Benchmarks that expose this kind of microarchitectural truth are working
as intended, but it means "wider is faster" cannot be assumed anywhere.

**Both halves of that are now measured on one page.** On Zen 5 the same width
doubling is worth nearly the full 2x, and AVX-512 clears the instruction-count
prediction rather than falling short of it as Cascade Lake did
. Width is worth what the datapath behind it is
worth, and generation is worth what the issue ports allow — two independent
variables that a single machine cannot separate, and three machines do.

## 6. Reproducibility is mostly outside the benchmark's control

The 10%-significance bar is easily swamped by things the benchmark cannot change:
CPU governor, competing system load, SMT, thermal and power state. On the
development machine, background load alone pushes multi-threaded run-to-run
variation well past the 10% bar
(the figures).

The only honest response is to **measure the environment, report it with the
result, and flag when conditions make a number untrustworthy** — which is what
the tool now does, including a non-zero exit status for results too noisy to
trust. It cannot make a busy machine quiet.

Practical consequence for anyone using this: results intended for comparison
should come from a quiet machine with the performance governor set. That is a
usage requirement, and it belongs in the operating procedure, not just in the
code.

## 7. Correctness verification is nearly free, and the design generalises

The requirement that a run prove correct computation turned out to be cheap and
to have an unexpectedly useful property. Each kernel XORs together every digest
it computes; because XOR is commutative and associative, that checksum is
**invariant to how the work was distributed** — across lanes, streams, threads,
and eventually devices.

One expected value therefore validates every implementation, and the same value
appears at any thread count. It doubles as a cross-machine fingerprint. Any
future backend, GPU included, gets its correctness test for free by construction.

Worth stating explicitly because it was not obvious in advance that the
correctness requirement and the portability requirement could be satisfied by the
same mechanism.

## 8. A decision with consequences that reached into the code

**The licence constrained the implementation, not just the header.** The obvious
sources to crib an MD5 from are all encumbered: the RFC 1321 reference
implementation carries an RSA notice, and permissive-licensed implementations
still require their notice be retained in derivatives. Neither could be accepted
under the public-domain dedication the project started with, so the MD5 core was
written from the specification with constants generated from the sine formula.

The project is BSD 3-Clause now, which *could* accept a retained notice — but
the code is already free of any, so relicensing changed no line of it. That is
the point worth keeping: decide the licence before writing the code, because the
strict choice leaves you free to loosen later and the loose one does not.

## 9. Validation hardware is a project dependency

The development machine (Intel N100) has no AVX-512, no discrete GPU, and 4
cores. The AVX-512 path — the one most likely to produce this benchmark's most
interesting result — compiles and has been verified by disassembly, but has never
executed. GPU paths will have the same problem.

**Access to representative hardware is a prerequisite for the results being
meaningful, not a nice-to-have**, and should be planned alongside the code.

## 10. Apple silicon is in scope now, under a condition

The brief says "Must run on Linux distributions. Apple and Windows are not
relevant for this application." That was right for what the project was then: a
self-contained binary for a bare Linux box, validated on rented cloud hardware.
It stays in the brief above, because the brief is a record of what was asked
rather than a live specification.

**What changed is the evidence, not the goal.** Every ARM measurement this
project holds came from a virtual machine — Graviton3, Graviton4 and Grace are
all EC2 or cloud instances, which is why `virtualized` exists as a recorded
field at all. Apple silicon is a bare-metal ARM implementation from a fourth
vendor, with a wide vector unit and a core design that shares nothing with
Neoverse. On a benchmark whose entire thesis is that conclusions drawn from one
part do not survive the next one, declining that data costs more than it saves.

**The condition is that Linux pays nothing.** Concretely, three things, and a
port that cannot meet all three is not worth having:

- **No change to generated code on Linux.** The threading primitives this
  harness needs are not portable, so a port routes them through a seam; the
  seam has to be `static inline` and fold to the same instructions. This is
  checkable with the `objdump` comparison AGENTS.md already describes, and the
  bar is the hash loop being bit-identical rather than merely equivalent.
- **No weakening of the correctness gate.** A platform that cannot pin threads
  may skip `check-pinning`, but the skip must be conditioned on the platform,
  not on a flag the binary under test reports about itself. The gate exists
  because a pinning defect once under-reported throughput several-fold while
  reporting `verified`, and a gate the subject can switch off is not that gate.
- **A platform that cannot pin must say so in the data**, not merely measure
  noisier. `pinned_cpus` alone cannot distinguish "the pin was refused" from
  "there was nothing to ask", and those are different facts about a result.
  Whatever field carries that distinction has to reach the capture CSVs, or a
  cross-machine query cannot see it.

**Windows remains out of scope**, unchanged and for the original reason.

Finding 9 above is also now out of date in the direction that matters: the
AVX-512 path has executed on three parts, and validation hardware arrived.
