# CPU kernels

Every CPU kernel lives here. One file per ISA; each is a single translation unit
compiled with its own `-m` flags and selected at runtime. The device kernels are
in `../gpu/`, and the OpenCL driver that runs them in `src/opencl/`.

## How it fits together

`<alg>_kernel_impl.h` holds the algorithm — the round schedule, the stream
interleaving, the iteration loop, the checksum reduction — written exactly once.
It is deliberately free of any ISA knowledge. Every vector operation it performs
is a macro the including file supplies.

An ISA file therefore does two things:

1. defines the `OPS_*` macros for its vector type,
2. includes `instantiate_all.h` once.

`instantiate_<alg>.h` maps `OPS_*` onto the per-template names and includes the
template. The template `#undef`s its own macros on the way out, which is what
allows repeated inclusion; the `OPS_*` macros survive, so an ISA defines its
operations once and gets every algorithm at every stream count.

```
        scalar.c / sse2.c / avx2.c / avx512.c      <- OPS_* macros
                        |
                instantiate_all.h                  <- every alg x every stream
                        |
              instantiate_<alg>.h                  <- OPS_* -> <ALG>K_*
                        |
              <alg>_kernel_impl.h                  <- the algorithm
```

**`shani.c` is the exception**, and deliberately so — see below.

## How the matrix works

There are algorithms x ISAs x stream counts kernels. Each needs an
instantiation, a forward declaration and a registry row — all mechanically
derivable, so all three are expanded by the preprocessor from one table in
[matrix.h](matrix.h). Nothing is written out by hand and nothing can drift out
of sync.

```
        matrix.h                 the table: which ISAs, which algorithms
           |
    +------+---------------------------+
    |                                  |
instantiate_all.h                  registry.c
    |  (included once per ISA .c)       (VB_FOR_EACH_KERNEL -> decls + rows)
    |
instantiate_<alg>.h                OPS_* -> per-template macros
    |
<alg>_kernel_impl.h                the algorithm, written once
```

**Why ISA-major** — each `.c` carries every algorithm rather than each algorithm
carrying every ISA: a translation unit can only be compiled with one set of `-m`
flags, so an algorithm-major layout could not give AVX-512 code `-mavx512f`
while keeping it out of the SSE2 build. This is a constraint, not a preference.

## Adding an ISA

1. **Write `src/kernels/cpu/<name>.c`**: the `OPS_*` (32-bit) and `OPS64_*` (64-bit)
   operation sets, then two lines:

   ```c
   #define VB_ISA <name>
   #include "instantiate_all.h"
   ```

   That emits every algorithm at every stream count.

2. **Add a block to [matrix.h](matrix.h)** and a token to `VB_FOR_EACH_KERNEL`:

   ```c
   #if VB_HAVE_NEON
   #  define VB_ISA_NEON(M) VB_FOR_ALGS(M, neon, "NEON", vb_cpu_has_neon, 4, 2, NULL, NULL)
   #else
   #  define VB_ISA_NEON(M)
   #endif
   ```

   The four trailing columns come in pairs. The first pair is lanes per
   register at 32-bit and 64-bit; SHA-512 gets the narrow one. The second pair
   is `NULL` for any fixed-width ISA. A vector-length-agnostic ISA such as SVE
   passes lane counts of `0` instead and names two functions there, which
   `registry.c` calls once at startup -- see the SVE block in `matrix.h`.

3. **Add build flags** in the top-level `Makefile`: a `KFLAGS_<name>` line, the
   name in `KERNELS`, and a `HAVE_` probe if the compiler might not support it.

`registry.c` is not touched. `make check` then validates the new kernels against
the references across message sizes, block boundaries and iteration counts.

## Adding an ISA that only does some algorithms

An ISA block does not have to go through `VB_FOR_ALGS`. If the hardware supports
only one algorithm, name it directly:

```c
#if VB_HAVE_SHANI
#  define VB_ISA_SHANI(M) \
       VB_FOR_STREAMS4(M, sha1, shani, "SHA-NI", VB_ALG_SHA1, vb_cpu_has_sha_ni, 1, NULL)
#else
#  define VB_ISA_SHANI(M)
#endif
```

Such an ISA usually also needs its own template rather than the `OPS_*` set, for
the reason in the next section, and so does not include `instantiate_all.h`.

## The SHA-NI path, and why it is shaped differently

`shani.c` breaks the `OPS_*` pattern, and the break is informative rather than
accidental.

Every other ISA here is a *wider or more expressive general vector ALU*. They
differ in how many messages fit in a register and how few instructions a round
function costs, so one template plus a per-ISA operation set covers all of them.

SHA-NI is not that. `SHA1RNDS4` performs four real SHA-1 rounds; `SHA1MSG1` and
`SHA1MSG2` perform the message expansion. The template's whole job has been
absorbed into the instructions, so there are no primitives to hand it — hence
`sha1_ni_kernel_impl.h`. Two consequences follow:

- **`lanes = 1`.** The 128-bit register holds *one* message's state, not a
  vector of messages. All parallelism has to come from stream interleaving, so
  the stream sweep matters more here than anywhere else.
- **One algorithm only.** There is no MD5 or SHA-512 to be had at any price,
  which is ../../../docs/design.md finding 1 restated as a build constraint.

**What it measures is a fixed-function unit, not integer SIMD.** The number is
only meaningful as a ratio against `sha1/avx2`, and must never be quoted as an
integer-SIMD figure.

## Adding an algorithm

Longer than adding an ISA, because an algorithm is not only kernels. No ISA file
is touched, and neither is `registry.c`.

**The algorithm itself**

1. Add `VB_ALG_<NAME>` to the `vb_alg_id` enum in
   [include/algorithm.h](../../../include/algorithm.h), before `VB_ALG_COUNT`.
2. Write `src/reference/<alg>.c`, the scalar reference every kernel is
   validated against, and declare it in
   [include/hashes.h](../../../include/hashes.h). The Makefile links
   `src/reference/*.c` by wildcard, so the directory is the convention: a
   reference put anywhere else silently will not be built.
3. Add a descriptor row to `src/algorithm.c` — digest words and bytes, block
   size, word size, length field, endianness, round count, and the reference
   function. If the digest is wider than SHA-512's eight words, raise
   `VB_MAX_DIGEST_WORDS` too.

**The CPU kernels**

4. Write `<alg>_kernel_impl.h` (the template) and `instantiate_<alg>.h` (which
   maps `OPS_*` onto its macro names).
5. Add a block to `instantiate_all.h`.
6. Add one line to `VB_FOR_ALGS` in [matrix.h](matrix.h), passing the 32-bit
   or 64-bit lane arguments according to the algorithm's word size: a 64-bit
   algorithm takes the narrow lane count and its lane function.

**The device kernel** (optional, but the matrix expects one — see the OpenCL
section below for what each piece does)

7. Add `src/kernels/gpu/<alg>.cl`: the kernel, its constant tables, and its steps
   written out. The Makefile picks it up by wildcard and embeds it; there is
   nothing to commit and nothing to keep in sync.
8. Add one line to `VB_FOR_EACH_DEVICE_ALG` in [matrix.h](matrix.h). Both the
   `PROGRAMS` table in `src/opencl/backend.c` and the four registry rows in
   `src/registry.c` expand from it, so the device side has no hand-written
   tables to keep in sync.

**Proving it works**

9. Add known-answer vectors to `tests/test_hashes.c` from the algorithm's
    specification — not from another implementation, which would only prove the
    two agree. `make check` then validates every kernel against the reference
    across message sizes, block boundaries and iteration counts.

Nothing else needs touching. In particular `tools/sweep.py` learns the new
algorithm automatically: it reads the algorithm list, geometry and minimum
iteration length from `valubench --list --json` rather than carrying its own
copy.

## Rules that are not negotiable

**Write the round functions for your ISA, not for the abstract algorithm.**
This is why `OPS_F..OPS_I` are per-ISA macros rather than one shared definition.
AVX-512 collapses each of them to a single `vpternlogd`; SSE2 and AVX2 need
three ops. A kernel that inherits the "generic" form leaves that on the table.

The same logic runs in reverse. Sharing the `x^y` term between consecutive
round-3 steps — both use `H(x,y,z) = x^y^z`, so the pair can be computed once —
is a real saving on AVX2, and *pointless* on AVX-512 where `vpternlogd` already
does `x^y^z` in one instruction. Applying it everywhere would make the AVX-512
path slower. Optimise per ISA.

**Message words come from the corpus, never from constants.** Kernels read every
message word with `OPS_LOAD` from the lane-interleaved corpus. If a compiler
could prove any of those words constant it would fold them into the round
constants — the shortcut the `md5-full` workload excludes, because it inflates
the score by doing less work per hash. This is also why the build must never
enable `-flto`.

**The corpus is pre-padded and pre-transposed; do neither in the kernel.**
Padding is applied at corpus build time, so a kernel just hashes `blocks`
blocks. The layout puts the `LANES` copies of word *j* contiguously, so a plain
vector load gets them — doing a transpose inside the timed region would measure
shuffle throughput instead of hash throughput.

**Streams must be expanded, not looped.** The template repeats each step across
streams with explicit macro expansion rather than a loop over a literal bound. A
loop usually unrolls, but if it ever failed to, the kernel would silently
collapse to a single dependency chain and under-report the hardware by a large
factor — a wrong number that still looks plausible. See ../../../docs/research.md §2.4.

The *steps* are expanded the same way, and for the same reason, but they do not
have to be transcribed one per line: `S1K_RUN20` and `S5K_RUN16` emit a run of
steps with the index resolved by the preprocessor. That is still expansion —
collapsing SHA-1's and SHA-512's step lists this way left the generated machine
code byte-identical on every ISA, which is the check to repeat if you touch
them. MD5's steps stay written out, because each one carries a different word
index and rotation and the list reads against RFC 1321 line by line.

**Every ISA path stays in its own object.** Nothing outside your `.c` file may
see your `-m` flags, or the dispatcher itself could be compiled with
instructions the running CPU lacks and fault before it can check.

## The OpenCL kernels

`src/kernels/gpu/*.cl` are real OpenCL files — syntax highlighting, no escaping,
readable diffs — and they are **complete**: constants and every round step are
written out in the file. `tools/embed_cl.c` turns each into a byte array the
binary carries, so nothing has to be installed or located at runtime.

```
    <alg>.cl  --embed_cl-->  build/<alg>_kernel.h  --#include-->  backend.c
```

**The steps are written out, not looped.** A loop is excluded for the same
reason it is on the CPU: a compiler that declined to unroll would collapse the
streams into one dependency chain and under-report the device. The `.cl` file
expands the *stream* dimension through its `EACH` macro, and the *step*
dimension is the list of `EACH` lines itself.

Both were once produced at run time by an assembler inside
`src/opencl/backend.c`, then by a build-time generator. Neither earned its
place: the schedules are frozen standards, so a tool that recomputed them on
every build was machinery around a constant. What the move did earn was
`backend.c` having no algorithm knowledge at all, and that is worth keeping —
everything hash-specific on the device side is now in the `.cl` file.

**The embedded headers are not committed.** They land in `build/` and are
rebuilt whenever a `.cl` file is newer. The output is a pure function of its
input and `embed_cl` needs nothing but a C compiler, which the build already
requires, so a checked-in copy could only ever be a second source of truth to
keep in sync. It did not stay in sync: a `check-embed` target existed solely to
catch drift, and the headers were once found truncated in a working tree with a
green build behind them. A file that is always generated cannot disagree with
its source.

The cost is that a cross build has to run `embed_cl` on the *host*, so it is
built with `$(HOSTCC)` rather than `$(CC)`.

**Each constant table is written where it is used**, rather than shared from
one header — MD5's in `src/reference/md5.c` and `src/kernels/cpu/md5_kernel_impl.h`,
SHA-1's in `src/reference/sha1.c` and the two SHA-1 templates, and every device copy
in its own `.cl` file. The exception is `include/sha512_const.h`: eighty 64-bit
round constants are large enough that a second copy would be a liability rather
than a convenience.

They are frozen values, so the duplication cannot rot — and it cannot go
unnoticed either. Every kernel is validated against the scalar reference by
`make check`, so a wrong digit in a kernel fails there; a wrong digit in the
*reference* fails the RFC 1321 and FIPS 180-4 known-answer vectors in
`tests/test_hashes.c`, which are independent of both.

Adding a device algorithm is two things: the `.cl` file (the Makefile picks it
up by wildcard) and one line in `VB_FOR_EACH_DEVICE_ALG` in `matrix.h`, naming
its entry point and digest shape. The `PROGRAMS` table in `src/opencl/backend.c`
and the four registry rows both expand from that line, so neither is written by
hand. The digest shape is two numbers — words and bytes per word — because
SHA-512's partials are 64-bit; they size the readback, the work-group scratch
and the fold, all of which are otherwise algorithm-agnostic.

**Transfer mode.** By default the corpus is uploaded once at context setup and
every launch runs against resident data. `--transfer stream` re-uploads before
each launch so the host-to-device link sits inside the timed region, which is
what the goal 2 crossover measurement needs. Enabling it forces `repeats` to 1:
that knob amplifies compute without amplifying transfer, so any other value
would inflate exactly the side of the ratio being measured. Geometry tuning
always runs in resident mode, or it would be timing the upload.

**One stream is usually the right answer on a device**, unlike on the CPU. A GPU
already has thousands of work-items in flight, so interleaving hides no latency
that was not already hidden and only costs registers — `sha512/ocl` loses 2.1x
between one stream and two. Kernels are still registered at 1-4 so the harness
can measure it rather than assume it.

**Review changes in the `.cl` file, not the header** — the header is machine
output. It is a byte array rather than a string literal because C99 only
guarantees 4095-character string literals and the kernel exceeds that; an array
initialiser has no such limit and stays warning-clean under `-Wpedantic`.

## Current kernels

| File | ISA | Lanes | Notes |
|---|---|---|---|
| `scalar.c` | portable C | 1 | Fallback; builds anywhere |
| `sse2.c` | SSE2 | 4 | x86-64 baseline, no runtime check needed |
| `avx2.c` | AVX2 | 8 | No vector rotate, no 3-input logic |
| `avx512.c` | AVX-512F | 16 | `vpternlogd` + `vprold`; validated on Cascade Lake and Zen 5 |
| `shani.c` | SHA-NI | 1 | SHA-1 only; fixed-function, not SIMD |
| `neon.c` | ARM NEON | 4 | AArch64; `vbslq` gives a 3-input select at 128 bits |
| `sve.c` | ARM SVE | VLA | Lane count chosen by the hardware, 128-2048 bits; resolved at startup |
| `sve2.c` | ARM SVE2 | VLA | As SVE, plus three-input select, three-way XOR, xor-rotate and shift-right-insert |

Device kernels live beside this directory in `../gpu/`: `md5.cl`, `sha1.cl`
and `sha512.cl`. The host-side OpenCL driver that runs them -- context, upload,
launch, readback -- is `src/opencl/`, and knows nothing about hash functions.

Planned: ARM's SHA-1 extension. It would need its own template: its
instructions decompose the rounds differently from x86's, so
`sha1_ni_kernel_impl.h` does not carry over. The SVE kernels above do not share
the fixed-width structure either; they are vector-length agnostic, which is why
they register a lane function rather than a lane count. See
../../../docs/research.md §4.2.
